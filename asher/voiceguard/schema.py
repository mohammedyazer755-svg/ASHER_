"""Serializable recording and dataset schemas for Asher VoiceGuard."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .exceptions import ManifestError


SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    """Return an unambiguous, JSON-friendly UTC timestamp."""

    return datetime.now(UTC).isoformat()


class SpeakerRole(str, Enum):
    OWNER = "owner"
    TRUSTED = "trusted"
    UNKNOWN = "unknown"


class SampleCondition(str, Enum):
    CLEAN = "clean"
    NOISY = "noisy"
    REPLAY = "replay"


class SampleOrigin(str, Enum):
    RECORDED = "recorded"
    IMPORTED = "imported"
    AUGMENTED = "augmented"


class TrainingTask(str, Enum):
    WAKE_WORD = "wake_word"
    SPEAKER_AUTH = "speaker_auth"


def _enum_value(value: str | Enum, enum_type: type[Enum], field_name: str) -> str:
    raw = value.value if isinstance(value, Enum) else str(value)
    try:
        return str(enum_type(raw).value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ManifestError(f"{field_name} must be one of: {allowed}") from exc


def _json_bool(value: Any, field_name: str) -> bool:
    """Accept JSON booleans only; strings such as ``"false"`` are invalid."""

    if not isinstance(value, bool):
        raise ManifestError(f"{field_name} must be a JSON boolean")
    return value


def _optional_json_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _json_bool(value, field_name)


def validate_relative_wav_path(value: str) -> str:
    """Validate and normalize a manifest-owned relative WAV path."""

    normalized = value.replace("\\", "/")
    if "\x00" in normalized or ":" in normalized:
        raise ManifestError("sample path contains an invalid drive or NUL marker")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError("sample path must stay inside its recording session")
    if path.suffix.lower() != ".wav":
        raise ManifestError("sample path must reference a .wav file")
    return path.as_posix()


@dataclass(frozen=True)
class SampleRecord:
    """One immutable PCM WAV sample described by a session manifest."""

    sample_id: str
    path: str
    sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int = 2
    contains_wake_phrase: bool = False
    condition: str = SampleCondition.CLEAN.value
    origin: str = SampleOrigin.RECORDED.value
    expected_authorized: bool | None = None
    source_sample_id: str | None = None
    created_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        if not self.sample_id or any(char in self.sample_id for char in "/\\"):
            raise ManifestError("sample_id must be a non-empty path-free identifier")
        object.__setattr__(self, "path", validate_relative_wav_path(self.path))
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256.lower()):
            raise ManifestError("sample sha256 must be a 64-character hexadecimal digest")
        if self.duration_seconds <= 0:
            raise ManifestError("sample duration must be greater than zero")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ManifestError("sample rate and channel count must be positive")
        if self.sample_width != 2:
            raise ManifestError("VoiceGuard manifests currently require 16-bit PCM WAV samples")
        object.__setattr__(
            self,
            "condition",
            _enum_value(self.condition, SampleCondition, "sample condition"),
        )
        object.__setattr__(self, "origin", _enum_value(self.origin, SampleOrigin, "sample origin"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "sha256": self.sha256,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "contains_wake_phrase": self.contains_wake_phrase,
            "condition": self.condition,
            "origin": self.origin,
            "expected_authorized": self.expected_authorized,
            "source_sample_id": self.source_sample_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleRecord":
        try:
            return cls(
                sample_id=str(value["sample_id"]),
                path=str(value["path"]),
                sha256=str(value["sha256"]).lower(),
                duration_seconds=float(value["duration_seconds"]),
                sample_rate=int(value["sample_rate"]),
                channels=int(value["channels"]),
                sample_width=int(value.get("sample_width", 2)),
                contains_wake_phrase=_json_bool(
                    value.get("contains_wake_phrase", False),
                    "contains_wake_phrase",
                ),
                condition=str(value.get("condition", SampleCondition.CLEAN.value)),
                origin=str(value.get("origin", SampleOrigin.RECORDED.value)),
                expected_authorized=_optional_json_bool(
                    value.get("expected_authorized"),
                    "expected_authorized",
                ),
                source_sample_id=(
                    None if value.get("source_sample_id") is None else str(value["source_sample_id"])
                ),
                created_at=str(value.get("created_at", "")) or utc_timestamp(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError("sample record is missing a required or valid field") from exc


@dataclass(frozen=True)
class SessionManifest:
    """A consented recording session; the unit used for leakage-free splits."""

    session_id: str
    speaker_id: str
    role: str
    environment: str
    consent_given_at: str
    created_at: str = field(default_factory=utc_timestamp)
    finalized_at: str | None = None
    revoked_at: str | None = None
    samples: tuple[SampleRecord, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported session manifest schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.session_id or any(char in self.session_id for char in "/\\"):
            raise ManifestError("session_id must be a non-empty path-free identifier")
        if not self.speaker_id.strip():
            raise ManifestError("speaker_id is required")
        if not self.environment.strip():
            raise ManifestError("recording environment is required")
        if not self.consent_given_at:
            raise ManifestError("recording consent timestamp is required")
        object.__setattr__(self, "role", _enum_value(self.role, SpeakerRole, "speaker role"))
        ids = [sample.sample_id for sample in self.samples]
        paths = [sample.path.casefold() for sample in self.samples]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ManifestError("sample identifiers and paths must be unique within a session")
        by_id = {sample.sample_id: sample for sample in self.samples}
        for sample in self.samples:
            if sample.origin != SampleOrigin.AUGMENTED.value:
                if sample.source_sample_id is not None:
                    raise ManifestError(
                        "only an augmented sample may reference a source recording"
                    )
                continue
            source = by_id.get(sample.source_sample_id or "")
            if source is None or source.origin == SampleOrigin.AUGMENTED.value:
                raise ManifestError(
                    "an augmented sample must reference a source recording in the same session"
                )
            if source.condition == SampleCondition.REPLAY.value:
                raise ManifestError("replay trials cannot be used as augmentation sources")
            if (
                sample.contains_wake_phrase != source.contains_wake_phrase
                or sample.expected_authorized != source.expected_authorized
            ):
                raise ManifestError(
                    "an augmented sample must preserve its source wake and authorization labels"
                )

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    @property
    def finalized(self) -> bool:
        return self.finalized_at is not None

    def add_sample(self, sample: SampleRecord) -> "SessionManifest":
        if self.finalized:
            raise ManifestError("cannot add audio to a finalized recording session")
        if self.revoked:
            raise ManifestError("cannot add audio to a revoked recording session")
        return replace(self, samples=(*self.samples, sample))

    def without_sample(self, sample_id: str) -> "SessionManifest":
        if self.finalized:
            raise ManifestError("cannot remove audio from a finalized recording session")
        return replace(self, samples=tuple(s for s in self.samples if s.sample_id != sample_id))

    def finalize(self, when: str | None = None) -> "SessionManifest":
        if self.finalized:
            return self
        if self.revoked:
            raise ManifestError("cannot finalize a revoked recording session")
        return replace(self, finalized_at=when or utc_timestamp())

    def revoke(self, when: str | None = None) -> "SessionManifest":
        return replace(self, revoked_at=when or utc_timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "role": self.role,
            "environment": self.environment,
            "consent_given_at": self.consent_given_at,
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
            "revoked_at": self.revoked_at,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SessionManifest":
        try:
            raw_samples = value.get("samples", [])
            if not isinstance(raw_samples, list):
                raise TypeError("samples must be a list")
            return cls(
                schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
                session_id=str(value["session_id"]),
                speaker_id=str(value["speaker_id"]),
                role=str(value["role"]),
                environment=str(value["environment"]),
                consent_given_at=str(value["consent_given_at"]),
                created_at=str(value.get("created_at", "")) or utc_timestamp(),
                finalized_at=(
                    None
                    if value.get("finalized_at") is None
                    else str(value["finalized_at"])
                ),
                revoked_at=None if value.get("revoked_at") is None else str(value["revoked_at"]),
                samples=tuple(SampleRecord.from_dict(item) for item in raw_samples),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError("session manifest is missing a required or valid field") from exc

    def resolve_sample_path(self, session_directory: Path, sample: SampleRecord) -> Path:
        """Resolve a sample while defending against manifest path traversal."""

        root = session_directory.resolve()
        candidate = (root / Path(sample.path)).resolve()
        if candidate != root and root not in candidate.parents:
            raise ManifestError("sample path escaped its recording session")
        return candidate
