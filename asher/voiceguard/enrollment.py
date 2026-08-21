"""Enrollment, revocation, and retraining lifecycle for VoiceGuard users."""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .dataset import load_dataset
from .exceptions import EnrollmentError, ManifestError
from .recording import RecordingSession, load_manifest
from .schema import SessionManifest, SpeakerRole
from .training import TrainingConfig, TrainingResult, VoiceGuardTrainer


REGISTRY_FILENAME = "enrollment_registry.json"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_registry(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class EnrollmentRecord:
    user_id: str
    role: str
    enrolled_at: str
    session_ids: tuple[str, ...]
    revoked_at: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "enrolled_at": self.enrolled_at,
            "session_ids": list(self.session_ids),
            "revoked_at": self.revoked_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnrollmentRecord":
        try:
            role = SpeakerRole(str(value["role"]))
            sessions = tuple(str(item) for item in value.get("session_ids", []))
            return cls(
                user_id=str(value["user_id"]),
                role=role.value,
                enrolled_at=str(value["enrolled_at"]),
                session_ids=sessions,
                revoked_at=None if value.get("revoked_at") is None else str(value["revoked_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EnrollmentError("enrollment registry contains an invalid user record") from exc


@dataclass(frozen=True)
class RevocationResult:
    user_id: str
    session_ids: tuple[str, ...]
    recordings_deleted: bool
    revoked_at: str


class EnrollmentManager:
    """Own the lifecycle of consented sessions and active user records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.recordings_root = self.root / "recordings"
        self.models_root = self.root / "models"
        self.registry_path = self.root / REGISTRY_FILENAME
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_registry(self) -> dict[str, EnrollmentRecord]:
        if not self.registry_path.exists():
            return {}
        try:
            with self.registry_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise EnrollmentError("could not read the local enrollment registry") from exc
        if not isinstance(value, Mapping):
            raise EnrollmentError("enrollment registry root must be an object")
        users = value.get("users", {})
        if not isinstance(users, Mapping):
            raise EnrollmentError("enrollment registry users must be an object")
        return {str(key): EnrollmentRecord.from_dict(item) for key, item in users.items()}

    def _write_registry(self, users: Mapping[str, EnrollmentRecord], *, active_model: str | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "updated_at": _timestamp(),
            "users": {key: value.to_dict() for key, value in users.items()},
        }
        if active_model is not None:
            payload["active_model"] = active_model
        elif self.registry_path.exists():
            try:
                with self.registry_path.open("r", encoding="utf-8") as stream:
                    previous = json.load(stream)
                if isinstance(previous, Mapping) and previous.get("active_model"):
                    payload["active_model"] = previous["active_model"]
            except (OSError, json.JSONDecodeError):
                pass
        _write_registry(self.registry_path, payload)

    def list_users(self, *, include_revoked: bool = False) -> tuple[EnrollmentRecord, ...]:
        users = self._read_registry()
        return tuple(
            sorted(
                (item for item in users.values() if include_revoked or item.active),
                key=lambda item: item.user_id,
            )
        )

    def begin_enrollment(
        self,
        user_id: str,
        *,
        role: str | SpeakerRole,
        environment: str,
        consent: bool,
    ) -> RecordingSession:
        """Start a new consented session; no raw audio is collected implicitly."""

        if not user_id.strip():
            raise EnrollmentError("user_id is required")
        try:
            selected_role = SpeakerRole(role)
        except ValueError as exc:
            raise EnrollmentError("role must be owner, trusted, or unknown") from exc
        return RecordingSession.create(
            self.recordings_root,
            speaker_id=user_id,
            role=selected_role,
            environment=environment,
            consent=consent,
        )

    def finalize_enrollment(self, session: RecordingSession, *, minimum_samples: int = 1) -> EnrollmentRecord:
        """Register a completed session only after checking its persisted manifest."""

        if minimum_samples <= 0:
            raise EnrollmentError("minimum_samples must be positive")
        manifest = load_manifest(session.directory)
        if manifest.session_id != session.manifest.session_id or manifest.revoked:
            raise EnrollmentError("session is missing, mismatched, or revoked")
        if len(manifest.samples) < minimum_samples:
            raise EnrollmentError("enrollment session does not contain enough recorded samples")
        with self._lock:
            users = self._read_registry()
            previous = users.get(manifest.speaker_id)
            session_ids = list(previous.session_ids) if previous and previous.active else []
            if manifest.session_id not in session_ids:
                session_ids.append(manifest.session_id)
            record = EnrollmentRecord(
                user_id=manifest.speaker_id,
                role=manifest.role,
                enrolled_at=previous.enrolled_at if previous and previous.active else _timestamp(),
                session_ids=tuple(session_ids),
                revoked_at=None,
            )
            users[manifest.speaker_id] = record
            self._write_registry(users)
            return record

    def revoke_user(self, user_id: str, *, delete_recordings: bool = False) -> RevocationResult:
        """Immediately disable a user; optional deletion is explicit and scoped."""

        with self._lock:
            users = self._read_registry()
            record = users.get(user_id)
            if record is None:
                raise EnrollmentError("user is not enrolled")
            when = _timestamp()
            sessions = tuple(record.session_ids)
            users[user_id] = EnrollmentRecord(
                user_id=record.user_id,
                role=record.role,
                enrolled_at=record.enrolled_at,
                session_ids=sessions,
                revoked_at=when,
            )
            # Mark each manifest before deleting anything so a partial operation
            # still prevents future training from using revoked audio.
            for session_id in sessions:
                directory = (self.recordings_root / session_id).resolve()
                if directory.parent != self.recordings_root.resolve():
                    raise EnrollmentError("registry session path escaped recordings root")
                if directory.is_dir():
                    try:
                        session = RecordingSession.open(directory)
                        session.revoke()
                    except ManifestError as exc:
                        raise EnrollmentError("could not revoke one of the user's recording sessions") from exc
            self._write_registry(users)
            if delete_recordings:
                for session_id in sessions:
                    directory = (self.recordings_root / session_id).resolve()
                    if directory.parent == self.recordings_root.resolve() and directory.is_dir():
                        shutil.rmtree(directory)
            return RevocationResult(
                user_id=user_id,
                session_ids=sessions,
                recordings_deleted=delete_recordings,
                revoked_at=when,
            )

    def retrain(
        self,
        *,
        trainer: VoiceGuardTrainer | None = None,
        config: TrainingConfig | None = None,
        model_path: str | Path | None = None,
    ) -> TrainingResult:
        """Retrain exclusively from currently non-revoked session manifests."""

        dataset = load_dataset(self.recordings_root, include_revoked=False)
        selected_trainer = trainer or VoiceGuardTrainer()
        result = selected_trainer.train_dataset(dataset, config)
        destination = Path(model_path) if model_path is not None else self.models_root / "voiceguard.json"
        result.model.save(destination)
        with self._lock:
            self._write_registry(self._read_registry(), active_model=str(destination.resolve()))
        return result

