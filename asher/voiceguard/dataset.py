"""Dataset discovery, task labels, and recording-session-separated splits."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .audio import sha256_file
from .exceptions import DatasetError, ManifestError
from .recording import MANIFEST_FILENAME, load_manifest
from .schema import SampleCondition, SampleRecord, SessionManifest, SpeakerRole, TrainingTask


WAKE_POSITIVE = "wake_positive"
WAKE_NEGATIVE = "wake_negative"


@dataclass(frozen=True)
class DatasetSample:
    session_id: str
    speaker_id: str
    role: str
    environment: str
    session_directory: Path
    record: SampleRecord

    @property
    def wav_path(self) -> Path:
        root = self.session_directory.resolve()
        value = (root / Path(self.record.path)).resolve()
        if value != root and root not in value.parents:
            raise DatasetError("sample path escaped its recording session")
        return value

    def label_for(self, task: str | TrainingTask) -> str:
        selected = TrainingTask(task)
        if selected is TrainingTask.WAKE_WORD:
            return WAKE_POSITIVE if self.record.contains_wake_phrase else WAKE_NEGATIVE
        # Speaker authentication is an identity decision, not a role
        # decision. Roles are authorization metadata applied *after* this
        # label is mapped through the active user store.
        return self.speaker_id

    def expected_authorized_for(
        self,
        task: str | TrainingTask,
        authorized_labels: Sequence[str],
    ) -> bool:
        if self.record.expected_authorized is not None:
            return self.record.expected_authorized
        if self.record.condition == SampleCondition.REPLAY.value:
            return False
        return self.label_for(task) in set(authorized_labels)


@dataclass(frozen=True)
class VoiceDataset:
    root: Path
    sessions: tuple[SessionManifest, ...]
    samples: tuple[DatasetSample, ...]

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(session.session_id for session in self.sessions)

    @property
    def fingerprint(self) -> str:
        """Hash non-secret sample identities and content digests for reproducibility."""

        digest = hashlib.sha256()
        for sample in sorted(self.samples, key=lambda item: (item.session_id, item.record.sample_id)):
            fields = (sample.session_id, sample.record.sample_id, sample.record.sha256)
            digest.update("\0".join(fields).encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class FeatureExample:
    sample_id: str
    session_id: str
    label: str
    features: tuple[float, ...]
    expected_authorized: bool
    condition: str = SampleCondition.CLEAN.value

    def __post_init__(self) -> None:
        if not self.sample_id or not self.session_id or not self.label:
            raise DatasetError("feature examples require sample, session, and class identifiers")
        if not self.features:
            raise DatasetError("feature vectors cannot be empty")
        if self.condition not in {item.value for item in SampleCondition}:
            raise DatasetError("feature example has an unsupported evaluation condition")


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[FeatureExample, ...]
    validation: tuple[FeatureExample, ...]
    test: tuple[FeatureExample, ...]

    @property
    def train_sessions(self) -> frozenset[str]:
        return frozenset(item.session_id for item in self.train)

    @property
    def validation_sessions(self) -> frozenset[str]:
        return frozenset(item.session_id for item in self.validation)

    @property
    def test_sessions(self) -> frozenset[str]:
        return frozenset(item.session_id for item in self.test)

    def assert_session_separated(self) -> None:
        groups = (self.train_sessions, self.validation_sessions, self.test_sessions)
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise DatasetError("recording session leakage was detected across dataset partitions")


def load_dataset(
    recordings_root: str | Path,
    *,
    include_revoked: bool = False,
    verify_checksums: bool = True,
) -> VoiceDataset:
    """Load manifest-owned WAV samples without searching or printing private contents."""

    root = Path(recordings_root).resolve()
    if not root.exists():
        raise DatasetError("recordings directory does not exist")
    sessions: list[SessionManifest] = []
    samples: list[DatasetSample] = []
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_FILENAME}")):
        session_directory = manifest_path.parent
        resolved_session_directory = session_directory.resolve()
        if (
            session_directory.is_symlink()
            or resolved_session_directory.parent != root
        ):
            raise DatasetError("recording session path escaped the recordings directory")
        try:
            manifest = load_manifest(manifest_path)
        except ManifestError as exc:
            raise DatasetError(f"invalid recording manifest in session directory {manifest_path.parent.name}") from exc
        if session_directory.name != manifest.session_id:
            raise DatasetError("recording manifest does not match its session directory")
        if manifest.revoked and not include_revoked:
            continue
        sessions.append(manifest)
        for record in manifest.samples:
            item = DatasetSample(
                session_id=manifest.session_id,
                speaker_id=manifest.speaker_id,
                role=manifest.role,
                environment=manifest.environment,
                session_directory=session_directory,
                record=record,
            )
            path = item.wav_path
            if not path.is_file():
                raise DatasetError("a manifest-owned WAV sample is missing")
            if verify_checksums and sha256_file(path) != record.sha256:
                raise DatasetError("a manifest-owned WAV sample failed its integrity check")
            samples.append(item)
    return VoiceDataset(root=root, sessions=tuple(sessions), samples=tuple(samples))


def _partition_bucket(
    session_ids: list[str],
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    count = len(session_ids)
    if count == 1:
        return session_ids, [], []
    if count == 2:
        return session_ids[:1], session_ids[1:], []
    validation_count = max(1, int(round(count * validation_fraction)))
    test_count = max(1, int(round(count * test_fraction)))
    while validation_count + test_count >= count:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = count - validation_count - test_count
    return (
        session_ids[:train_count],
        session_ids[train_count : train_count + validation_count],
        session_ids[train_count + validation_count :],
    )


def session_separated_split(
    examples: Iterable[FeatureExample],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """Deterministically stratify whole recording sessions, never individual clips."""

    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise DatasetError("validation/test fractions must be positive and sum to less than one")
    items = tuple(examples)
    grouped: dict[str, list[FeatureExample]] = defaultdict(list)
    for item in items:
        grouped[item.session_id].append(item)
    if len(grouped) < 3:
        raise DatasetError("at least three recording sessions are required for train/validation/test separation")

    # A stratum describes all labels and authorization outcomes in a session.
    # This preserves positive/negative coverage when each speaker is recorded in
    # several sessions while still keeping mixed wake-word sessions intact.
    strata: dict[tuple[tuple[str, ...], tuple[bool, ...]], list[str]] = defaultdict(list)
    for session_id, values in grouped.items():
        key = (
            tuple(sorted({item.label for item in values})),
            tuple(sorted({item.expected_authorized for item in values})),
        )
        strata[key].append(session_id)

    train_ids: list[str] = []
    validation_ids: list[str] = []
    test_ids: list[str] = []
    generator = random.Random(seed)
    for key in sorted(strata, key=repr):
        session_ids = sorted(strata[key])
        generator.shuffle(session_ids)
        train, validation, test = _partition_bucket(
            session_ids,
            validation_fraction,
            test_fraction,
        )
        train_ids.extend(train)
        validation_ids.extend(validation)
        test_ids.extend(test)

    # If sparse strata left a global holdout empty, move only a non-essential
    # training session. Training later gives a clear error if this would remove
    # a class entirely instead of silently leaking samples.
    if not validation_ids and len(train_ids) >= 3:
        validation_ids.append(train_ids.pop())
    if not test_ids and len(train_ids) >= 2:
        test_ids.append(train_ids.pop())
    if not validation_ids or not test_ids:
        raise DatasetError(
            "record sessions across additional days/environments so both validation and test holdouts exist"
        )

    def collect(session_ids: Iterable[str]) -> tuple[FeatureExample, ...]:
        allowed = set(session_ids)
        return tuple(item for item in items if item.session_id in allowed)

    result = DatasetSplit(
        train=collect(train_ids),
        validation=collect(validation_ids),
        test=collect(test_ids),
    )
    result.assert_session_separated()
    return result
