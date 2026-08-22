"""Privacy-safe, dependency-free readiness checks for VoiceGuard training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from .dataset import DatasetSplit, FeatureExample, VoiceDataset, session_separated_split
from .exceptions import DatasetError
from .schema import SampleCondition, SampleOrigin, TrainingTask


@dataclass(frozen=True)
class ReadinessPolicy:
    """Structural minimums; these are not claims of biometric accuracy."""

    minimum_samples_per_session: int = 3
    minimum_sessions_per_class: int = 3
    minimum_session_separation_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        if self.minimum_samples_per_session <= 0:
            raise ValueError("minimum_samples_per_session must be positive")
        if self.minimum_sessions_per_class < 3:
            raise ValueError(
                "minimum_sessions_per_class must be at least three for train/validation/test separation"
            )
        if self.minimum_session_separation_seconds < 0:
            raise ValueError("minimum_session_separation_seconds cannot be negative")


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str
    details: Mapping[str, int | bool | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SplitCoverage:
    sample_count: int
    session_count: int
    class_count: int
    authorized_sample_count: int
    unauthorized_sample_count: int

    @property
    def has_authorization_coverage(self) -> bool:
        return self.authorized_sample_count > 0 and self.unauthorized_sample_count > 0

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "sample_count": self.sample_count,
            "session_count": self.session_count,
            "class_count": self.class_count,
            "authorized_sample_count": self.authorized_sample_count,
            "unauthorized_sample_count": self.unauthorized_sample_count,
            "has_authorization_coverage": self.has_authorization_coverage,
        }


@dataclass(frozen=True)
class TrainingReadiness:
    """An aggregate-only status object that never exposes IDs, paths, or digests."""

    task: str
    ready: bool
    sample_count: int
    session_count: int
    class_count: int
    authorized_class_count: int
    unauthorized_class_count: int
    authorized_sample_count: int
    unauthorized_sample_count: int
    minimum_samples_per_session: int
    minimum_sessions_per_class: int
    minimum_session_separation_seconds: float
    minimum_observed_samples_per_session: int
    minimum_observed_sessions_per_class: int
    unavailable_conditions: tuple[str, ...]
    split_coverage: Mapping[str, SplitCoverage]
    issues: tuple[ReadinessIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": self.task,
            "ready": self.ready,
            "sample_count": self.sample_count,
            "session_count": self.session_count,
            "class_count": self.class_count,
            "authorized_class_count": self.authorized_class_count,
            "unauthorized_class_count": self.unauthorized_class_count,
            "authorized_sample_count": self.authorized_sample_count,
            "unauthorized_sample_count": self.unauthorized_sample_count,
            "minimum_samples_per_session": self.minimum_samples_per_session,
            "minimum_sessions_per_class": self.minimum_sessions_per_class,
            "minimum_session_separation_seconds": self.minimum_session_separation_seconds,
            "minimum_observed_samples_per_session": self.minimum_observed_samples_per_session,
            "minimum_observed_sessions_per_class": self.minimum_observed_sessions_per_class,
            "unavailable_conditions": list(self.unavailable_conditions),
            "split_coverage": {
                name: coverage.to_dict() for name, coverage in self.split_coverage.items()
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def require_ready(self) -> None:
        if self.ready:
            return
        summary = "; ".join(issue.message for issue in self.issues)
        raise DatasetError(f"VoiceGuard training data is not ready: {summary}")


def _coverage(values: Sequence[FeatureExample]) -> SplitCoverage:
    return SplitCoverage(
        sample_count=len(values),
        session_count=len({item.session_id for item in values}),
        class_count=len({item.label for item in values}),
        authorized_sample_count=sum(item.expected_authorized for item in values),
        unauthorized_sample_count=sum(not item.expected_authorized for item in values),
    )


def _split_coverage(split: DatasetSplit) -> dict[str, SplitCoverage]:
    return {
        "train": _coverage(split.train),
        "validation": _coverage(split.validation),
        "test": _coverage(split.test),
    }


def _aware_timestamp(value: str) -> datetime | None:
    """Parse ISO timestamps without guessing a timezone or repairing metadata."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _normalized_environment(value: str) -> str:
    return " ".join(value.casefold().split())


def _independent_session_selection(
    dataset: VoiceDataset,
    *,
    labels_by_session: Mapping[str, set[str]],
    separation_seconds: float,
) -> tuple[frozenset[str], dict[str, int], int]:
    """Select a deterministic metadata-independent subset.

    A later session in the same normalized environment must be separated by the
    policy interval. A different declared environment is separate evidence.
    Correlated extras are excluded rather than allowed to leak across dataset
    partitions. This is only a collection-quality guard; metadata cannot prove
    physical independence.
    """

    parsed: dict[str, tuple[datetime, datetime, str]] = {}
    invalid_count = 0
    for manifest in dataset.sessions:
        created = _aware_timestamp(manifest.created_at)
        consented = _aware_timestamp(manifest.consent_given_at)
        sample_times = tuple(
            _aware_timestamp(sample.created_at) for sample in manifest.samples
        )
        environment = _normalized_environment(manifest.environment)
        if (
            created is None
            or consented is None
            or consented > created
            or any(value is None or value < created for value in sample_times)
            or not environment
        ):
            invalid_count += 1
            continue
        ended = max(
            (created, *(value for value in sample_times if value is not None))
        )
        parsed[manifest.session_id] = (created, ended, environment)

    selected_by_label: dict[str, set[str]] = {}
    all_labels = set().union(*labels_by_session.values()) if labels_by_session else set()
    for label in all_labels:
        candidates = sorted(
            (
                (session_id, *parsed[session_id])
                for session_id, labels in labels_by_session.items()
                if label in labels and session_id in parsed
            ),
            key=lambda item: (item[1], item[0]),
        )
        latest_by_environment: dict[str, datetime] = {}
        accepted_ids: set[str] = set()
        for session_id, created, ended, environment in candidates:
            previous = latest_by_environment.get(environment)
            if previous is None or (created - previous).total_seconds() >= separation_seconds:
                accepted_ids.add(session_id)
                latest_by_environment[environment] = ended
        selected_by_label[label] = accepted_ids

    # Mixed-label wake sessions are retained only when independently selected
    # for every label they contain. This prevents one label from smuggling a
    # correlated session into another label's held-out partition.
    selected_ids = frozenset(
        session_id
        for session_id, labels in labels_by_session.items()
        if session_id in parsed
        and labels
        and all(session_id in selected_by_label.get(label, set()) for label in labels)
    )
    counts = {
        label: sum(
            session_id in selected_ids and label in labels
            for session_id, labels in labels_by_session.items()
        )
        for label in all_labels
    }
    return selected_ids, counts, invalid_count


def independent_dataset_view(
    dataset: VoiceDataset,
    *,
    task: str | TrainingTask = TrainingTask.SPEAKER_AUTH,
    separation_seconds: float = 1_800.0,
) -> VoiceDataset:
    """Return the same deterministic independent-session view used by readiness."""

    selected_task = TrainingTask(task)
    labels_by_session: dict[str, set[str]] = defaultdict(set)
    for sample in dataset.samples:
        if sample.record.origin != SampleOrigin.AUGMENTED.value:
            labels_by_session[sample.session_id].add(sample.label_for(selected_task))
    selected_ids, _counts, _invalid = _independent_session_selection(
        dataset,
        labels_by_session=labels_by_session,
        separation_seconds=separation_seconds,
    )
    return VoiceDataset(
        root=dataset.root,
        sessions=tuple(
            session for session in dataset.sessions if session.session_id in selected_ids
        ),
        samples=tuple(
            sample for sample in dataset.samples if sample.session_id in selected_ids
        ),
    )


def assess_dataset_readiness(
    dataset: VoiceDataset,
    *,
    task: str | TrainingTask = TrainingTask.SPEAKER_AUTH,
    authorized_labels: Sequence[str] = (),
    policy: ReadinessPolicy | None = None,
    minimum_training_samples: int = 4,
    seed: int = 0,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> TrainingReadiness:
    """Assess structural trainability without extracting features or importing ML.

    The result deliberately contains aggregate counts from recorded/imported
    source clips only.  Augmented derivatives are training aids, not readiness
    or held-out evidence.  Readiness proves split and label coverage, not that
    a future biometric model will be accurate enough for any particular use.
    """

    selected_task = TrainingTask(task)
    selected_policy = policy or ReadinessPolicy()
    if minimum_training_samples <= 0:
        raise ValueError("minimum_training_samples must be positive")
    selected_authorized = tuple(
        dict.fromkeys(str(label) for label in authorized_labels if str(label))
    )
    authorized_set = set(selected_authorized)
    issues: list[ReadinessIssue] = []

    all_source_samples = tuple(
        sample
        for sample in dataset.samples
        if sample.record.origin != SampleOrigin.AUGMENTED.value
    )
    all_labels = tuple(sample.label_for(selected_task) for sample in all_source_samples)
    labels_by_session: dict[str, set[str]] = defaultdict(set)
    for sample, label in zip(all_source_samples, all_labels):
        labels_by_session[sample.session_id].add(label)
    selected_session_ids, independent_counts, invalid_independence_metadata = (
        _independent_session_selection(
            dataset,
            labels_by_session=labels_by_session,
            separation_seconds=selected_policy.minimum_session_separation_seconds,
        )
    )
    source_samples = tuple(
        sample for sample in all_source_samples if sample.session_id in selected_session_ids
    )
    labels = tuple(sample.label_for(selected_task) for sample in source_samples)
    classes = set(all_labels)
    non_replay_classes = {
        label
        for sample, label in zip(source_samples, labels)
        if sample.record.condition != SampleCondition.REPLAY.value
    }
    expected = tuple(
        sample.expected_authorized_for(selected_task, selected_authorized)
        for sample in source_samples
    )
    authorized_samples = sum(expected)
    unauthorized_samples = len(expected) - authorized_samples

    if not source_samples:
        issues.append(
            ReadinessIssue("no_samples", "No finalized active recording samples are available.")
        )
    if len(source_samples) < minimum_training_samples:
        issues.append(
            ReadinessIssue(
                "insufficient_samples",
                "The finalized dataset has fewer samples than the configured training minimum.",
                {"required": minimum_training_samples, "observed": len(source_samples)},
            )
        )
    if len(classes) < 2:
        issues.append(
            ReadinessIssue(
                "insufficient_classes",
                "At least two recorded classes are required.",
                {"required": 2, "observed": len(classes)},
            )
        )
    missing_authorized = authorized_set - classes
    if not authorized_set or missing_authorized:
        issues.append(
            ReadinessIssue(
                "authorized_labels_missing",
                "One or more configured authorized classes have no finalized recordings.",
                {"missing_class_count": len(missing_authorized) if authorized_set else 1},
            )
        )
    if authorized_samples == 0:
        issues.append(
            ReadinessIssue("no_authorized_samples", "No authorized calibration samples are available.")
        )
    if unauthorized_samples == 0:
        issues.append(
            ReadinessIssue(
                "no_unauthorized_samples", "No unauthorized calibration samples are available."
            )
        )
    if (
        selected_task is TrainingTask.SPEAKER_AUTH
        and not (non_replay_classes - authorized_set)
    ):
        issues.append(
            ReadinessIssue(
                "no_unauthorized_identity_class",
                "Speaker authentication needs a non-replay unauthorized identity class.",
            )
        )

    unique_digests_by_session: dict[str, set[str]] = defaultdict(set)
    for sample in all_source_samples:
        unique_digests_by_session[sample.session_id].add(sample.record.sha256)
    empty_or_short_sessions = sum(
        len(unique_digests_by_session.get(session.session_id, set()))
        < selected_policy.minimum_samples_per_session
        for session in dataset.sessions
    )
    minimum_observed_samples = (
        min(
            len(unique_digests_by_session.get(session.session_id, set()))
            for session in dataset.sessions
        )
        if dataset.sessions
        else 0
    )
    if empty_or_short_sessions:
        issues.append(
            ReadinessIssue(
                "insufficient_samples_per_session",
                "One or more finalized sessions have too few unique source clips.",
                {
                    "required_per_session": selected_policy.minimum_samples_per_session,
                    "affected_session_count": empty_or_short_sessions,
                },
            )
        )

    if invalid_independence_metadata:
        issues.append(
            ReadinessIssue(
                "invalid_session_independence_metadata",
                "One or more sessions lack valid timezone-aware collection metadata.",
                {"affected_session_count": invalid_independence_metadata},
            )
        )
    short_classes = sum(
        independent_counts.get(label, 0) < selected_policy.minimum_sessions_per_class
        for label in classes
    )
    minimum_observed_sessions = (
        min((independent_counts.get(label, 0) for label in classes), default=0)
    )
    if classes and short_classes:
        issues.append(
            ReadinessIssue(
                "insufficient_independent_sessions_per_class",
                "One or more classes need later or environment-distinct recording sessions.",
                {
                    "required_per_class": selected_policy.minimum_sessions_per_class,
                    "affected_class_count": short_classes,
                },
            )
        )

    digests_to_sessions: dict[str, set[str]] = defaultdict(set)
    for sample in dataset.samples:
        digests_to_sessions[sample.record.sha256].add(sample.session_id)
    duplicate_groups = sum(len(session_ids) > 1 for session_ids in digests_to_sessions.values())
    if duplicate_groups:
        issues.append(
            ReadinessIssue(
                "cross_session_duplicate_audio",
                "Identical audio occurs in more than one recording session.",
                {"duplicate_group_count": duplicate_groups},
            )
        )

    conflicts = 0
    for sample, label in zip(source_samples, labels):
        explicit = sample.record.expected_authorized
        if (
            selected_task is TrainingTask.SPEAKER_AUTH
            and sample.record.condition != SampleCondition.REPLAY.value
            and explicit is not None
            and explicit != (label in authorized_set)
        ):
            conflicts += 1
    if conflicts:
        issues.append(
            ReadinessIssue(
                "authorization_label_conflict",
                "Recorded authorization annotations conflict with configured classes.",
                {"affected_sample_count": conflicts},
            )
        )

    examples = tuple(
        FeatureExample(
            sample_id=f"readiness-{index}",
            session_id=sample.session_id,
            label=label,
            features=(0.0,),
            expected_authorized=authorized,
            condition=sample.record.condition,
            origin=sample.record.origin,
        )
        for index, (sample, label, authorized) in enumerate(
            zip(source_samples, labels, expected)
        )
    )
    split_coverage: dict[str, SplitCoverage] = {}
    try:
        split = session_separated_split(
            examples,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    except DatasetError:
        issues.append(
            ReadinessIssue(
                "split_unavailable",
                "Finalized sessions cannot yet form train, validation, and test partitions.",
            )
        )
    else:
        split.assert_session_separated()
        split_coverage = _split_coverage(split)
        for name, values in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            missing_count = len(classes - {item.label for item in values})
            if missing_count:
                issues.append(
                    ReadinessIssue(
                        f"{name}_class_coverage",
                        f"The {name} partition does not cover every recorded class.",
                        {"missing_class_count": missing_count},
                    )
                )
        for name in ("validation", "test"):
            coverage = split_coverage[name]
            if not coverage.has_authorization_coverage:
                issues.append(
                    ReadinessIssue(
                        f"{name}_authorization_coverage",
                        f"The {name} partition needs both authorized and unauthorized recordings.",
                    )
                )

    present_conditions = {sample.record.condition for sample in source_samples}
    unavailable_conditions = tuple(
        condition.value for condition in SampleCondition if condition.value not in present_conditions
    )
    return TrainingReadiness(
        task=selected_task.value,
        ready=not issues,
        sample_count=len(source_samples),
        session_count=len({sample.session_id for sample in source_samples}),
        class_count=len(classes),
        authorized_class_count=len(classes & authorized_set),
        unauthorized_class_count=len(classes - authorized_set),
        authorized_sample_count=authorized_samples,
        unauthorized_sample_count=unauthorized_samples,
        minimum_samples_per_session=selected_policy.minimum_samples_per_session,
        minimum_sessions_per_class=selected_policy.minimum_sessions_per_class,
        minimum_session_separation_seconds=selected_policy.minimum_session_separation_seconds,
        minimum_observed_samples_per_session=minimum_observed_samples,
        minimum_observed_sessions_per_class=minimum_observed_sessions,
        unavailable_conditions=unavailable_conditions,
        split_coverage=split_coverage,
        issues=tuple(issues),
    )


__all__ = [
    "ReadinessIssue",
    "ReadinessPolicy",
    "SplitCoverage",
    "TrainingReadiness",
    "assess_dataset_readiness",
    "independent_dataset_view",
]
