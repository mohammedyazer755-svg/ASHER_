"""Session-aware classifier training and validation for VoiceGuard."""

from __future__ import annotations

import importlib
import importlib.util
import math
import random
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .dataset import (
    DatasetSplit,
    FeatureExample,
    VoiceDataset,
    session_separated_split,
)
from .exceptions import DatasetError, MLDependencyError, ModelError
from .features import FeatureExtractor, StatisticalFeatureExtractor
from .metrics import EvaluationObservation, EvaluationReport, evaluate_predictions
from .model import CalibratedVoiceGuardModel, model_content_fingerprint
from .readiness import independent_dataset_view
from .schema import SampleCondition, SampleOrigin, TrainingTask


@dataclass(frozen=True)
class TrainingConfig:
    task: str = TrainingTask.SPEAKER_AUTH.value
    authorized_labels: tuple[str, ...] | None = None
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 0
    max_iterations: int = 1000
    minimum_training_samples: int = 4

    def resolved_authorized_labels(self) -> tuple[str, ...]:
        task = TrainingTask(self.task)
        if self.authorized_labels is not None:
            labels = tuple(dict.fromkeys(str(item) for item in self.authorized_labels))
        elif task is TrainingTask.WAKE_WORD:
            labels = ("wake_positive",)
        else:
            labels = ("owner", "trusted")
        if not labels:
            raise DatasetError("at least one authorized class is required")
        return labels

    def resolved_authorized_labels_for_dataset(self, dataset: VoiceDataset) -> tuple[str, ...]:
        """Resolve owner/trusted identities from the supplied private dataset."""

        if self.authorized_labels is not None or TrainingTask(self.task) is TrainingTask.WAKE_WORD:
            return self.resolved_authorized_labels()
        labels = tuple(
            sorted(
                {
                    sample.speaker_id
                    for sample in dataset.samples
                    if sample.role in {"owner", "trusted"}
                }
            )
        )
        if not labels:
            raise DatasetError("the dataset contains no owner/trusted speaker identities")
        return labels


@dataclass(frozen=True)
class TrainingArtifacts:
    """Private, versioned files persisted for an activated training run."""

    model_path: Path
    validation_report_path: Path
    test_report_path: Path


@dataclass(frozen=True)
class TrainingResult:
    model: CalibratedVoiceGuardModel
    split: DatasetSplit
    validation_report: EvaluationReport
    test_report: EvaluationReport
    artifacts: TrainingArtifacts | None = None

    @property
    def measured_test(self) -> bool:
        return self.test_report.measured


def ml_dependencies_available() -> tuple[bool, str | None]:
    """Report optional numerical dependencies without importing them eagerly."""

    missing = [name for name in ("numpy", "sklearn") if importlib.util.find_spec(name) is None]
    if missing:
        return False, "VoiceGuard training requires optional dependencies: " + ", ".join(missing)
    return True, None


def _require_ml() -> tuple[Any, Any, Any]:
    available, message = ml_dependencies_available()
    if not available:
        raise MLDependencyError(message or "optional numerical dependencies are unavailable")
    try:
        numpy = importlib.import_module("numpy")
        preprocessing = importlib.import_module("sklearn.preprocessing")
        linear_model = importlib.import_module("sklearn.linear_model")
    except (ImportError, OSError) as exc:
        raise MLDependencyError(
            "VoiceGuard training requires optional numpy and scikit-learn dependencies"
        ) from exc
    return numpy, preprocessing.StandardScaler, linear_model.LogisticRegression


def feature_examples_from_dataset(
    dataset: VoiceDataset,
    extractor: FeatureExtractor | None = None,
    *,
    task: str | TrainingTask = TrainingTask.SPEAKER_AUTH,
    authorized_labels: Sequence[str] | None = None,
) -> tuple[FeatureExample, ...]:
    selected_task = TrainingTask(task)
    selected_extractor = extractor or StatisticalFeatureExtractor()
    if authorized_labels is None:
        labels = (
            ("wake_positive",)
            if selected_task is TrainingTask.WAKE_WORD
            else tuple(
                sorted(
                    {
                        sample.speaker_id
                        for sample in dataset.samples
                        if sample.role in {"owner", "trusted"}
                    }
                )
            )
        )
    else:
        labels = tuple(authorized_labels)
    if not labels:
        raise DatasetError("at least one authorized identity is required")
    result: list[FeatureExample] = []
    for sample in dataset.samples:
        features = tuple(float(value) for value in selected_extractor.extract_wav(sample.wav_path))
        result.append(
            FeatureExample(
                sample_id=sample.record.sample_id,
                session_id=sample.session_id,
                label=sample.label_for(selected_task),
                features=features,
                expected_authorized=sample.expected_authorized_for(selected_task, labels),
                condition=sample.record.condition,
                origin=sample.record.origin,
                source_sample_id=sample.record.source_sample_id,
            )
        )
    if not result:
        raise DatasetError("no non-revoked WAV samples are available for training")
    return tuple(result)


def _observations(
    model: CalibratedVoiceGuardModel,
    examples: Sequence[FeatureExample],
) -> tuple[EvaluationObservation, ...]:
    output: list[EvaluationObservation] = []
    for item in examples:
        score, label, _ = model.score_and_label(item.features)
        acceptance_eligible = label in model.authorized_labels
        output.append(
            EvaluationObservation(
                sample_id=item.sample_id,
                true_label=item.label,
                score=score,
                predicted_label=(
                    label
                    if acceptance_eligible and score >= model.threshold
                    else "unknown"
                ),
                expected_authorized=item.expected_authorized,
                condition=item.condition,
                acceptance_eligible=acceptance_eligible,
            )
        )
    return tuple(output)


def _calibrate_threshold(
    provisional: CalibratedVoiceGuardModel,
    validation: Sequence[FeatureExample],
) -> tuple[float, dict[str, Any]]:
    observations = _observations(provisional, validation)
    if not observations:
        raise DatasetError("validation partition is empty; threshold cannot be calibrated")
    if not any(item.expected_authorized for item in observations) or not any(
        not item.expected_authorized for item in observations
    ):
        raise DatasetError(
            "validation sessions must include both authorized and unauthorized recordings for calibration"
        )
    scores = sorted({max(0.0, min(1.0, item.score)) for item in observations})
    candidates = sorted({0.0, 1.0, *scores, *[(left + right) / 2 for left, right in zip(scores, scores[1:])]})

    def objective(threshold: float) -> tuple[float, float, float]:
        report = evaluate_predictions(observations, threshold=threshold)
        far = report.false_accept_rate if report.false_accept_rate is not None else 1.0
        frr = report.false_reject_rate if report.false_reject_rate is not None else 1.0
        # Equal-error calibration, then prefer the safer (lower FAR) threshold.
        return far + frr, far, -threshold

    threshold = min(candidates, key=objective)
    report = evaluate_predictions(observations, threshold=threshold)
    return threshold, {
        "method": "held_out_equal_error_search",
        "validation_sample_count": len(validation),
        "validation_false_accept_rate": report.false_accept_rate,
        "validation_false_reject_rate": report.false_reject_rate,
        "validation_f1": report.f1,
    }


class VoiceGuardTrainer:
    """Train and calibrate a small personalized classifier head."""

    def __init__(self, extractor: FeatureExtractor | None = None) -> None:
        self.extractor = extractor or StatisticalFeatureExtractor()

    def train_dataset(self, dataset: VoiceDataset, config: TrainingConfig | None = None) -> TrainingResult:
        selected = config or TrainingConfig()
        independent = independent_dataset_view(
            dataset,
            task=selected.task,
        )
        authorized_labels = selected.resolved_authorized_labels_for_dataset(independent)
        selected = replace(selected, authorized_labels=authorized_labels)
        examples = feature_examples_from_dataset(
            independent,
            self.extractor,
            task=selected.task,
            authorized_labels=authorized_labels,
        )
        return self.train_examples(
            examples,
            selected,
            dataset_fingerprint=independent.fingerprint,
        )

    def train_examples(
        self,
        examples: Iterable[FeatureExample],
        config: TrainingConfig | None = None,
        *,
        dataset_fingerprint: str | None = None,
    ) -> TrainingResult:
        selected = config or TrainingConfig()
        task = TrainingTask(selected.task)
        authorized_labels = selected.resolved_authorized_labels()
        values = tuple(examples)
        source_values = tuple(item for item in values if not item.is_augmented)
        fit_source_values = tuple(
            item
            for item in source_values
            if item.condition != SampleCondition.REPLAY.value
        )
        if len(fit_source_values) < selected.minimum_training_samples:
            raise DatasetError(
                f"at least {selected.minimum_training_samples} non-replay source examples are required for training"
            )
        dimensions = len(values[0].features)
        if any(len(item.features) != dimensions for item in values):
            raise DatasetError("all feature vectors must have the same dimension")
        source_index = {
            (item.session_id, item.sample_id): item
            for item in source_values
        }
        if len(source_index) != len(source_values):
            raise DatasetError("source feature sample identifiers must be unique within each session")
        for item in values:
            if not item.is_augmented:
                continue
            source = source_index.get((item.session_id, item.source_sample_id or ""))
            if source is None:
                raise DatasetError(
                    "an augmented example must reference a source example in the same session"
                )
            if source.condition == SampleCondition.REPLAY.value:
                raise DatasetError("replay trials cannot be used as augmentation sources")
            if (
                item.label != source.label
                or item.expected_authorized != source.expected_authorized
            ):
                raise DatasetError(
                    "an augmented example must preserve its source classifier and authorization labels"
                )

        source_labels = {item.label for item in fit_source_values}
        if {item.label for item in values if item.condition != SampleCondition.REPLAY.value} - source_labels:
            raise DatasetError("augmented examples cannot introduce a classifier class")
        authorized_set = set(authorized_labels)
        if not authorized_set.issubset(source_labels):
            raise DatasetError("one or more configured authorized classes lack source recordings")
        if task is TrainingTask.SPEAKER_AUTH and not (source_labels - authorized_set):
            raise DatasetError(
                "speaker authentication requires at least one non-replay unauthorized identity class"
            )

        source_split = session_separated_split(
            source_values,
            validation_fraction=selected.validation_fraction,
            test_fraction=selected.test_fraction,
            seed=selected.seed,
        )
        # Split only source recordings.  Derivatives follow their source
        # session into training, but derivatives belonging to held-out
        # sessions are excluded from calibration and evaluation.
        split = DatasetSplit(
            train=tuple(
                item for item in values if item.session_id in source_split.train_sessions
            ),
            validation=source_split.validation,
            test=source_split.test,
        )
        split.assert_session_separated()
        fit_train = tuple(
            item
            for item in split.train
            if item.condition != SampleCondition.REPLAY.value
        )
        train_labels = {item.label for item in fit_train}
        all_labels = source_labels
        missing = all_labels - train_labels
        if missing:
            raise DatasetError(
                "the training partition is missing one or more classifier classes; "
                "record every class in multiple independent sessions"
            )
        if len(train_labels) < 2:
            raise DatasetError("at least two classes are required for a verifier classifier")

        numpy, standard_scaler_type, logistic_type = _require_ml()
        matrix = numpy.asarray([item.features for item in fit_train], dtype=float)
        target = numpy.asarray([item.label for item in fit_train], dtype=str)
        scaler = standard_scaler_type()
        scaled = scaler.fit_transform(matrix)
        try:
            classifier = logistic_type(
                max_iter=selected.max_iterations,
                random_state=selected.seed,
                solver="lbfgs",
            )
            classifier.fit(scaled, target)
        except Exception as exc:
            raise DatasetError("scikit-learn could not fit the VoiceGuard classifier on these sessions") from exc

        classes = tuple(str(item) for item in classifier.classes_.tolist())
        raw_coefficients = classifier.coef_.tolist()
        raw_intercepts = classifier.intercept_.tolist()
        coefficients = tuple(tuple(float(value) for value in row) for row in raw_coefficients)
        intercepts = tuple(float(value) for value in raw_intercepts)
        binary_positive_class = (
            classes[-1]
            if len(classes) == 2 and len(coefficients) == 1
            else None
        )
        provisional_metadata = {
            "schema_version": 1,
            "task": task.value,
            "classes": list(classes),
            "authorized_labels": list(authorized_labels),
            "classifier": {
                "name": "scikit-learn LogisticRegression",
                "student_trained": True,
                "solver": "lbfgs",
            },
            "dataset": {
                "sample_count": len(values),
                "source_sample_count": len(source_values),
                "augmented_sample_count": len(values) - len(source_values),
                "fit_eligible_source_sample_count": len(fit_source_values),
                "session_count": len({item.session_id for item in values}),
                "fingerprint": dataset_fingerprint,
            },
        }
        provisional = CalibratedVoiceGuardModel(
            task=task.value,
            classes=classes,
            coefficients=coefficients,
            intercepts=intercepts,
            feature_mean=tuple(float(value) for value in scaler.mean_.tolist()),
            feature_scale=tuple(float(value) for value in scaler.scale_.tolist()),
            threshold=0.5,
            authorized_labels=authorized_labels,
            extractor_metadata=self.extractor.metadata,
            metadata=provisional_metadata,
            binary_positive_class=binary_positive_class,
        )
        threshold, calibration = _calibrate_threshold(provisional, split.validation)
        base_metadata = dict(provisional_metadata)
        base_metadata.update(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "calibration": calibration,
                "split": {
                    "train_session_count": len(split.train_sessions),
                    "validation_session_count": len(split.validation_sessions),
                    "test_session_count": len(split.test_sessions),
                    "augmented_train_sample_count": sum(
                        item.origin == SampleOrigin.AUGMENTED.value for item in fit_train
                    ),
                    "fit_train_sample_count": len(fit_train),
                    "excluded_replay_train_sample_count": sum(
                        item.condition == SampleCondition.REPLAY.value for item in split.train
                    ),
                    "held_out_source_only": True,
                    "session_separation": True,
                },
                "security_limitations": (
                    "VoiceGuard is not sufficient proof for payments, account security, or other high-risk actions.",
                    "Replay resistance is not inferred without real replay recordings.",
                    "Augmented derivatives are training-only and are excluded from held-out metrics.",
                ),
            }
        )
        unversioned_model = CalibratedVoiceGuardModel(
            task=task.value,
            classes=classes,
            coefficients=coefficients,
            intercepts=intercepts,
            feature_mean=provisional.feature_mean,
            feature_scale=provisional.feature_scale,
            threshold=threshold,
            authorized_labels=authorized_labels,
            extractor_metadata=self.extractor.metadata,
            metadata=base_metadata,
            binary_positive_class=binary_positive_class,
        )
        base_metadata["model_version"] = model_content_fingerprint(unversioned_model)
        model = replace(unversioned_model, metadata=base_metadata)
        validation_report = evaluate_predictions(_observations(model, split.validation), threshold=threshold)
        test_report = evaluate_predictions(_observations(model, split.test), threshold=threshold)
        return TrainingResult(
            model=model,
            split=split,
            validation_report=validation_report,
            test_report=test_report,
        )


def train_from_feature_examples(
    examples: Iterable[FeatureExample],
    *,
    config: TrainingConfig | None = None,
    extractor: FeatureExtractor | None = None,
) -> TrainingResult:
    return VoiceGuardTrainer(extractor=extractor).train_examples(examples, config)


def prepare_training_examples(
    dataset: VoiceDataset,
    extractor: FeatureExtractor | None = None,
    *,
    task: str | TrainingTask = TrainingTask.SPEAKER_AUTH,
    authorized_labels: Sequence[str] | None = None,
) -> tuple[FeatureExample, ...]:
    """Public preparation entry point shared by UI and batch workflows."""

    return feature_examples_from_dataset(
        dataset,
        extractor,
        task=task,
        authorized_labels=authorized_labels,
    )


def prepare_positive_negative_examples(
    dataset: VoiceDataset,
    extractor: FeatureExtractor | None = None,
) -> tuple[FeatureExample, ...]:
    """Prepare real wake-phrase positives and negatives.

    The returned examples retain their session IDs and condition tags. No
    synthetic negative or replay result is invented; callers must collect or
    import those recordings explicitly.
    """

    return feature_examples_from_dataset(
        dataset,
        extractor,
        task=TrainingTask.WAKE_WORD,
        authorized_labels=("wake_positive",),
    )


def _task_config(
    config: TrainingConfig | None,
    *,
    task: TrainingTask,
    authorized_labels: tuple[str, ...],
) -> TrainingConfig:
    base = config or TrainingConfig()
    return replace(
        base,
        task=task.value,
        authorized_labels=base.authorized_labels or authorized_labels,
    )


def train_wake_verifier(
    source: VoiceDataset | Iterable[FeatureExample],
    *,
    extractor: FeatureExtractor | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train the personalized ``Hey Asher`` positive/negative verifier."""

    selected = _task_config(
        config,
        task=TrainingTask.WAKE_WORD,
        authorized_labels=("wake_positive",),
    )
    trainer = VoiceGuardTrainer(extractor=extractor)
    if isinstance(source, VoiceDataset):
        return trainer.train_dataset(source, selected)
    return trainer.train_examples(source, selected)


def train_speaker_classifier(
    source: VoiceDataset | Iterable[FeatureExample],
    *,
    extractor: FeatureExtractor | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train an identity-labelled classifier from enrolled speaker sessions."""

    base = config or TrainingConfig()
    selected = replace(base, task=TrainingTask.SPEAKER_AUTH.value)
    trainer = VoiceGuardTrainer(extractor=extractor)
    if isinstance(source, VoiceDataset):
        # Dataset speaker-auth labels are stable user IDs.  Leaving the
        # default unset lets train_dataset resolve the enrolled owner/trusted
        # identities instead of incorrectly treating role names as classes.
        return trainer.train_dataset(source, selected)
    values = tuple(source)
    if selected.authorized_labels is None:
        inferred = tuple(
            sorted({item.label for item in values if item.expected_authorized})
        )
        if not inferred:
            raise DatasetError(
                "speaker feature examples require an explicit or annotated authorized identity"
            )
        selected = replace(selected, authorized_labels=inferred)
    return trainer.train_examples(values, selected)
