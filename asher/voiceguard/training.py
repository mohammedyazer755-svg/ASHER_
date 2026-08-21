"""Session-aware classifier training and validation for VoiceGuard."""

from __future__ import annotations

import importlib
import math
import random
from dataclasses import dataclass, replace
from datetime import UTC, datetime
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
from .model import CalibratedVoiceGuardModel, model_fingerprint
from .schema import SampleCondition, TrainingTask


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
class TrainingResult:
    model: CalibratedVoiceGuardModel
    split: DatasetSplit
    validation_report: EvaluationReport
    test_report: EvaluationReport

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
            )
        )
    if not result:
        raise DatasetError("no non-revoked WAV samples are available for training")
    return tuple(result)


def _score_from_probabilities(
    probabilities: dict[str, float],
    authorized_labels: Sequence[str],
) -> tuple[float, str]:
    authorized = {label: probabilities.get(label, 0.0) for label in authorized_labels}
    if not authorized:
        label = max(probabilities, key=probabilities.get)
        return probabilities[label], label
    label = max(authorized, key=authorized.get)
    # For several authorized identities, total authorized mass is the
    # authentication score while argmax chooses the displayed identity.
    return sum(authorized.values()), label


def _observations(
    model: CalibratedVoiceGuardModel,
    examples: Sequence[FeatureExample],
) -> tuple[EvaluationObservation, ...]:
    output: list[EvaluationObservation] = []
    for item in examples:
        score, label, _ = model.score_and_label(item.features)
        output.append(
            EvaluationObservation(
                sample_id=item.sample_id,
                true_label=item.label,
                score=score,
                predicted_label=label if score >= model.threshold else "unknown",
                expected_authorized=item.expected_authorized,
                condition=item.condition,
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
        authorized_labels = selected.resolved_authorized_labels_for_dataset(dataset)
        selected = replace(selected, authorized_labels=authorized_labels)
        examples = feature_examples_from_dataset(
            dataset,
            self.extractor,
            task=selected.task,
            authorized_labels=authorized_labels,
        )
        return self.train_examples(examples, selected, dataset_fingerprint=dataset.fingerprint)

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
        if len(values) < selected.minimum_training_samples:
            raise DatasetError(
                f"at least {selected.minimum_training_samples} feature examples are required for training"
            )
        dimensions = len(values[0].features)
        if any(len(item.features) != dimensions for item in values):
            raise DatasetError("all feature vectors must have the same dimension")
        split = session_separated_split(
            values,
            validation_fraction=selected.validation_fraction,
            test_fraction=selected.test_fraction,
            seed=selected.seed,
        )
        split.assert_session_separated()
        train_labels = {item.label for item in split.train}
        all_labels = {item.label for item in values}
        missing = all_labels - train_labels
        if missing:
            raise DatasetError(
                "training partition is missing classes " + ", ".join(sorted(missing)) + "; record each class in multiple sessions"
            )
        if len(train_labels) < 2:
            raise DatasetError("at least two classes are required for a verifier classifier")
        if not set(authorized_labels).issubset(all_labels):
            raise DatasetError("configured authorized labels are absent from the dataset")

        numpy, standard_scaler_type, logistic_type = _require_ml()
        matrix = numpy.asarray([item.features for item in split.train], dtype=float)
        target = numpy.asarray([item.label for item in split.train], dtype=str)
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
                    "session_separation": True,
                },
                "security_limitations": (
                    "VoiceGuard is not sufficient proof for payments, account security, or other high-risk actions.",
                    "Replay resistance is not inferred without real replay recordings.",
                ),
            }
        )
        fingerprint_payload = {
            "task": task.value,
            "classes": classes,
            "coefficients": coefficients,
            "intercepts": intercepts,
            "threshold": threshold,
            "extractor": self.extractor.metadata.to_dict(),
            "dataset_fingerprint": dataset_fingerprint,
        }
        base_metadata["model_version"] = model_fingerprint(fingerprint_payload)
        model = CalibratedVoiceGuardModel(
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
        )
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
    """Train the owner/trusted/unknown classifier head on enrolled sessions."""

    selected = _task_config(
        config,
        task=TrainingTask.SPEAKER_AUTH,
        authorized_labels=("owner", "trusted"),
    )
    trainer = VoiceGuardTrainer(extractor=extractor)
    if isinstance(source, VoiceDataset):
        return trainer.train_dataset(source, selected)
    return trainer.train_examples(source, selected)
