"""Portable calibrated VoiceGuard classifier models and inference results."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .exceptions import ModelError
from .features import FeatureExtractor, FeatureExtractorMetadata


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    predicted_label: str
    score: float
    threshold: float
    task: str
    reason: str
    security_notice: str = (
        "VoiceGuard is a convenience access signal; use device authentication for high-risk actions."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "predicted_label": self.predicted_label,
            "score": self.score,
            "threshold": self.threshold,
            "task": self.task,
            "reason": self.reason,
            "security_notice": self.security_notice,
        }


@dataclass(frozen=True)
class CalibratedVoiceGuardModel:
    """A JSON-serializable linear classifier with a held-out threshold."""

    task: str
    classes: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    threshold: float
    authorized_labels: tuple[str, ...]
    extractor_metadata: FeatureExtractorMetadata
    metadata: Mapping[str, Any] = field(default_factory=dict)
    binary_positive_class: str | None = None

    def __post_init__(self) -> None:
        if not self.classes or len(self.coefficients) != len(self.intercepts):
            raise ModelError("model classes, coefficients, and intercepts are inconsistent")
        if len(self.coefficients) != len(self.classes) and not (
            len(self.classes) == 2 and len(self.coefficients) == 1
        ):
            raise ModelError("model coefficient rows do not match model classes")
        dimensions = len(self.feature_mean)
        if dimensions == 0 or len(self.feature_scale) != dimensions:
            raise ModelError("model feature scaling metadata is invalid")
        if any(len(row) != dimensions for row in self.coefficients):
            raise ModelError("model coefficient dimensions are invalid")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.feature_scale):
            raise ModelError("model feature scales must be finite and positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ModelError("model threshold must be between 0 and 1")
        if not set(self.authorized_labels).issubset(set(self.classes)):
            raise ModelError("authorized labels must be present in model classes")
        if self.binary_positive_class is not None and self.binary_positive_class not in self.classes:
            raise ModelError("binary positive class must be present in model classes")

    @property
    def feature_dimension(self) -> int:
        return len(self.feature_mean)

    @property
    def model_version(self) -> str:
        value = self.metadata.get("model_version")
        return str(value) if value else "unknown"

    def _scaled(self, features: Sequence[float]) -> tuple[float, ...]:
        if len(features) != self.feature_dimension:
            raise ModelError(
                f"feature dimension mismatch: expected {self.feature_dimension}, received {len(features)}"
            )
        values = tuple(float(value) for value in features)
        if any(not math.isfinite(value) for value in values):
            raise ModelError("feature vectors must contain finite values")
        return tuple(
            (value - mean) / scale for value, mean, scale in zip(values, self.feature_mean, self.feature_scale)
        )

    def predict_proba(self, features: Sequence[float]) -> dict[str, float]:
        scaled = self._scaled(features)
        if len(self.classes) == 2 and len(self.coefficients) == 1:
            logit = self.intercepts[0] + sum(weight * value for weight, value in zip(self.coefficients[0], scaled))
            logit = max(-60.0, min(60.0, logit))
            positive = 1.0 / (1.0 + math.exp(-logit))
            positive_class = self.binary_positive_class or self.classes[-1]
            raw_by_class = {
                positive_class: positive,
                next(label for label in self.classes if label != positive_class): 1.0 - positive,
            }
            return {
                label: float(raw_by_class[label])
                for label in self.classes
            }
        else:
            logits = [
                intercept + sum(weight * value for weight, value in zip(row, scaled))
                for row, intercept in zip(self.coefficients, self.intercepts)
            ]
            maximum = max(logits)
            exponentials = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in logits]
            total = sum(exponentials)
            raw = tuple(value / total for value in exponentials)
        return {label: float(probability) for label, probability in zip(self.classes, raw)}

    def score_and_label(self, features: Sequence[float]) -> tuple[float, str, dict[str, float]]:
        probabilities = self.predict_proba(features)
        if self.task == "speaker_auth":
            # Speaker authentication is identity-first.  Classify exactly one
            # identity across every class, then authorize that same identity.
            # Summing several authorized probabilities can otherwise turn an
            # unknown argmax into an accepted (and potentially wrong) user.
            predicted = max(probabilities, key=probabilities.get)
            score = probabilities[predicted] if predicted in self.authorized_labels else 0.0
            return float(score), predicted, probabilities
        authorized = {label: probabilities[label] for label in self.authorized_labels}
        if authorized:
            score = sum(authorized.values()) if len(authorized) > 1 else next(iter(authorized.values()))
            best_authorized = max(authorized, key=authorized.get)
        else:
            score = max(probabilities.values())
            best_authorized = max(probabilities, key=probabilities.get)
        return score, best_authorized, probabilities

    def verify_features(self, features: Sequence[float]) -> VerificationResult:
        score, predicted_class, probabilities = self.score_and_label(features)
        accepted = (
            predicted_class in self.authorized_labels
            and score >= self.threshold
        )
        if accepted:
            predicted = predicted_class
            reason = "predicted identity is authorized and met the calibrated threshold"
        else:
            predicted = "unknown" if self.task == "speaker_auth" else max(
                probabilities,
                key=probabilities.get,
            )
            reason = (
                "predicted identity is not authorized"
                if predicted_class not in self.authorized_labels
                else "predicted identity confidence fell below the calibrated threshold"
            )
        return VerificationResult(
            accepted=accepted,
            predicted_label=predicted,
            score=float(score),
            threshold=self.threshold,
            task=self.task,
            reason=reason,
        )

    def verify_wav(self, path: str | Path, extractor: FeatureExtractor) -> VerificationResult:
        if extractor.metadata.extractor_id != self.extractor_metadata.extractor_id:
            raise ModelError("the supplied feature extractor does not match the trained model")
        return self.verify_features(extractor.extract_wav(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": self.task,
            "classes": list(self.classes),
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "threshold": self.threshold,
            "authorized_labels": list(self.authorized_labels),
            "extractor": self.extractor_metadata.to_dict(),
            "metadata": dict(self.metadata),
            "binary_positive_class": self.binary_positive_class,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibratedVoiceGuardModel":
        try:
            extractor_value = value["extractor"]
            if not isinstance(extractor_value, Mapping):
                raise TypeError("extractor metadata must be an object")
            extractor = FeatureExtractorMetadata(
                extractor_id=str(extractor_value["extractor_id"]),
                display_name=str(extractor_value["display_name"]),
                implementation=str(extractor_value["implementation"]),
                provenance=str(extractor_value["provenance"]),
                is_pretrained=bool(extractor_value["is_pretrained"]),
                is_student_trained=bool(extractor_value["is_student_trained"]),
                model_version=str(extractor_value["model_version"]),
                source=str(extractor_value["source"]),
                license=None if extractor_value.get("license") is None else str(extractor_value["license"]),
                details=dict(extractor_value.get("details", {})),
            )
            return cls(
                task=str(value["task"]),
                classes=tuple(str(item) for item in value["classes"]),
                coefficients=tuple(tuple(float(item) for item in row) for row in value["coefficients"]),
                intercepts=tuple(float(item) for item in value["intercepts"]),
                feature_mean=tuple(float(item) for item in value["feature_mean"]),
                feature_scale=tuple(float(item) for item in value["feature_scale"]),
                threshold=float(value["threshold"]),
                authorized_labels=tuple(str(item) for item in value["authorized_labels"]),
                extractor_metadata=extractor,
                metadata=dict(value.get("metadata", {})),
                binary_positive_class=(
                    None
                    if value.get("binary_positive_class") is None
                    else str(value["binary_positive_class"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError("saved VoiceGuard model metadata is invalid") from exc

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ModelError("could not persist VoiceGuard model") from exc
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedVoiceGuardModel":
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError("could not load VoiceGuard model") from exc
        if not isinstance(value, Mapping):
            raise ModelError("saved VoiceGuard model must be a JSON object")
        return cls.from_dict(value)


def model_fingerprint(payload: Mapping[str, Any]) -> str:
    """Stable model identity used in metadata, excluding timestamps."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_content_fingerprint(model: CalibratedVoiceGuardModel) -> str:
    """Hash inference fields plus stable training-data provenance."""

    dataset_metadata = model.metadata.get("dataset", {})
    dataset_fingerprint = (
        dataset_metadata.get("fingerprint")
        if isinstance(dataset_metadata, Mapping)
        else None
    )

    return model_fingerprint(
        {
            "task": model.task,
            "classes": list(model.classes),
            "coefficients": [list(row) for row in model.coefficients],
            "intercepts": list(model.intercepts),
            "feature_mean": list(model.feature_mean),
            "feature_scale": list(model.feature_scale),
            "threshold": model.threshold,
            "authorized_labels": list(model.authorized_labels),
            "binary_positive_class": model.binary_positive_class,
            "extractor": model.extractor_metadata.to_dict(),
            # A model bundle includes measured reports and provenance. Distinct
            # independent datasets must not collide merely because they happen
            # to fit identical weights.
            "dataset_fingerprint": dataset_fingerprint,
        }
    )
