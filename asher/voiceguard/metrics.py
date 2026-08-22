"""Measured VoiceGuard evaluation metrics with explicit missing-data handling."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import SampleCondition


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


@dataclass(frozen=True)
class EvaluationObservation:
    """One model score and ground-truth annotation used by the evaluator."""

    sample_id: str
    true_label: str
    score: float
    predicted_label: str
    expected_authorized: bool
    condition: str = SampleCondition.CLEAN.value
    acceptance_eligible: bool = True

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("evaluation observations require sample_id")
        if not math.isfinite(float(self.score)):
            raise ValueError("evaluation score must be finite")
        condition = self.condition.value if isinstance(self.condition, SampleCondition) else str(self.condition)
        object.__setattr__(self, "condition", condition)
        if condition not in {item.value for item in SampleCondition}:
            raise ValueError("unsupported evaluation condition")


@dataclass(frozen=True)
class ConditionMetrics:
    condition: str
    sample_count: int
    correct: int
    accuracy: float | None
    false_accept_rate: float | None
    false_reject_rate: float | None
    replay_acceptance_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "sample_count": self.sample_count,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "false_accept_rate": self.false_accept_rate,
            "false_reject_rate": self.false_reject_rate,
            "replay_acceptance_rate": self.replay_acceptance_rate,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """A report containing only metrics calculated from supplied observations."""

    threshold: float
    sample_count: int
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    binary_confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None
    false_accept_rate: float | None
    false_reject_rate: float | None
    authorized_identity_sample_count: int
    authorized_identity_error_count: int
    authorized_identity_accuracy: float | None
    condition_metrics: Mapping[str, ConditionMetrics]
    unavailable_conditions: tuple[str, ...]
    threshold_curve: tuple[Mapping[str, float | None], ...]
    measured: bool
    notes: tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def replay_acceptance_rate(self) -> float | None:
        item = self.condition_metrics.get(SampleCondition.REPLAY.value)
        return None if item is None else item.replay_acceptance_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "threshold": self.threshold,
            "sample_count": self.sample_count,
            "measured": self.measured,
            "labels": list(self.labels),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "binary_confusion_matrix": [list(row) for row in self.binary_confusion_matrix],
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "false_accept_rate": self.false_accept_rate,
            "false_reject_rate": self.false_reject_rate,
            "authorized_identity_sample_count": self.authorized_identity_sample_count,
            "authorized_identity_error_count": self.authorized_identity_error_count,
            "authorized_identity_accuracy": self.authorized_identity_accuracy,
            "replay_acceptance_rate": self.replay_acceptance_rate,
            "condition_metrics": {
                key: value.to_dict() for key, value in self.condition_metrics.items()
            },
            "unavailable_conditions": list(self.unavailable_conditions),
            "threshold_curve": [dict(item) for item in self.threshold_curve],
            "notes": list(self.notes),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(destination)
        return destination


def _binary_counts(observations: Sequence[EvaluationObservation], threshold: float) -> tuple[int, int, int, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for item in observations:
        accepted = item.acceptance_eligible and item.score >= threshold
        if item.expected_authorized and accepted:
            true_positive += 1
        elif item.expected_authorized and not accepted:
            false_negative += 1
        elif not item.expected_authorized and accepted:
            false_positive += 1
        else:
            true_negative += 1
    return true_negative, false_positive, false_negative, true_positive


def _condition_metrics(
    observations: Sequence[EvaluationObservation],
    threshold: float,
) -> ConditionMetrics:
    tn, fp, fn, tp = _binary_counts(observations, threshold)
    correct = sum(
        (item.acceptance_eligible and item.score >= threshold) == item.expected_authorized
        for item in observations
    )
    replay_rate = (
        _safe_ratio(
            sum(item.acceptance_eligible and item.score >= threshold for item in observations),
            len(observations),
        )
        if observations and observations[0].condition == SampleCondition.REPLAY.value
        else None
    )
    return ConditionMetrics(
        condition=observations[0].condition,
        sample_count=len(observations),
        correct=correct,
        accuracy=_safe_ratio(correct, len(observations)),
        false_accept_rate=_safe_ratio(fp, fp + tn),
        false_reject_rate=_safe_ratio(fn, fn + tp),
        replay_acceptance_rate=replay_rate,
    )


def _threshold_curve(observations: Sequence[EvaluationObservation]) -> tuple[Mapping[str, float | None], ...]:
    if not observations:
        return ()
    values = sorted({max(0.0, min(1.0, float(item.score))) for item in observations})
    candidates = sorted({0.0, 1.0, *values, *[(left + right) / 2 for left, right in zip(values, values[1:])]})
    curve: list[Mapping[str, float | None]] = []
    for threshold in candidates:
        tn, fp, fn, tp = _binary_counts(observations, threshold)
        curve.append(
            {
                "threshold": threshold,
                "false_accept_rate": _safe_ratio(fp, fp + tn),
                "false_reject_rate": _safe_ratio(fn, fn + tp),
            }
        )
    return tuple(curve)


def evaluate_predictions(
    observations: Iterable[EvaluationObservation],
    *,
    threshold: float,
) -> EvaluationReport:
    """Calculate confusion/F1/FAR/FRR and condition metrics from real labels.

    Empty or condition-missing subsets are represented as ``None`` and listed
    in ``unavailable_conditions``; no synthetic samples or guessed metrics are
    created.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("evaluation threshold must be between 0 and 1")
    items = tuple(observations)
    labels = tuple(sorted({item.true_label for item in items} | {item.predicted_label for item in items}))
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for item in items:
        matrix[index[item.true_label]][index[item.predicted_label]] += 1
    tn, fp, fn, tp = _binary_counts(items, threshold)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = _safe_ratio(tp + tn, len(items))
    authorized_identity_items = tuple(item for item in items if item.expected_authorized)
    authorized_identity_errors = sum(
        item.predicted_label != item.true_label
        for item in authorized_identity_items
    )
    authorized_identity_accuracy = _safe_ratio(
        len(authorized_identity_items) - authorized_identity_errors,
        len(authorized_identity_items),
    )

    by_condition: dict[str, list[EvaluationObservation]] = {item.value: [] for item in SampleCondition}
    for item in items:
        by_condition[item.condition].append(item)
    condition_results = {
        condition: _condition_metrics(values, threshold)
        for condition, values in by_condition.items()
        if values
    }
    unavailable = tuple(condition for condition, values in by_condition.items() if not values)
    notes: list[str] = []
    if not items:
        notes.append("No evaluation recordings were supplied; metrics are unavailable.")
    if SampleCondition.REPLAY.value in unavailable:
        notes.append("Replay acceptance was not measured because no real replay recordings were supplied.")
    if SampleCondition.NOISY.value in unavailable:
        notes.append("Noisy-condition accuracy was not measured because no tagged noisy recordings were supplied.")

    return EvaluationReport(
        threshold=threshold,
        sample_count=len(items),
        labels=labels,
        confusion_matrix=tuple(tuple(row) for row in matrix),
        binary_confusion_matrix=((tn, fp), (fn, tp)),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        false_accept_rate=_safe_ratio(fp, fp + tn),
        false_reject_rate=_safe_ratio(fn, fn + tp),
        authorized_identity_sample_count=len(authorized_identity_items),
        authorized_identity_error_count=authorized_identity_errors,
        authorized_identity_accuracy=authorized_identity_accuracy,
        condition_metrics=condition_results,
        unavailable_conditions=unavailable,
        threshold_curve=_threshold_curve(items),
        measured=bool(items),
        notes=tuple(notes),
    )


def observations_from_scores(
    *,
    sample_ids: Sequence[str],
    true_labels: Sequence[str],
    scores: Sequence[float],
    predicted_labels: Sequence[str],
    expected_authorized: Sequence[bool],
    conditions: Sequence[str] | None = None,
    acceptance_eligible: Sequence[bool] | None = None,
) -> tuple[EvaluationObservation, ...]:
    """Convenience constructor that validates aligned evaluation arrays."""

    lengths = {len(sample_ids), len(true_labels), len(scores), len(predicted_labels), len(expected_authorized)}
    if conditions is not None:
        lengths.add(len(conditions))
    if acceptance_eligible is not None:
        lengths.add(len(acceptance_eligible))
    if len(lengths) != 1:
        raise ValueError("evaluation arrays must have equal lengths")
    selected_conditions = conditions or [SampleCondition.CLEAN.value] * len(sample_ids)
    selected_eligibility = acceptance_eligible or [True] * len(sample_ids)
    return tuple(
        EvaluationObservation(
            sample_id=sample_ids[index],
            true_label=true_labels[index],
            score=float(scores[index]),
            predicted_label=predicted_labels[index],
            expected_authorized=bool(expected_authorized[index]),
            condition=selected_conditions[index],
            acceptance_eligible=bool(selected_eligibility[index]),
        )
        for index in range(len(sample_ids))
    )
