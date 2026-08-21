"""Non-private speech fixtures and measured accuracy/latency metrics."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .vocabulary import comparison_key


@dataclass(frozen=True)
class VoiceFixture:
    fixture_id: str
    expected_transcript: str
    expected_intent: str
    expected_contact: str | None = None
    audio_path: str | None = None
    condition: str = "synthetic-text"


@dataclass(frozen=True)
class VoicePrediction:
    fixture_id: str
    transcript: str
    intent: str | None = None
    contact: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class VoiceEvaluationReport:
    sample_count: int
    total_reference_words: int
    total_word_errors: int
    word_error_rate: float | None
    contact_case_count: int
    contact_accuracy: float | None
    intent_case_count: int
    intent_accuracy: float | None
    latency_sample_count: int
    median_latency_ms: float | None
    p95_latency_ms: float | None
    missing_prediction_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _words(value: str) -> tuple[str, ...]:
    return tuple(str(value).casefold().split())


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return Levenshtein word errors and reference word count."""

    expected = _words(reference)
    actual = _words(hypothesis)
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1], len(expected)


def word_error_rate(reference: str, hypothesis: str) -> float:
    errors, count = word_error_counts(reference, hypothesis)
    if count == 0:
        return 0.0 if not _words(hypothesis) else 1.0
    return errors / count


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def evaluate_predictions(
    fixtures: Iterable[VoiceFixture],
    predictions: Iterable[VoicePrediction],
) -> VoiceEvaluationReport:
    fixture_list = tuple(fixtures)
    prediction_by_id = {item.fixture_id: item for item in predictions}
    total_errors = 0
    total_words = 0
    contact_cases = 0
    contact_correct = 0
    intent_cases = 0
    intent_correct = 0
    latencies: list[float] = []
    missing = 0
    for fixture in fixture_list:
        prediction = prediction_by_id.get(fixture.fixture_id)
        if prediction is None:
            missing += 1
            errors, words = word_error_counts(fixture.expected_transcript, "")
            total_errors += errors
            total_words += words
            if fixture.expected_contact is not None:
                contact_cases += 1
            if fixture.expected_intent:
                intent_cases += 1
            continue
        errors, words = word_error_counts(
            fixture.expected_transcript,
            prediction.transcript,
        )
        total_errors += errors
        total_words += words
        if fixture.expected_contact is not None:
            contact_cases += 1
            if prediction.contact is not None and comparison_key(
                prediction.contact
            ) == comparison_key(fixture.expected_contact):
                contact_correct += 1
        if fixture.expected_intent:
            intent_cases += 1
            if prediction.intent == fixture.expected_intent:
                intent_correct += 1
        if prediction.latency_ms is not None and math.isfinite(prediction.latency_ms):
            latencies.append(max(0.0, float(prediction.latency_ms)))
    return VoiceEvaluationReport(
        sample_count=len(fixture_list),
        total_reference_words=total_words,
        total_word_errors=total_errors,
        word_error_rate=(total_errors / total_words if total_words else None),
        contact_case_count=contact_cases,
        contact_accuracy=(contact_correct / contact_cases if contact_cases else None),
        intent_case_count=intent_cases,
        intent_accuracy=(intent_correct / intent_cases if intent_cases else None),
        latency_sample_count=len(latencies),
        median_latency_ms=(statistics.median(latencies) if latencies else None),
        p95_latency_ms=_percentile(latencies, 0.95),
        missing_prediction_count=missing,
    )


def generate_non_private_fixtures(count: int = 100) -> tuple[VoiceFixture, ...]:
    """Generate a deterministic text corpus containing no user contact data."""

    if count < 1:
        raise ValueError("count must be positive")
    contacts = ("Avery Stone", "Jordan Reed", "Morgan Vale", "Riley North")
    applications = ("Notepad", "Calculator", "Web Browser", "Text Editor")
    templates: list[tuple[str, str, str | None]] = []
    for contact in contacts:
        spelled = " ".join(contact.split()[0].upper())
        hyphenated = "-".join(contact.split()[0].upper())
        templates.extend(
            [
                (f"search {contact}", "whatsapp_search", contact),
                (f"search {spelled}", "whatsapp_search", contact),
                (f"search {hyphenated}", "whatsapp_search", contact),
                (f"send the project update to {contact}", "send_whatsapp", contact),
            ]
        )
    for application in applications:
        templates.extend(
            [
                (f"open {application}", "open_app", None),
                (f"close {application}", "close_app", None),
            ]
        )
    templates.extend(
        [
            ("increase volume", "volume_up", None),
            ("decrease volume", "volume_down", None),
            ("toggle mute", "toggle_mute", None),
            ("take a screenshot", "take_screenshot", None),
            ("go to sleep", "sleep", None),
        ]
    )
    output: list[VoiceFixture] = []
    for index in range(count):
        transcript, intent, contact = templates[index % len(templates)]
        condition = ("quiet", "fan-noise", "household-noise")[index % 3]
        output.append(
            VoiceFixture(
                fixture_id=f"voice-{index + 1:04d}",
                expected_transcript=transcript,
                expected_intent=intent,
                expected_contact=contact,
                condition=condition,
            )
        )
    return tuple(output)


def write_fixture_manifest(fixtures: Iterable[VoiceFixture], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in fixtures]
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return target


def read_fixture_manifest(path: str | Path) -> tuple[VoiceFixture, ...]:
    output: list[VoiceFixture] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            output.append(VoiceFixture(**json.loads(line)))
    return tuple(output)


def evaluate_audio_fixtures(
    fixtures: Iterable[VoiceFixture],
    transcribe: Callable[[Path], VoicePrediction],
) -> VoiceEvaluationReport:
    """Evaluate only real fixture audio; missing paths remain explicit failures."""

    fixture_list = tuple(fixtures)
    predictions: list[VoicePrediction] = []
    for fixture in fixture_list:
        if not fixture.audio_path:
            continue
        audio_path = Path(fixture.audio_path)
        if not audio_path.is_file():
            continue
        prediction = transcribe(audio_path)
        if prediction.fixture_id != fixture.fixture_id:
            prediction = VoicePrediction(
                fixture_id=fixture.fixture_id,
                transcript=prediction.transcript,
                intent=prediction.intent,
                contact=prediction.contact,
                latency_ms=prediction.latency_ms,
            )
        predictions.append(prediction)
    return evaluate_predictions(fixture_list, predictions)
