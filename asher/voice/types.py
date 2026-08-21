"""Typed speech results shared by local and optional cloud transcribers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class TranscriptSegment:
    """One time-aligned decoder segment.

    ``confidence`` is an engineering estimate derived from decoder log
    probability and no-speech probability.  It is deliberately not described
    as a calibrated identity or security probability.
    """

    start_seconds: float
    end_seconds: float
    text: str
    average_log_probability: float = -1.0
    no_speech_probability: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_seconds", max(0.0, float(self.start_seconds)))
        object.__setattr__(
            self,
            "end_seconds",
            max(float(self.start_seconds), float(self.end_seconds)),
        )
        object.__setattr__(
            self,
            "no_speech_probability",
            _probability(self.no_speech_probability),
        )
        object.__setattr__(self, "confidence", _probability(self.confidence))

    @property
    def avg_logprob(self) -> float:
        return self.average_log_probability

    @property
    def no_speech_prob(self) -> float:
        return self.no_speech_probability


@dataclass(frozen=True)
class TranscriptResult:
    """Complete transcription with the original decoder text preserved."""

    raw_text: str
    normalized_text: str
    segments: tuple[TranscriptSegment, ...] = ()
    language: str | None = None
    language_probability: float | None = None
    acoustic_confidence: float = 0.0
    no_speech_probability: float = 1.0
    audio_duration_seconds: float | None = None
    latency_ms: float = 0.0
    provider: str = "unknown"
    model: str = "unknown"
    device: str = "cpu"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_text", str(self.raw_text))
        object.__setattr__(self, "normalized_text", str(self.normalized_text))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(
            self,
            "acoustic_confidence",
            _probability(self.acoustic_confidence),
        )
        object.__setattr__(
            self,
            "no_speech_probability",
            _probability(self.no_speech_probability),
        )
        if self.language_probability is not None:
            object.__setattr__(
                self,
                "language_probability",
                _probability(self.language_probability),
            )
        object.__setattr__(self, "latency_ms", max(0.0, float(self.latency_ms)))

    @property
    def has_speech(self) -> bool:
        return bool(self.normalized_text.strip()) and self.no_speech_probability < 0.85

    def is_confident(self, threshold: float = 0.55) -> bool:
        return self.has_speech and self.acoustic_confidence >= float(threshold)

    @property
    def raw_transcript(self) -> str:
        """Alias used by callers that distinguish raw from normalized text."""

        return self.raw_text

    @property
    def normalized(self) -> str:
        return self.normalized_text

    @property
    def confidence(self) -> float:
        return self.acoustic_confidence
