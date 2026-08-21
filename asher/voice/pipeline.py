"""Confidence- and ambiguity-gated speech command pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol

from .types import TranscriptResult
from .vocabulary import CommandResolution, DynamicVocabulary


class Transcriber(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> TranscriptResult: ...


class PipelineStatus(str, Enum):
    ACCEPTED = "accepted"
    NO_SPEECH = "no_speech"
    LOW_CONFIDENCE = "low_confidence"
    CLARIFICATION_REQUIRED = "clarification_required"


@dataclass(frozen=True)
class VoicePipelineResult:
    status: PipelineStatus
    transcript: TranscriptResult
    executable_command: str | None = None
    clarification: str | None = None
    command_resolution: CommandResolution | None = None
    used_fallback: bool = False

    @property
    def may_execute(self) -> bool:
        return self.status == PipelineStatus.ACCEPTED and bool(self.executable_command)


class VoiceAccuracyPipeline:
    """Turn audio into a command only after confidence and name checks pass."""

    def __init__(
        self,
        transcriber: Transcriber,
        vocabulary: DynamicVocabulary,
        *,
        confidence_threshold: float = 0.55,
        no_speech_threshold: float = 0.85,
        fallback_transcriber: Transcriber | None = None,
        remote_vocabulary: Sequence[str] = (),
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= no_speech_threshold <= 1.0:
            raise ValueError("no_speech_threshold must be between 0 and 1")
        self.transcriber = transcriber
        self.vocabulary = vocabulary
        self.confidence_threshold = float(confidence_threshold)
        self.no_speech_threshold = float(no_speech_threshold)
        self.fallback_transcriber = fallback_transcriber
        self.remote_vocabulary = tuple(
            dict.fromkeys(str(item).strip() for item in remote_vocabulary if str(item).strip())
        )

    def _transcribe(
        self,
        audio: Any,
        *,
        allow_remote_fallback: bool,
        cancellation: Any | None,
    ) -> tuple[TranscriptResult, bool]:
        vocabulary = self.vocabulary.prompt_terms()
        result = self.transcriber.transcribe(
            audio,
            vocabulary=vocabulary,
            cancellation=cancellation,
        )
        if (
            allow_remote_fallback
            and self.fallback_transcriber is not None
            and result.acoustic_confidence < self.confidence_threshold
        ):
            fallback = self.fallback_transcriber.transcribe(
                audio,
                # Do not transmit the complete local directory by default.
                # Integrators may provide only the names relevant to this turn.
                vocabulary=self.remote_vocabulary,
                cancellation=cancellation,
            )
            if fallback.acoustic_confidence > result.acoustic_confidence:
                metadata = dict(fallback.metadata)
                metadata.update(
                    {
                        "fallback_used": True,
                        "local_raw_text": result.raw_text,
                        "local_acoustic_confidence": result.acoustic_confidence,
                    }
                )
                return replace(fallback, metadata=metadata), True
        return result, False

    def process(
        self,
        audio: Any,
        *,
        contact_expected: bool | None = None,
        allow_remote_fallback: bool = False,
        security_sensitive: bool = False,
        cancellation: Any | None = None,
    ) -> VoicePipelineResult:
        transcript, used_fallback = self._transcribe(
            audio,
            allow_remote_fallback=(allow_remote_fallback and not security_sensitive),
            cancellation=cancellation,
        )
        if (
            not transcript.normalized_text.strip()
            or transcript.no_speech_probability >= self.no_speech_threshold
        ):
            return VoicePipelineResult(
                status=PipelineStatus.NO_SPEECH,
                transcript=transcript,
                clarification="I did not hear a complete command. Please try again.",
                used_fallback=used_fallback,
            )
        if transcript.acoustic_confidence < self.confidence_threshold:
            return VoicePipelineResult(
                status=PipelineStatus.LOW_CONFIDENCE,
                transcript=transcript,
                clarification="I am not confident I heard that correctly. Please repeat it.",
                used_fallback=used_fallback,
            )
        explicit_contact_context = contact_expected is not None
        resolution = self.vocabulary.repair_contact_command(
            transcript.normalized_text,
            assume_plain_search_is_contact=(
                True if contact_expected is None else contact_expected
            ),
        )
        # With no explicit application context, an unmatched plain ``search``
        # can remain a browser query.  Known names still resolve and ambiguous
        # names still require clarification.  Callers that know the active app
        # can pass True to require a clarification for unknown contacts.
        if (
            not explicit_contact_context
            and resolution.contact is not None
            and resolution.contact.status.value == "unknown"
            and transcript.normalized_text.casefold().startswith("search ")
        ):
            resolution = CommandResolution(
                original_text=resolution.original_text,
                resolved_text=resolution.original_text,
                contact=resolution.contact,
            )
        if not resolution.executable:
            return VoicePipelineResult(
                status=PipelineStatus.CLARIFICATION_REQUIRED,
                transcript=transcript,
                clarification=resolution.clarification,
                command_resolution=resolution,
                used_fallback=used_fallback,
            )
        return VoicePipelineResult(
            status=PipelineStatus.ACCEPTED,
            transcript=transcript,
            executable_command=resolution.resolved_text,
            command_resolution=resolution,
            used_fallback=used_fallback,
        )
