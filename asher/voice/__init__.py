"""Privacy-preserving, optional-dependency-safe speech components for ASHER.

Importing this package never opens a microphone or loads a speech model.  The
heavy providers are instantiated only when their explicit ``load`` or
``transcribe`` methods are called.
"""

from .pipeline import PipelineStatus, VoiceAccuracyPipeline, VoicePipelineResult
from .evaluation import (
    VoiceEvaluationReport,
    VoiceFixture,
    VoicePrediction,
    evaluate_audio_fixtures,
    evaluate_predictions,
    generate_non_private_fixtures,
    read_fixture_manifest,
    word_error_counts,
    word_error_rate,
    write_fixture_manifest,
)
from .capture import (
    AudioFrame,
    CapturedTurn,
    TurnCapture,
    VadConfig,
    VadDecision,
    VoiceActivityDetector,
    pcm16_rms,
)
from .transcription import (
    DependencyUnavailableError,
    FasterWhisperTranscriber,
    LazyFasterWhisperTranscriber,
    LazyTranscriber,
    TranscriptionConfig,
)
from .types import TranscriptResult, TranscriptSegment
from .vocabulary import (
    CommandResolution,
    DynamicVocabulary,
    NameResolution,
    ResolutionStatus,
    discover_windows_applications,
)
from .wakeword import (
    DEFAULT_WAKE_PHRASES,
    EnergyGate,
    LazyOpenWakeWordDetector,
    TextWakeDetector,
    WakeDetection,
    WakeMatch,
    match_wake_phrase,
)

__all__ = [
    "CommandResolution",
    "AudioFrame",
    "CapturedTurn",
    "DEFAULT_WAKE_PHRASES",
    "DependencyUnavailableError",
    "DynamicVocabulary",
    "discover_windows_applications",
    "EnergyGate",
    "VoiceEvaluationReport",
    "VoiceFixture",
    "VoicePrediction",
    "FasterWhisperTranscriber",
    "LazyOpenWakeWordDetector",
    "LazyFasterWhisperTranscriber",
    "LazyTranscriber",
    "NameResolution",
    "PipelineStatus",
    "ResolutionStatus",
    "TextWakeDetector",
    "TurnCapture",
    "TranscriptResult",
    "TranscriptSegment",
    "TranscriptionConfig",
    "VadConfig",
    "VadDecision",
    "VoiceActivityDetector",
    "VoiceAccuracyPipeline",
    "VoicePipelineResult",
    "WakeDetection",
    "WakeMatch",
    "match_wake_phrase",
    "pcm16_rms",
    "evaluate_audio_fixtures",
    "evaluate_predictions",
    "generate_non_private_fixtures",
    "read_fixture_manifest",
    "word_error_counts",
    "word_error_rate",
    "write_fixture_manifest",
]
