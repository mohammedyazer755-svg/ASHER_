"""VoiceGuard-specific exceptions with actionable, privacy-safe messages."""

from __future__ import annotations


class VoiceGuardError(Exception):
    """Base class for recoverable VoiceGuard failures."""


class ManifestError(VoiceGuardError):
    """A recording-session manifest is invalid or inconsistent."""


class AudioFormatError(VoiceGuardError):
    """An audio file is not supported by the local PCM WAV pipeline."""


class RecordingUnavailableError(VoiceGuardError):
    """Live microphone recording cannot be used in this environment."""


class DatasetError(VoiceGuardError):
    """The dataset is incomplete, unsafe, or unsuitable for training."""


class MLDependencyError(VoiceGuardError):
    """An optional machine-learning dependency is unavailable."""


class ModelError(VoiceGuardError):
    """A saved model is invalid or incompatible with its extractor."""


class EnrollmentError(VoiceGuardError):
    """Enrollment or revocation could not be completed safely."""
