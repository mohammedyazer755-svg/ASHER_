"""Lazy Faster-Whisper transcription with VAD and CPU fallback.

The module intentionally has no top-level Faster-Whisper, CTranslate2, Torch,
microphone, or CUDA imports.  This makes configuration screens and unit tests
safe on machines where the optional speech stack is not installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import gc
import os
import sys
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .types import TranscriptResult, TranscriptSegment


class DependencyUnavailableError(RuntimeError):
    """Raised only when transcription is requested without its provider."""


class TranscriptionError(RuntimeError):
    pass


class CancellationLike(Protocol):
    def raise_if_cancelled(self) -> None: ...


DEFAULT_VAD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.50,
    "min_speech_duration_ms": 180,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 180,
}


@dataclass(frozen=True)
class TranscriptionConfig:
    model_size: str = "distil-large-v3"
    device: str = "auto"
    cuda_compute_type: str = "int8_float16"
    cpu_compute_type: str = "int8"
    allow_cpu_fallback: bool = True
    language: str | None = "en"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    condition_on_previous_text: bool = False
    vad_filter: bool = True
    vad_parameters: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_VAD_PARAMETERS)
    )
    word_timestamps: bool = False
    low_confidence_threshold: float = 0.55
    no_speech_threshold: float = 0.85

    def __post_init__(self) -> None:
        device = self.device.strip().lower()
        if not self.model_size.strip():
            raise ValueError("model_size must not be empty")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.beam_size < 1 or self.best_of < 1:
            raise ValueError("beam_size and best_of must be positive")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.no_speech_threshold <= 1.0:
            raise ValueError("no_speech_threshold must be between 0 and 1")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "vad_parameters", dict(self.vad_parameters))


ModelFactory = Callable[..., Any]
CudaProbe = Callable[[], bool]


_WINDOWS_CUDA_DLL_HANDLES: list[Any] = []
_WINDOWS_CUDA_DLL_PATHS: set[str] = set()


def _bundled_cuda_dll_directories(site_packages: str | Path) -> tuple[Path, ...]:
    """Return venv-local CUDA runtime directories that contain required DLLs.

    The NVIDIA Windows wheels install cuBLAS/cuDNN beneath ``site-packages/nvidia``.
    CTranslate2 loads these libraries lazily when the first CUDA inference actually
    runs, so successful model construction alone does not prove the runtime DLLs
    are discoverable.
    """

    root = Path(site_packages)
    candidates = (
        (root / "nvidia" / "cublas" / "bin", "cublas64_12.dll"),
        (root / "nvidia" / "cudnn" / "bin", "cudnn64_9.dll"),
    )
    return tuple(directory for directory, dll in candidates if (directory / dll).is_file())


def _prepare_windows_cuda_runtime() -> tuple[str, ...]:
    """Expose venv-local CUDA DLLs to this ASHER process only.

    No global Windows PATH or CUDA installation is modified.  Python's DLL search
    path and the child-process PATH are extended lazily immediately before local
    CUDA speech probing/model use.
    """

    if os.name != "nt":
        return ()

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    directories = _bundled_cuda_dll_directories(site_packages)
    if not directories:
        return ()

    resolved: list[str] = []
    for directory in directories:
        value = str(directory.resolve())
        resolved.append(value)
        if value in _WINDOWS_CUDA_DLL_PATHS:
            continue

        add_directory = getattr(os, "add_dll_directory", None)
        if callable(add_directory):
            try:
                # Keep the returned handles alive for the process lifetime. Closing
                # them removes the directory from Python's DLL search path.
                _WINDOWS_CUDA_DLL_HANDLES.append(add_directory(value))
            except OSError:
                # PATH below is still useful for CTranslate2's lazy runtime loader.
                pass
        _WINDOWS_CUDA_DLL_PATHS.add(value)

    existing = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    existing_folded = {item.casefold() for item in existing}
    prepend = [item for item in resolved if item.casefold() not in existing_folded]
    if prepend:
        os.environ["PATH"] = os.pathsep.join((*prepend, *existing))
    return tuple(resolved)


def _default_cuda_probe() -> bool:
    """Check CUDA lazily, preferring CTranslate2 over importing Torch."""

    # If the actual provider is absent, do not pay the several-second Torch
    # import merely to produce the eventual dependency error.  Test doubles
    # can still inject ``cuda_probe`` explicitly.
    if importlib.util.find_spec("faster_whisper") is None:
        return False
    _prepare_windows_cuda_runtime()
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        count_fn = getattr(ctranslate2, "get_cuda_device_count", None)
        if callable(count_fn):
            return int(count_fn()) > 0
    except Exception:
        pass
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _default_model_factory(model_size: str, **kwargs: Any) -> Any:
    if str(kwargs.get("device", "")).casefold() == "cuda":
        _prepare_windows_cuda_runtime()
    try:
        module = importlib.import_module("faster_whisper")
    except (ImportError, ModuleNotFoundError) as error:
        raise DependencyUnavailableError(
            "Faster-Whisper is not installed. Install the optional voice dependencies "
            "before starting local transcription."
        ) from error
    return module.WhisperModel(model_size, **kwargs)


def _read(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _confidence(avg_log_probability: float, no_speech_probability: float) -> float:
    # Decoder log probabilities are not calibrated probabilities.  The bounded
    # transform is useful only for routing and clarification decisions.
    speech = 1.0 - max(0.0, min(1.0, no_speech_probability))
    likelihood = math.exp(min(0.0, max(-12.0, avg_log_probability)))
    return max(0.0, min(1.0, likelihood * speech))


def normalize_transcript_surface(text: str) -> str:
    """Normalize spacing/punctuation while preserving payload case and wording."""

    value = str(text).replace("\u2019", "'").replace("`", "'")
    value = " ".join(value.strip().split())
    return value.rstrip(" \t\r\n.!?;:")


class FasterWhisperTranscriber:
    """Thread-safe lazy wrapper around ``faster_whisper.WhisperModel``."""

    def __init__(
        self,
        config: TranscriptionConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
        cuda_probe: CudaProbe | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or TranscriptionConfig()
        self._model_factory = model_factory or _default_model_factory
        self._cuda_probe = cuda_probe or _default_cuda_probe
        self._clock = clock
        self._model: Any | None = None
        self._active_device: str | None = None
        self._load_lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def active_device(self) -> str | None:
        return self._active_device

    def _preferred_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if self._cuda_probe() else "cpu"
        return self.config.device

    def _create_model(self, device: str) -> Any:
        compute_type = (
            self.config.cuda_compute_type
            if device == "cuda"
            else self.config.cpu_compute_type
        )
        return self._model_factory(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
        )

    def load(self) -> "FasterWhisperTranscriber":
        with self._load_lock:
            if self._model is not None:
                return self
            device = self._preferred_device()
            try:
                self._model = self._create_model(device)
                self._active_device = device
            except DependencyUnavailableError:
                raise
            except Exception as error:
                if device != "cuda" or not self.config.allow_cpu_fallback:
                    raise TranscriptionError(
                        f"Could not load speech model on {device}: {error}"
                    ) from error
                try:
                    self._model = self._create_model("cpu")
                    self._active_device = "cpu"
                except DependencyUnavailableError:
                    raise
                except Exception as cpu_error:
                    raise TranscriptionError(
                        "Could not load speech model on CUDA or CPU"
                    ) from cpu_error
        return self

    def unload(self) -> None:
        with self._load_lock:
            was_cuda = self._active_device == "cuda"
            self._model = None
            self._active_device = None
            if was_cuda:
                self._release_cuda_memory()

    @staticmethod
    def _release_cuda_memory() -> None:
        gc.collect()
        # Do not import Torch merely to clear a cache; if it is already loaded,
        # releasing its CUDA allocator is cheap and useful during fallback.
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _decode(
        self,
        audio: str | Path | Any,
        *,
        prompt: str,
    ) -> tuple[list[Any], Any]:
        assert self._model is not None
        kwargs: dict[str, Any] = {
            "language": self.config.language,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "vad_filter": self.config.vad_filter,
            "vad_parameters": dict(self.config.vad_parameters),
            "word_timestamps": self.config.word_timestamps,
        }
        if prompt:
            kwargs["initial_prompt"] = prompt
        decoded = self._model.transcribe(audio, **kwargs)
        if isinstance(decoded, tuple) and len(decoded) == 2:
            segments, info = decoded
        else:
            segments, info = decoded, None
        return list(segments or ()), info

    def transcribe(
        self,
        audio: str | Path | Any,
        *,
        vocabulary: Sequence[str] = (),
        initial_prompt: str = "",
        cancellation: CancellationLike | None = None,
    ) -> TranscriptResult:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.load()
        prompt_parts = [initial_prompt.strip()]
        clean_vocabulary = tuple(
            dict.fromkeys(str(item).strip() for item in vocabulary if str(item).strip())
        )
        if clean_vocabulary:
            prompt_parts.append("Relevant names and applications: " + ", ".join(clean_vocabulary))
        prompt = " ".join(part for part in prompt_parts if part)

        started = self._clock()
        try:
            raw_segments, info = self._decode(audio, prompt=prompt)
        except Exception as error:
            if self._active_device != "cuda" or not self.config.allow_cpu_fallback:
                raise TranscriptionError(f"Speech transcription failed: {error}") from error
            with self._load_lock:
                self._model = None
                self._release_cuda_memory()
                try:
                    self._model = self._create_model("cpu")
                    self._active_device = "cpu"
                except Exception as cpu_error:
                    raise TranscriptionError(
                        "Speech transcription failed on CUDA and CPU"
                    ) from cpu_error
            raw_segments, info = self._decode(audio, prompt=prompt)
        latency_ms = (self._clock() - started) * 1000.0
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        parsed_segments: list[TranscriptSegment] = []
        weighted_confidence = 0.0
        weighted_no_speech = 0.0
        total_duration = 0.0
        for item in raw_segments:
            start = max(0.0, _safe_float(_read(item, "start", 0.0), 0.0))
            end = max(start, _safe_float(_read(item, "end", start), start))
            duration = max(0.05, end - start)
            avg_logprob = _safe_float(_read(item, "avg_logprob", -1.0), -1.0)
            no_speech = max(
                0.0,
                min(1.0, _safe_float(_read(item, "no_speech_prob", 0.0), 0.0)),
            )
            confidence = _confidence(avg_logprob, no_speech)
            segment_text = _read(item, "text", "")
            parsed_segments.append(
                TranscriptSegment(
                    start_seconds=start,
                    end_seconds=end,
                    text="" if segment_text is None else str(segment_text),
                    average_log_probability=avg_logprob,
                    no_speech_probability=no_speech,
                    confidence=confidence,
                )
            )
            total_duration += duration
            weighted_confidence += confidence * duration
            weighted_no_speech += no_speech * duration

        raw_text = "".join(segment.text for segment in parsed_segments).strip()
        if total_duration:
            acoustic = weighted_confidence / total_duration
            no_speech_probability = weighted_no_speech / total_duration
        else:
            acoustic = 0.0
            no_speech_probability = 1.0

        language = _read(info, "language", self.config.language)
        language_probability = _read(info, "language_probability", None)
        duration_value = _read(info, "duration", None)
        return TranscriptResult(
            raw_text=raw_text,
            normalized_text=normalize_transcript_surface(raw_text),
            segments=tuple(parsed_segments),
            language=str(language) if language else None,
            language_probability=(
                _safe_float(language_probability, 0.0)
                if language_probability is not None
                else None
            ),
            acoustic_confidence=acoustic,
            no_speech_probability=no_speech_probability,
            audio_duration_seconds=(
                max(0.0, _safe_float(duration_value, 0.0))
                if duration_value is not None
                else None
            ),
            latency_ms=latency_ms,
            provider="faster-whisper",
            model=self.config.model_size,
            device=self._active_device or "cpu",
            metadata={
                "vad_filter": self.config.vad_filter,
                "vad_parameters": dict(self.config.vad_parameters),
                "beam_size": self.config.beam_size,
                "condition_on_previous_text": self.config.condition_on_previous_text,
            },
        )

    def transcribe_audio(self, audio: str | Path | Any, **kwargs: Any) -> TranscriptResult:
        """Explicitly named compatibility entry point."""

        return self.transcribe(audio, **kwargs)


# Descriptive aliases make migration from older listener code straightforward
# without creating a second implementation.
LazyFasterWhisperTranscriber = FasterWhisperTranscriber
LazyTranscriber = FasterWhisperTranscriber
