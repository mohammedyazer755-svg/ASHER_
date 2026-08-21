"""Wake phrase boundary matching and optional low-resource detector adapters."""

from __future__ import annotations

import importlib
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


DEFAULT_WAKE_PHRASES = (
    "hey asher",
    "hello asher",
    "okay asher",
    "ok asher",
    "asher",
)


@dataclass(frozen=True)
class WakeMatch:
    detected: bool
    phrase: str | None = None
    command: str = ""
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class WakeDetection:
    detected: bool
    score: float
    phrase: str | None = None
    provider: str = "unknown"
    latency_ms: float = 0.0
    command: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))
        object.__setattr__(self, "latency_ms", max(0.0, float(self.latency_ms)))


@runtime_checkable
class WakeDetector(Protocol):
    """Small interface suitable for a CPU standby worker."""

    def detect(self, sample: Any) -> WakeDetection: ...


def _phrase_pattern(phrase: str) -> str:
    tokens = re.findall(r"\w+", str(phrase).casefold(), flags=re.UNICODE)
    return r"[\s,._-]+".join(re.escape(token) for token in tokens)


def match_wake_phrase(
    text: str,
    phrases: Sequence[str] = DEFAULT_WAKE_PHRASES,
) -> WakeMatch:
    """Match a wake phrase as complete words, never as part of another word."""

    value = str(text)
    best: tuple[int, int, str] | None = None
    for phrase in sorted(
        (str(item).strip() for item in phrases if str(item).strip()),
        key=len,
        reverse=True,
    ):
        pattern = _phrase_pattern(phrase)
        if not pattern:
            continue
        found = re.search(rf"(?<!\w){pattern}(?!\w)", value, flags=re.IGNORECASE)
        if found is None:
            continue
        candidate = (found.start(), found.end(), phrase)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return WakeMatch(False)
    start, end, phrase = best
    command = value[end:].strip(" \t\r\n,.-:;!?")
    return WakeMatch(True, phrase=phrase, command=command, start=start, end=end)


class TextWakeDetector:
    """Deterministic transcript detector used after speech decoding and in tests."""

    def __init__(self, phrases: Sequence[str] = DEFAULT_WAKE_PHRASES) -> None:
        self.phrases = tuple(phrases)

    def detect(self, sample: Any) -> WakeDetection:
        started = time.perf_counter()
        match = match_wake_phrase(str(sample), self.phrases)
        return WakeDetection(
            detected=match.detected,
            score=1.0 if match.detected else 0.0,
            phrase=match.phrase,
            command=match.command,
            provider="text-boundary",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def match(self, text: str) -> WakeMatch:
        return match_wake_phrase(text, self.phrases)


WakeModelFactory = Callable[[Path], Any]


def _openwakeword_factory(model_path: Path) -> Any:
    try:
        module = importlib.import_module("openwakeword.model")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "openWakeWord is optional and is not installed; text wake matching remains available"
        ) from error
    return module.Model(wakeword_models=[str(model_path)])


def _maximum_score(prediction: Any) -> tuple[str | None, float]:
    if not isinstance(prediction, Mapping):
        return None, 0.0
    best_name: str | None = None
    best_score = 0.0
    for name, value in prediction.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = value
        elif not isinstance(value, (str, bytes, int, float)):
            try:
                values = tuple(value)
            except TypeError:
                values = (value,)
        else:
            values = (value,)
        for item in values:
            try:
                score = float(item)
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_name = str(name)
                best_score = score
    return best_name, max(0.0, min(1.0, best_score))


class LazyOpenWakeWordDetector:
    """Optional personalised wake verifier loaded only on first audio frame."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.65,
        phrase: str = "hey asher",
        model_factory: WakeModelFactory | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.phrase = phrase
        self._factory = model_factory or _openwakeword_factory
        self._clock = clock
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "LazyOpenWakeWordDetector":
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise FileNotFoundError(f"Wake model not found: {self.model_path}")
                self._model = self._factory(self.model_path)
        return self

    def reset(self) -> None:
        model = self._model
        reset = getattr(model, "reset", None)
        if callable(reset):
            reset()

    def detect(self, sample: Any) -> WakeDetection:
        self.load()
        started = self._clock()
        prediction = self._model.predict(sample)
        _, score = _maximum_score(prediction)
        return WakeDetection(
            detected=score >= self.threshold,
            score=score,
            phrase=self.phrase if score >= self.threshold else None,
            command="",
            provider="openwakeword",
            latency_ms=(self._clock() - started) * 1000.0,
        )


class EnergyGate:
    """Very cheap standby gate for avoiding model work during silence.

    This is only an energy gate, not an authentication signal or a wake-word
    recognizer.  Active frames should still be passed to a real wake detector.
    """

    def __init__(self, threshold: float = 0.012) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def detect(self, sample: Any) -> WakeDetection:
        try:
            import array

            values = array.array("h")
            values.frombytes(bytes(sample))
            rms = (
                sum((item / 32768.0) ** 2 for item in values) / len(values)
            ) ** 0.5 if values else 0.0
        except (TypeError, ValueError, BufferError):
            rms = 0.0
        score = max(0.0, min(1.0, rms / max(self.threshold, 1e-9)))
        return WakeDetection(
            detected=rms >= self.threshold,
            score=score,
            provider="energy-gate",
        )
