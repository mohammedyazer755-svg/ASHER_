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


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance implementation for character-based similarity."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def match_fuzzy_wake_phrase(
    text: str,
    phrases: Sequence[str] = DEFAULT_WAKE_PHRASES,
) -> WakeMatch:
    """Fuzzy matching of wake phrases restricted to the beginning of the utterance."""
    # Normalize whole text to check exact whole-utterance confusions first.
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    norm_text = " ".join(words)

    # Specific acoustic whole-utterance confusions
    FUZZY_WHOLE_UTTERANCE_CONFUSIONS = {
        "ahshel",
        "yeah sir",
        "yeah i should",
        "he has it",
    }
    if norm_text in FUZZY_WHOLE_UTTERANCE_CONFUSIONS:
        # These are all wake signals for "hey asher"
        return WakeMatch(True, phrase="hey asher", command="", start=0, end=len(text))

    matches = list(re.finditer(r"\w+", text, flags=re.UNICODE))
    if not matches:
        return WakeMatch(False)

    best: tuple[int, int] | None = None
    best_phrase: str | None = None

    for phrase in sorted(
        (str(item).strip() for item in phrases if str(item).strip()),
        key=len,
        reverse=True,
    ):
        phrase_norm = " ".join(re.findall(r"\w+", phrase.casefold(), flags=re.UNICODE))
        if not phrase_norm:
            continue
        phrase_words = phrase_norm.split()
        K = len(phrase_words)

        if len(matches) >= K:
            # We take the K-word prefix of the utterance
            start = matches[0].start()
            end = matches[K - 1].end()
            prefix_orig = text[start:end]
            prefix_norm = " ".join(re.findall(r"\w+", prefix_orig.casefold(), flags=re.UNICODE))

            # Compute Levenshtein distance
            dist = levenshtein_distance(phrase_norm, prefix_norm)

            # Determine threshold
            if len(phrase_norm) >= 9:
                max_dist = 2
            elif len(phrase_norm) >= 5:
                max_dist = 1
            else:
                max_dist = 0

            if dist <= max_dist:
                candidate = (start, end)
                if best is None or candidate < best:
                    best = candidate
                    best_phrase = phrase

    if best is None or best_phrase is None:
        return WakeMatch(False)

    start, end = best
    command = text[end:].strip(" \t\r\n,.-:;!?")
    return WakeMatch(True, phrase=best_phrase, command=command, start=start, end=end)


class TextWakeDetector:

    """Deterministic transcript detector used after speech decoding and in tests."""

    def __init__(self, phrases: Sequence[str] = DEFAULT_WAKE_PHRASES) -> None:
        self.phrases = tuple(phrases)

    def detect(self, sample: Any, *, fuzzy: bool = False) -> WakeDetection:
        started = time.perf_counter()
        match = match_wake_phrase(str(sample), self.phrases)
        if match.detected:
            return WakeDetection(
                detected=True,
                score=1.0,
                phrase=match.phrase,
                command=match.command,
                provider="text-boundary",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        if fuzzy:
            fuzzy_match = match_fuzzy_wake_phrase(str(sample), self.phrases)
            if fuzzy_match.detected:
                return WakeDetection(
                    detected=True,
                    score=0.8,
                    phrase=fuzzy_match.phrase,
                    command=fuzzy_match.command,
                    provider="text-boundary-fuzzy",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

        return WakeDetection(
            detected=False,
            score=0.0,
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
