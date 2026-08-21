"""Lightweight local features and explicitly labelled pretrained adapters."""

from __future__ import annotations

import importlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .audio import PcmAudio, read_wav
from .exceptions import MLDependencyError, ModelError


@dataclass(frozen=True)
class FeatureExtractorMetadata:
    extractor_id: str
    display_name: str
    implementation: str
    provenance: str
    is_pretrained: bool
    is_student_trained: bool
    model_version: str
    source: str
    license: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class FeatureExtractor(Protocol):
    @property
    def metadata(self) -> FeatureExtractorMetadata: ...

    def extract_wav(self, path: str | Path) -> Sequence[float]: ...


def _mono(audio: PcmAudio) -> list[float]:
    scale = 32768.0
    if audio.channels == 1:
        return [sample / scale for sample in audio.samples]
    return [
        sum(audio.samples[index : index + audio.channels]) / (audio.channels * scale)
        for index in range(0, len(audio.samples), audio.channels)
    ]


class StatisticalFeatureExtractor:
    """Dependency-free baseline features; not a pretrained speaker model."""

    metadata = FeatureExtractorMetadata(
        extractor_id="asher.statistical_pcm.v1",
        display_name="ASHER lightweight PCM statistics",
        implementation="local deterministic signal statistics",
        provenance="built_in_nonpretrained",
        is_pretrained=False,
        is_student_trained=False,
        model_version="1",
        source="ASHER VoiceGuard source",
        license=None,
        details={"feature_dimension": 20},
    )

    def extract_audio(self, audio: PcmAudio) -> tuple[float, ...]:
        values = _mono(audio)
        count = len(values)
        mean = sum(values) / count
        centered = [value - mean for value in values]
        rms = math.sqrt(sum(value * value for value in values) / count)
        standard_deviation = math.sqrt(sum(value * value for value in centered) / count)
        mean_absolute = sum(abs(value) for value in values) / count
        peak = max(abs(value) for value in values)
        zero_crossings = sum(
            1 for left, right in zip(values, values[1:]) if (left < 0 <= right) or (left >= 0 > right)
        ) / max(1, count - 1)
        differences = [right - left for left, right in zip(values, values[1:])]
        difference_rms = math.sqrt(sum(value * value for value in differences) / max(1, len(differences)))
        sorted_absolute = sorted(abs(value) for value in values)

        def quantile(fraction: float) -> float:
            return sorted_absolute[min(count - 1, int(round((count - 1) * fraction)))]

        segment_rms: list[float] = []
        for segment in range(8):
            start = segment * count // 8
            end = max(start + 1, (segment + 1) * count // 8)
            block = values[start:end]
            segment_rms.append(math.sqrt(sum(value * value for value in block) / len(block)))

        return (
            math.log1p(audio.duration_seconds),
            mean,
            standard_deviation,
            rms,
            mean_absolute,
            peak,
            zero_crossings,
            difference_rms,
            peak / max(rms, 1e-12),
            quantile(0.25),
            quantile(0.50),
            quantile(0.75),
            *segment_rms,
        )

    def extract_wav(self, path: str | Path) -> tuple[float, ...]:
        return self.extract_audio(read_wav(path))


class PretrainedEmbeddingAdapter:
    """Wrap an external embedding callable and label its provenance explicitly.

    The wrapped feature extractor is *never* described as student-trained. The
    personalized classifier trained on enrollment sessions is the student work.
    """

    def __init__(
        self,
        extractor: Callable[[Path], Sequence[float]],
        *,
        extractor_id: str,
        display_name: str,
        model_version: str,
        source: str,
        license: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not extractor_id or not source:
            raise ValueError("pretrained adapters require a stable extractor_id and source")
        self._extractor = extractor
        self._metadata = FeatureExtractorMetadata(
            extractor_id=extractor_id,
            display_name=display_name,
            implementation="external pretrained embedding adapter",
            provenance="pretrained_external",
            is_pretrained=True,
            is_student_trained=False,
            model_version=model_version,
            source=source,
            license=license,
            details=dict(details or {}),
        )

    @property
    def metadata(self) -> FeatureExtractorMetadata:
        return self._metadata

    def extract_wav(self, path: str | Path) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in self._extractor(Path(path)))
        except Exception as exc:
            raise ModelError("pretrained embedding extraction failed") from exc
        if not values or any(not math.isfinite(value) for value in values):
            raise ModelError("pretrained extractor returned an invalid embedding")
        return values


class SpeechBrainECAPAAdapter:
    """Optional, lazy SpeechBrain ECAPA-TDNN embedding adapter.

    No SpeechBrain, Torch, model download, or device allocation occurs at import
    or construction time. Calling ``extract_wav`` may require downloading the
    externally pretrained model unless it is already cached.
    """

    def __init__(
        self,
        *,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.source = source
        self.savedir = None if savedir is None else Path(savedir)
        self.device = device
        self._classifier: Any = None
        self._metadata = FeatureExtractorMetadata(
            extractor_id=f"speechbrain.ecapa:{source}",
            display_name="SpeechBrain ECAPA-TDNN speaker embedding",
            implementation="optional SpeechBrain adapter",
            provenance="pretrained_external",
            is_pretrained=True,
            is_student_trained=False,
            model_version="upstream-pinned-by-cache",
            source=source,
            license=None,
            details={
                "network_or_cache_may_be_required": True,
                "student_trained_component": "none; classifier head only",
            },
        )

    @property
    def metadata(self) -> FeatureExtractorMetadata:
        return self._metadata

    def _load(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            speaker_module = importlib.import_module("speechbrain.inference.speaker")
        except (ImportError, OSError) as exc:
            raise MLDependencyError(
                "SpeechBrain ECAPA embeddings require optional 'speechbrain' and 'torch' dependencies"
            ) from exc
        kwargs: dict[str, Any] = {"source": self.source, "run_opts": {"device": self.device}}
        if self.savedir is not None:
            kwargs["savedir"] = str(self.savedir)
        try:
            self._classifier = speaker_module.EncoderClassifier.from_hparams(**kwargs)
        except Exception as exc:
            raise MLDependencyError(
                "the pretrained SpeechBrain model could not be loaded from its cache/source"
            ) from exc
        return self._classifier

    def extract_wav(self, path: str | Path) -> tuple[float, ...]:
        audio = read_wav(path)
        if audio.sample_rate != 16_000:
            raise ModelError("SpeechBrain ECAPA adapter currently requires 16 kHz PCM WAV input")
        mono = _mono(audio)
        try:
            torch = importlib.import_module("torch")
            waveform = torch.tensor(mono, dtype=torch.float32).unsqueeze(0)
            embedding = self._load().encode_batch(waveform)
            values = embedding.detach().cpu().reshape(-1).tolist()
        except (MLDependencyError, ModelError):
            raise
        except Exception as exc:
            raise ModelError("SpeechBrain ECAPA embedding extraction failed") from exc
        result = tuple(float(value) for value in values)
        if not result or any(not math.isfinite(value) for value in result):
            raise ModelError("SpeechBrain returned an invalid embedding")
        return result
