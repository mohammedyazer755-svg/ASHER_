"""Deterministic gain, noise, and reverberation augmentation for PCM WAVs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .audio import PcmAudio, read_wav, write_wav
from .recording import RecordingSession
from .schema import SampleCondition, SampleOrigin, SampleRecord


@dataclass(frozen=True)
class AugmentationConfig:
    """One reproducible augmentation recipe.

    Replay recordings are intentionally not synthesized here; replay evaluation
    must use a real replay capture tagged with ``SampleCondition.REPLAY``.
    """

    gain_db: float = 0.0
    noise_snr_db: float | None = None
    reverb_decay: float = 0.0
    reverb_delay_ms: float = 35.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not -30.0 <= self.gain_db <= 30.0:
            raise ValueError("gain_db must be between -30 and 30")
        if self.noise_snr_db is not None and not -10.0 <= self.noise_snr_db <= 80.0:
            raise ValueError("noise_snr_db must be between -10 and 80")
        if not 0.0 <= self.reverb_decay < 1.0:
            raise ValueError("reverb_decay must be in [0, 1)")
        if not 1.0 <= self.reverb_delay_ms <= 500.0:
            raise ValueError("reverb_delay_ms must be between 1 and 500")


def _clip(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def augment_audio(audio: PcmAudio, config: AugmentationConfig) -> PcmAudio:
    """Apply a deterministic augmentation recipe without optional dependencies."""

    gain = 10.0 ** (config.gain_db / 20.0)
    values = [float(sample) * gain for sample in audio.samples]

    if config.reverb_decay:
        delay_frames = max(1, int(round(audio.sample_rate * config.reverb_delay_ms / 1000.0)))
        delay = delay_frames * audio.channels
        reverberated = list(values)
        for index in range(delay, len(reverberated)):
            reverberated[index] += config.reverb_decay * reverberated[index - delay]
        values = reverberated

    if config.noise_snr_db is not None:
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        reference_rms = max(rms, 32767.0 * 0.001)
        noise_rms = reference_rms / (10.0 ** (config.noise_snr_db / 20.0))
        generator = random.Random(config.seed)
        values = [value + generator.gauss(0.0, noise_rms) for value in values]

    return PcmAudio(
        tuple(_clip(value) for value in values),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
    )


def augment_wav(
    source: str | Path,
    destination: str | Path,
    config: AugmentationConfig,
) -> Path:
    return write_wav(destination, augment_audio(read_wav(source), config))


def augment_session(
    session: RecordingSession,
    recipes: Iterable[AugmentationConfig],
    *,
    include_imported: bool = True,
) -> tuple[SampleRecord, ...]:
    """Create manifest-tracked noisy derivatives of original session clips."""

    eligible_origins = {SampleOrigin.RECORDED.value}
    if include_imported:
        eligible_origins.add(SampleOrigin.IMPORTED.value)
    sources = tuple(sample for sample in session.manifest.samples if sample.origin in eligible_origins)
    created: list[SampleRecord] = []
    for source in sources:
        source_path = session.manifest.resolve_sample_path(session.directory, source)
        audio = read_wav(source_path)
        for recipe in recipes:
            augmented = augment_audio(audio, recipe)
            created.append(
                session.add_audio(
                    augmented,
                    contains_wake_phrase=source.contains_wake_phrase,
                    condition=SampleCondition.NOISY,
                    origin=SampleOrigin.AUGMENTED,
                    expected_authorized=source.expected_authorized,
                    source_sample_id=source.sample_id,
                )
            )
    return tuple(created)
