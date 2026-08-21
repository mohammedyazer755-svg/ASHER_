"""Dependency-free voice-activity turn capture primitives.

An audio backend can feed PCM16 frames into :class:`TurnCapture`; this module
does not open a microphone itself.  That separation keeps standby and tests
lightweight while allowing a Windows/UI worker to own the actual device.
"""

from __future__ import annotations

import array
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16_000
    frame_duration_ms: int = 20
    start_consecutive_frames: int = 2
    end_silence_ms: int = 550
    pre_roll_ms: int = 160
    max_turn_ms: int = 12_000
    absolute_threshold: float = 0.012
    noise_multiplier: float = 2.8

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_duration_ms <= 0:
            raise ValueError("sample_rate and frame_duration_ms must be positive")
        if self.start_consecutive_frames < 1:
            raise ValueError("start_consecutive_frames must be positive")

    @property
    def frame_samples(self) -> int:
        return max(1, round(self.sample_rate * self.frame_duration_ms / 1000))

    @property
    def end_silence_frames(self) -> int:
        return max(1, math.ceil(self.end_silence_ms / self.frame_duration_ms))

    @property
    def pre_roll_frames(self) -> int:
        return max(0, math.ceil(self.pre_roll_ms / self.frame_duration_ms))

    @property
    def max_turn_frames(self) -> int:
        return max(1, math.ceil(self.max_turn_ms / self.frame_duration_ms))


@dataclass(frozen=True)
class AudioFrame:
    pcm16: bytes
    sample_rate: int = 16_000


@dataclass(frozen=True)
class VadDecision:
    rms: float
    threshold: float
    speech: bool


@dataclass(frozen=True)
class CapturedTurn:
    pcm16: bytes
    sample_rate: int
    frame_count: int
    speech_started: bool
    ended_on_silence: bool


def pcm16_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    samples = array.array("h")
    try:
        samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
    except (BufferError, ValueError):
        return 0.0
    if not samples:
        return 0.0
    return math.sqrt(sum((sample / 32768.0) ** 2 for sample in samples) / len(samples))


class VoiceActivityDetector:
    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self._noise_rms = 0.0

    @property
    def noise_rms(self) -> float:
        return self._noise_rms

    def calibrate(self, frames: Iterable[AudioFrame]) -> float:
        values = [pcm16_rms(frame.pcm16) for frame in frames]
        self._noise_rms = sum(values) / len(values) if values else 0.0
        return self._noise_rms

    def threshold(self) -> float:
        return max(
            self.config.absolute_threshold,
            self._noise_rms * self.config.noise_multiplier,
        )

    def decide(self, frame: AudioFrame) -> VadDecision:
        rms = pcm16_rms(frame.pcm16)
        threshold = self.threshold()
        return VadDecision(rms=rms, threshold=threshold, speech=rms >= threshold)


class TurnCapture:
    """Collect one complete speech turn from already framed PCM audio."""

    def __init__(self, vad: VoiceActivityDetector | None = None) -> None:
        self.vad = vad or VoiceActivityDetector()
        self.config = self.vad.config

    def capture(
        self,
        frames: Iterable[AudioFrame],
        *,
        cancellation: Any | None = None,
    ) -> CapturedTurn:
        pre_roll: list[AudioFrame] = []
        selected: list[AudioFrame] = []
        started = False
        consecutive_speech = 0
        silence_frames = 0
        ended_on_silence = False
        for frame in frames:
            if frame.sample_rate != self.config.sample_rate:
                raise ValueError("audio frame sample rate does not match VAD configuration")
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            decision = self.vad.decide(frame)
            if not started:
                pre_roll.append(frame)
                if len(pre_roll) > self.config.pre_roll_frames:
                    pre_roll.pop(0)
                consecutive_speech = consecutive_speech + 1 if decision.speech else 0
                if consecutive_speech < self.config.start_consecutive_frames:
                    continue
                started = True
                selected.extend(pre_roll)
                pre_roll.clear()
                silence_frames = 0
                continue

            selected.append(frame)
            if decision.speech:
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= self.config.end_silence_frames:
                    ended_on_silence = True
                    break
            if len(selected) >= self.config.max_turn_frames:
                break

        if not selected:
            return CapturedTurn(
                pcm16=b"",
                sample_rate=self.config.sample_rate,
                frame_count=0,
                speech_started=False,
                ended_on_silence=False,
            )
        return CapturedTurn(
            pcm16=b"".join(frame.pcm16 for frame in selected),
            sample_rate=selected[0].sample_rate,
            frame_count=len(selected),
            speech_started=started,
            ended_on_silence=ended_on_silence,
        )
