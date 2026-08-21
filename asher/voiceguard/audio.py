"""Small, dependency-free PCM WAV primitives used by VoiceGuard."""

from __future__ import annotations

import hashlib
import os
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .exceptions import AudioFormatError


@dataclass(frozen=True)
class PcmAudio:
    """Interleaved signed 16-bit PCM samples."""

    samples: tuple[int, ...]
    sample_rate: int = 16_000
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0:
            raise AudioFormatError("sample_rate and channels must be positive")
        if not self.samples or len(self.samples) % self.channels:
            raise AudioFormatError("PCM audio must contain complete, non-empty frames")
        if any(sample < -32768 or sample > 32767 for sample in self.samples):
            raise AudioFormatError("PCM sample values must fit signed 16-bit audio")

    @property
    def sample_width(self) -> int:
        return 2

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


def _pcm_bytes(samples: tuple[int, ...]) -> bytes:
    data = array("h", samples)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def read_wav(path: str | Path) -> PcmAudio:
    """Read an uncompressed 16-bit PCM WAV without loading optional audio stacks."""

    source = Path(path)
    try:
        with wave.open(str(source), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise AudioFormatError("compressed WAV files are not supported")
            if stream.getsampwidth() != 2:
                raise AudioFormatError("VoiceGuard requires 16-bit PCM WAV files")
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        if isinstance(exc, AudioFormatError):
            raise
        raise AudioFormatError(f"could not read PCM WAV: {source.name}") from exc

    values = array("h")
    values.frombytes(frames)
    if sys.byteorder != "little":
        values.byteswap()
    return PcmAudio(tuple(values), sample_rate=sample_rate, channels=channels)


def write_wav(path: str | Path, audio: PcmAudio) -> Path:
    """Atomically write a real RIFF/WAVE file containing signed 16-bit PCM."""

    destination = Path(path)
    if destination.suffix.lower() != ".wav":
        raise AudioFormatError("WAV destination must use the .wav extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with wave.open(str(temporary), "wb") as stream:
            stream.setnchannels(audio.channels)
            stream.setsampwidth(audio.sample_width)
            stream.setframerate(audio.sample_rate)
            stream.writeframes(_pcm_bytes(audio.samples))
        os.replace(temporary, destination)
    except (wave.Error, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise AudioFormatError(f"could not write PCM WAV: {destination.name}") from exc
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
