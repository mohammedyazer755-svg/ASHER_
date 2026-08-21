"""Lazy compatibility wrappers for the legacy voice listener.

The old module loaded Whisper, Torch, and a microphone at import time. The
active listener is asher.voice.runtime and opens hardware only from
python main.py --voice. These helpers remain import-safe for diagnostics.
"""

from __future__ import annotations

import os
from typing import Any

from voice.text_normalizer import normalise_voice_command


WHISPER_MODEL_NAME = os.getenv("ASHER_WHISPER_MODEL", "small.en")
MICROPHONE_INDEX = int(os.getenv("ASHER_MIC_INDEX", "0")) if os.getenv("ASHER_MIC_INDEX", "").isdigit() else None
DEFAULT_TIMEOUT = 6
DEFAULT_PHRASE_TIME_LIMIT = 12
DEVICE = "auto"
ACTIVE_DEVICE = "auto"


def list_microphones() -> list[str]:
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        devices = sd.query_devices()
    except Exception:
        return []
    names: list[str] = []
    for item in devices:
        if isinstance(item, dict) and int(item.get("max_input_channels", 0)) > 0:
            names.append(str(item.get("name", "Input device")))
    return names


def selected_microphone_name() -> str:
    if MICROPHONE_INDEX is None:
        return "Windows default microphone"
    names = list_microphones()
    if 0 <= MICROPHONE_INDEX < len(names):
        return f"[{MICROPHONE_INDEX}] {names[MICROPHONE_INDEX]}"
    return f"Unknown microphone index {MICROPHONE_INDEX}"


def calibrate_microphone(duration: float = 1.2) -> None:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not list_microphones():
        raise RuntimeError("No input microphone is available; no calibration was performed")


def clean_command(command: str) -> str:
    return normalise_voice_command(str(command))


def listen(*, timeout: float = DEFAULT_TIMEOUT, phrase_time_limit: float = DEFAULT_PHRASE_TIME_LIMIT, **_kwargs: Any) -> str:
    if timeout <= 0 or phrase_time_limit <= 0:
        raise ValueError("voice timeouts must be positive")
    raise RuntimeError(
        "The legacy one-shot listener is disabled. Start the authenticated runtime with: "
        "python main.py --voice"
    )


def listen_command() -> str:
    return listen()


def take_command() -> str:
    return listen()


def is_exit_command(command: str) -> bool:
    return str(command).strip().casefold() in {"exit", "quit", "goodbye", "bye"}


__all__ = [
    "ACTIVE_DEVICE",
    "DEFAULT_PHRASE_TIME_LIMIT",
    "DEFAULT_TIMEOUT",
    "DEVICE",
    "MICROPHONE_INDEX",
    "WHISPER_MODEL_NAME",
    "calibrate_microphone",
    "clean_command",
    "is_exit_command",
    "listen",
    "listen_command",
    "list_microphones",
    "selected_microphone_name",
    "take_command",
]

