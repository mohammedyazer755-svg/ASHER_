"""Validated ASHER configuration loaded without exposing secret values."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from asher.paths import RuntimePaths


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _ollama_endpoint() -> tuple[str, bool]:
    """Validate the Ollama endpoint and report whether it is loopback-only.

    Ollama is treated as a local provider by the planner.  Silently accepting
    an arbitrary host would therefore disclose private prompts while still
    labelling the request ``offline``.  Remote endpoints require an explicit
    opt-in and are marked online by the provider composition.
    """

    raw = (os.getenv("ASHER_OLLAMA_URL") or "http://127.0.0.1:11434").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ASHER_OLLAMA_URL must be an HTTP(S) URL with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("ASHER_OLLAMA_URL must not contain credentials, query, or fragment data")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("ASHER_OLLAMA_URL has an invalid port") from error
    host = parsed.hostname.casefold().strip("[]")
    local = host in {"localhost", "127.0.0.1", "::1"}
    if not local and not _boolean("ASHER_ALLOW_REMOTE_OLLAMA", False):
        raise ValueError(
            "ASHER_OLLAMA_URL points off-device; set ASHER_ALLOW_REMOTE_OLLAMA=true "
            "only after reviewing the privacy boundary"
        )
    return raw, local


@dataclass(frozen=True)
class AsherConfig:
    owner_name: str
    runtime: RuntimePaths
    dry_run: bool = True
    session_minutes: int = 10
    confirmation_seconds: int = 90
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_local: bool = True
    ollama_model: str = "qwen3:4b"
    openai_routine_model: str = "gpt-5.6-luna"
    openai_reasoning_model: str = "gpt-5.6-terra"
    openai_transcribe_model: str = "gpt-4o-transcribe"
    openai_tts_model: str = "gpt-4o-mini-tts"
    voice_profile: str = "asher_male"
    whisper_model: str = "small.en"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    @property
    def openai_enabled(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", "").strip())

    @classmethod
    def load(cls, runtime_dir: str | Path | None = None) -> "AsherConfig":
        owner_name = (
            os.getenv("ASHER_OWNER_NAME")
            or os.getenv("NAME")
            or "Owner"
        ).strip()
        if not owner_name:
            owner_name = "Owner"

        device = os.getenv("ASHER_WHISPER_DEVICE", "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("ASHER_WHISPER_DEVICE must be auto, cpu, or cuda")

        ollama_url, ollama_local = _ollama_endpoint()

        return cls(
            owner_name=owner_name,
            runtime=RuntimePaths.discover(runtime_dir),
            dry_run=_boolean("ASHER_DRY_RUN", True),
            session_minutes=_integer("ASHER_SESSION_MINUTES", 10, 1, 120),
            confirmation_seconds=_integer("ASHER_CONFIRMATION_SECONDS", 90, 15, 600),
            ollama_url=ollama_url,
            ollama_local=ollama_local,
            ollama_model=os.getenv("ASHER_OLLAMA_MODEL", "qwen3:4b").strip(),
            openai_routine_model=os.getenv("ASHER_OPENAI_ROUTINE_MODEL", "gpt-5.6-luna").strip(),
            openai_reasoning_model=os.getenv("ASHER_OPENAI_REASONING_MODEL", "gpt-5.6-terra").strip(),
            openai_transcribe_model=os.getenv("ASHER_OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe").strip(),
            openai_tts_model=os.getenv("ASHER_OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip(),
            voice_profile=os.getenv("ASHER_VOICE_PROFILE", "asher_male").strip(),
            whisper_model=os.getenv("ASHER_WHISPER_MODEL", "small.en").strip(),
            whisper_device=device,
            whisper_compute_type=os.getenv("ASHER_WHISPER_COMPUTE_TYPE", "auto").strip(),
        )
