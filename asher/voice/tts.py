"""Provider-independent, interruptible text-to-speech services.

Providers are deliberately lazy: importing this module never initializes SAPI,
loads the OpenAI SDK, contacts a network service, or starts a worker thread.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable
from uuid import uuid4

from asher.core.redaction import contains_prohibited_secret, redact_text


OPENAI_TTS_DISCLOSURE = "Online voices are AI-generated, not human voices."
OPENAI_VOICE_IDS = {"male": "cedar", "female": "marin"}


class TTSError(RuntimeError):
    """Base error raised by a speech provider."""


class TTSUnavailableError(TTSError):
    """The selected provider cannot run in the current environment."""


class UnknownVoiceProfile(KeyError):
    """A requested voice profile is not registered."""


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """A stable user-facing voice profile; provider IDs live only here."""

    name: str
    label: str
    provider: str
    voice_id: str | None = None
    gender_hint: str | None = None
    style: str = "warm, natural, practical, and concise"
    speed: float = 1.0
    volume: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Voice profile name is required")
        if not self.label.strip():
            raise ValueError("Voice profile label is required")
        if not self.provider.strip():
            raise ValueError("Voice profile provider is required")
        normalized_speed = float(self.speed)
        normalized_volume = float(self.volume)
        object.__setattr__(self, "speed", normalized_speed)
        object.__setattr__(self, "volume", normalized_volume)
        if not 0.5 <= normalized_speed <= 2.0:
            raise ValueError("Voice speed must be between 0.5 and 2.0")
        if not 0.0 <= normalized_volume <= 1.0:
            raise ValueError("Voice volume must be between 0.0 and 1.0")
        if self.gender_hint not in {None, "male", "female", "neutral"}:
            raise ValueError("gender_hint must be male, female, neutral, or None")

    def adjusted(
        self,
        *,
        speed: float | None = None,
        style: str | None = None,
        volume: float | None = None,
    ) -> "VoiceProfile":
        return replace(
            self,
            speed=self.speed if speed is None else speed,
            style=self.style if style is None else style,
            volume=self.volume if volume is None else volume,
        )


class VoiceProfileRegistry:
    """Thread-safe registry used by settings and speech playback."""

    def __init__(self, profiles: tuple[VoiceProfile, ...] = ()) -> None:
        self._lock = threading.RLock()
        self._profiles: dict[str, VoiceProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: VoiceProfile, *, replace_existing: bool = False) -> None:
        with self._lock:
            if profile.name in self._profiles and not replace_existing:
                raise ValueError(f"Voice profile already exists: {profile.name}")
            self._profiles[profile.name] = profile

    def get(self, name: str) -> VoiceProfile:
        with self._lock:
            try:
                return self._profiles[name]
            except KeyError as error:
                raise UnknownVoiceProfile(name) from error

    def all(self) -> tuple[VoiceProfile, ...]:
        with self._lock:
            return tuple(self._profiles.values())

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._profiles)


DEFAULT_VOICE_PROFILES = (
    # ``asher_*`` names are the persisted configuration names.  The
    # ``offline_*``/``online_*`` aliases keep older settings files readable.
    VoiceProfile(
        name="asher_male",
        label="Male — Windows offline",
        provider="sapi",
        gender_hint="male",
    ),
    VoiceProfile(
        name="asher_female",
        label="Female — Windows offline",
        provider="sapi",
        gender_hint="female",
    ),
    VoiceProfile(
        name="offline_male",
        label="Male — Windows offline (compatibility)",
        provider="sapi",
        gender_hint="male",
    ),
    VoiceProfile(
        name="offline_female",
        label="Female — Windows offline (compatibility)",
        provider="sapi",
        gender_hint="female",
    ),
    VoiceProfile(
        name="online_male",
        label="Male — OpenAI online",
        provider="openai",
        voice_id=OPENAI_VOICE_IDS["male"],
        gender_hint="male",
    ),
    VoiceProfile(
        name="online_female",
        label="Female — OpenAI online",
        provider="openai",
        voice_id=OPENAI_VOICE_IDS["female"],
        gender_hint="female",
    ),
    VoiceProfile(
        name="asher_male_online",
        label="Male — OpenAI online (compatibility)",
        provider="openai",
        voice_id=OPENAI_VOICE_IDS["male"],
        gender_hint="male",
    ),
    VoiceProfile(
        name="asher_female_online",
        label="Female — OpenAI online (compatibility)",
        provider="openai",
        voice_id=OPENAI_VOICE_IDS["female"],
        gender_hint="female",
    ),
)


@runtime_checkable
class SpeechProvider(Protocol):
    """Blocking provider contract. ``TTSManager`` supplies async execution."""

    def speak(
        self,
        text: str,
        profile: VoiceProfile,
        stop_event: threading.Event,
    ) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechResult:
    utterance_id: str
    profile_name: str
    provider: str
    success: bool
    cancelled: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0


class SpeechHandle:
    """Wait for, inspect, or cooperatively cancel one queued utterance."""

    def __init__(self, utterance_id: str, on_cancel: Callable[["SpeechHandle"], None]) -> None:
        self.utterance_id = utterance_id
        self.stop_event = threading.Event()
        self._done = threading.Event()
        self._result: SpeechResult | None = None
        self._lock = threading.Lock()
        self._on_cancel = on_cancel

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self.stop_event.is_set()

    def cancel(self) -> bool:
        if self._done.is_set() or self.stop_event.is_set():
            return False
        self.stop_event.set()
        self._on_cancel(self)
        return True

    def wait(self, timeout: float | None = None) -> SpeechResult | None:
        if not self._done.wait(timeout):
            return None
        with self._lock:
            return self._result

    def result(self, timeout: float | None = None) -> SpeechResult:
        result = self.wait(timeout)
        if result is None:
            raise TimeoutError("Speech did not finish before the timeout")
        return result

    def _finish(self, result: SpeechResult) -> None:
        with self._lock:
            if self._done.is_set():
                return
            self._result = result
            self._done.set()


class TTSManager:
    """Serializes speech safely while allowing callers to remain responsive."""

    def __init__(
        self,
        registry: VoiceProfileRegistry | None = None,
        *,
        selected_profile: str = "asher_male",
        fallback_profile: str | None = None,
    ) -> None:
        self.registry = registry or VoiceProfileRegistry(DEFAULT_VOICE_PROFILES)
        self.registry.get(selected_profile)
        if fallback_profile is not None:
            self.registry.get(fallback_profile)
        self._selected_profile = selected_profile
        self._fallback_profile = fallback_profile
        self._providers: dict[str, SpeechProvider] = {}
        self._lock = threading.RLock()
        self._active: tuple[SpeechHandle, SpeechProvider] | None = None
        self._handles: set[SpeechHandle] = set()
        self._closed = False
        self._executor: ThreadPoolExecutor | None = None

    @property
    def selected_profile(self) -> VoiceProfile:
        with self._lock:
            name = self._selected_profile
        return self.registry.get(name)

    @property
    def selected_profile_name(self) -> str:
        with self._lock:
            return self._selected_profile

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def active_handle(self) -> SpeechHandle | None:
        with self._lock:
            return self._active[0] if self._active is not None else None

    def set_profile(self, name: str) -> VoiceProfile:
        profile = self.registry.get(name)
        with self._lock:
            self._selected_profile = name
        return profile

    def register_provider(self, name: str, provider: SpeechProvider) -> None:
        if not name.strip():
            raise ValueError("Provider name is required")
        if not isinstance(provider, SpeechProvider):
            raise TypeError("Provider must implement speak() and stop()")
        with self._lock:
            self._providers[name] = provider

    def provider_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._providers)

    def unregister_provider(self, name: str) -> SpeechProvider | None:
        with self._lock:
            return self._providers.pop(name, None)

    def speak_async(
        self,
        text: str,
        *,
        profile_name: str | None = None,
        speed: float | None = None,
        style: str | None = None,
        interrupt: bool = False,
    ) -> SpeechHandle:
        clean_text = str(text).strip()
        if not clean_text:
            raise ValueError("Speech text cannot be empty")
        if interrupt:
            self.stop()

        with self._lock:
            if self._closed:
                raise RuntimeError("TTS manager is closed")
            selected = profile_name or self._selected_profile
            profile = self.registry.get(selected).adjusted(speed=speed, style=style)
            handle = SpeechHandle(uuid4().hex, self._cancel_handle)
            self._handles.add(handle)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asher-tts")
            executor = self._executor

        executor.submit(self._run, handle, clean_text, profile)
        return handle

    def speak(
        self,
        text: str,
        *,
        profile_name: str | None = None,
        speed: float | None = None,
        style: str | None = None,
        interrupt: bool = False,
        timeout: float | None = None,
    ) -> SpeechResult:
        return self.speak_async(
            text,
            profile_name=profile_name,
            speed=speed,
            style=style,
            interrupt=interrupt,
        ).result(timeout)

    def stop(self) -> int:
        """Cancel active and queued speech and interrupt the active provider."""

        with self._lock:
            handles = tuple(self._handles)
            active = self._active
        for handle in handles:
            handle.stop_event.set()
        if active is not None:
            try:
                active[1].stop()
            except Exception:
                pass
        return len(handles)

    def close(self, *, wait: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        self.stop()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)

    def _cancel_handle(self, handle: SpeechHandle) -> None:
        with self._lock:
            active = self._active
        if active is not None and active[0] is handle:
            try:
                active[1].stop()
            except Exception:
                pass

    def _run(self, handle: SpeechHandle, text: str, profile: VoiceProfile) -> None:
        started = time.monotonic()
        provider: SpeechProvider | None = None
        effective_profile = profile
        error: str | None = None
        try:
            if handle.stop_event.is_set():
                return
            with self._lock:
                provider = self._providers.get(profile.provider)
                if provider is None:
                    raise TTSUnavailableError(
                        f"Speech provider '{profile.provider}' is not configured"
                    )
                self._active = (handle, provider)
            if handle.stop_event.is_set():
                return
            try:
                provider.speak(text, profile, handle.stop_event)
            except Exception as caught:
                fallback_name = self._fallback_profile
                if (
                    handle.stop_event.is_set()
                    or fallback_name is None
                    or fallback_name == profile.name
                ):
                    raise
                fallback_profile = self.registry.get(fallback_name)
                with self._lock:
                    fallback_provider = self._providers.get(fallback_profile.provider)
                    if fallback_provider is None:
                        raise caught
                    self._active = (handle, fallback_provider)
                if handle.stop_event.is_set():
                    return
                fallback_provider.speak(text, fallback_profile, handle.stop_event)
                effective_profile = fallback_profile
        except Exception as caught:
            error = redact_text(f"{type(caught).__name__}: {caught}")
        finally:
            cancelled = handle.stop_event.is_set()
            result = SpeechResult(
                utterance_id=handle.utterance_id,
                profile_name=effective_profile.name,
                provider=effective_profile.provider,
                success=error is None and not cancelled,
                cancelled=cancelled,
                error=None if cancelled else error,
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )
            with self._lock:
                if self._active is not None and self._active[0] is handle:
                    self._active = None
                self._handles.discard(handle)
            handle._finish(result)


class SapiSpeechProvider:
    """Offline Windows speech through pyttsx3's SAPI5 driver."""

    def __init__(self, engine_factory: Callable[[], Any] | None = None) -> None:
        self._engine_factory = engine_factory
        self._lock = threading.RLock()
        self._engine: Any | None = None

    def _create_engine(self) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory()
        if os.name != "nt":
            raise TTSUnavailableError("SAPI speech is available only on Windows")
        try:
            import pyttsx3
        except ImportError as error:
            raise TTSUnavailableError(
                "Offline speech requires pyttsx3; install the project dependencies"
            ) from error
        return pyttsx3.init("sapi5")

    @staticmethod
    def _select_voice(engine: Any, profile: VoiceProfile) -> str | None:
        voices = tuple(engine.getProperty("voices") or ())
        if not voices:
            return None
        if profile.voice_id:
            for voice in voices:
                candidate = getattr(voice, "id", None) or getattr(voice, "name", None)
                if str(candidate or "") == profile.voice_id:
                    return str(candidate)
        if profile.gender_hint:
            hint = profile.gender_hint.lower()
            for voice in voices:
                metadata = " ".join(
                    str(getattr(voice, field, "")) for field in ("gender", "name", "id")
                ).lower()
                if hint in metadata:
                    return str(getattr(voice, "id", None) or getattr(voice, "name", ""))
        fallback_index = 1 if profile.gender_hint == "female" and len(voices) > 1 else 0
        return str(
            getattr(voices[fallback_index], "id", None)
            or getattr(voices[fallback_index], "name", "")
        )

    def speak(
        self,
        text: str,
        profile: VoiceProfile,
        stop_event: threading.Event,
    ) -> None:
        if stop_event.is_set():
            return
        engine = self._create_engine()
        with self._lock:
            self._engine = engine
        try:
            engine.setProperty("rate", round(180 * profile.speed))
            engine.setProperty("volume", profile.volume)
            voice_id = self._select_voice(engine, profile)
            if voice_id:
                engine.setProperty("voice", voice_id)
            if stop_event.is_set():
                return
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            finally:
                with self._lock:
                    if self._engine is engine:
                        self._engine = None

    def stop(self) -> None:
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


class _WindowsWavePlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def play_file(self, path: Path, stop_event: threading.Event) -> None:
        if os.name != "nt":
            raise TTSUnavailableError("OpenAI audio playback currently requires Windows")
        import winsound

        if not stop_event.is_set():
            winsound.PlaySound(str(path), winsound.SND_FILENAME)

    def stop(self) -> None:
        if os.name != "nt":
            return
        import winsound

        with self._lock:
            winsound.PlaySound(None, 0)


class OpenAISpeechProvider:
    """Optional request-based OpenAI Speech API provider with local playback."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini-tts",
        player: Any | None = None,
    ) -> None:
        self._client_instance = client
        self._api_key = api_key
        self.model = model
        self._player = player or _WindowsWavePlayer()

    @property
    def disclosure(self) -> str:
        """Text the UI can display whenever an online voice is selected."""

        return OPENAI_TTS_DISCLOSURE

    def _client(self) -> Any:
        if self._client_instance is not None:
            return self._client_instance
        try:
            from openai import OpenAI
        except ImportError as error:
            raise TTSUnavailableError(
                "Online speech requires the optional openai package"
            ) from error
        self._client_instance = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self._client_instance

    def speak(
        self,
        text: str,
        profile: VoiceProfile,
        stop_event: threading.Event,
    ) -> None:
        if stop_event.is_set():
            return
        if contains_prohibited_secret(text):
            raise TTSError("Online speech refuses credential-like text")
        if not profile.voice_id:
            raise TTSError("An OpenAI voice ID is required by this profile")

        request: dict[str, Any] = {
            "model": self.model,
            "voice": profile.voice_id,
            "input": text,
            "response_format": "wav",
            "speed": profile.speed,
        }
        if profile.style.strip():
            request["instructions"] = redact_text(profile.style.strip())

        file_descriptor, raw_path = tempfile.mkstemp(prefix="asher_tts_", suffix=".wav")
        os.close(file_descriptor)
        output_path = Path(raw_path)
        try:
            speech_api = self._client().audio.speech
            streaming = getattr(speech_api, "with_streaming_response", None)
            if streaming is not None:
                with streaming.create(**request) as response:
                    response.stream_to_file(output_path)
            else:
                response = speech_api.create(**request)
                if hasattr(response, "write_to_file"):
                    response.write_to_file(output_path)
                elif hasattr(response, "content"):
                    output_path.write_bytes(bytes(response.content))
                elif hasattr(response, "read"):
                    output_path.write_bytes(bytes(response.read()))
                else:
                    raise TTSError("OpenAI speech response cannot be written to a file")
            if not stop_event.is_set():
                self._player.play_file(output_path, stop_event)
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    def stop(self) -> None:
        try:
            self._player.stop()
        except Exception:
            pass


def build_default_tts(
    *,
    selected_profile: str = "asher_male",
    openai_client: Any | None = None,
) -> TTSManager:
    """Build the standard Windows/offline + optional online provider set."""

    configured_profile = os.getenv("ASHER_VOICE_PROFILE", "").strip()
    if selected_profile == "asher_male" and configured_profile:
        known = {profile.name for profile in DEFAULT_VOICE_PROFILES}
        configured = next(
            (profile for profile in DEFAULT_VOICE_PROFILES if profile.name == configured_profile),
            None,
        )
        if configured_profile in known and not (
            configured is not None
            and configured.provider == "openai"
            and not os.getenv("OPENAI_API_KEY", "").strip()
        ):
            selected_profile = configured_profile
    manager = TTSManager(selected_profile=selected_profile, fallback_profile="asher_male")
    manager.register_provider("sapi", SapiSpeechProvider())
    manager.register_provider(
        "openai",
        OpenAISpeechProvider(
            client=openai_client,
            model=os.getenv("ASHER_OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
            or "gpt-4o-mini-tts",
        ),
    )
    return manager


_default_manager_lock = threading.Lock()
_default_manager: TTSManager | None = None


def get_default_tts() -> TTSManager:
    """Lazily obtain the process-local manager for simple integrations."""

    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = build_default_tts()
        return _default_manager


def speak(
    text: str,
    *,
    voice_profile: str | None = None,
    speed: float | None = None,
    style: str | None = None,
    interrupt: bool = False,
    timeout: float | None = None,
) -> SpeechResult:
    """Convenience blocking interface backed by the same provider registry."""

    return get_default_tts().speak(
        text,
        profile_name=voice_profile,
        speed=speed,
        style=style,
        interrupt=interrupt,
        timeout=timeout,
    )


def speak_async(
    text: str,
    *,
    voice_profile: str | None = None,
    speed: float | None = None,
    style: str | None = None,
    interrupt: bool = False,
) -> SpeechHandle:
    return get_default_tts().speak_async(
        text,
        profile_name=voice_profile,
        speed=speed,
        style=style,
        interrupt=interrupt,
    )


__all__ = [
    "DEFAULT_VOICE_PROFILES",
    "OPENAI_TTS_DISCLOSURE",
    "OPENAI_VOICE_IDS",
    "OpenAISpeechProvider",
    "OpenAITTSProvider",
    "SAPIProvider",
    "SapiSpeechProvider",
    "SpeechHandle",
    "SpeechProvider",
    "SpeechResult",
    "TTSError",
    "TTSManager",
    "TTSService",
    "TTSUnavailableError",
    "UnknownVoiceProfile",
    "VoiceProfile",
    "VoiceProfileRegistry",
    "build_default_tts",
    "get_default_tts",
    "speak",
    "speak_async",
]

# Small compatibility aliases make the provider boundary easy to discover for
# callers that use the shorter service/provider terminology.
TTSService = TTSManager
SAPIProvider = SapiSpeechProvider
OpenAITTSProvider = OpenAISpeechProvider
