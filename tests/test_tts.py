"""Deterministic tests for ASHER's provider-independent TTS boundary."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from asher.voice.tts import (
    DEFAULT_VOICE_PROFILES,
    OpenAISpeechProvider,
    SapiSpeechProvider,
    TTSManager,
    TTSError,
    TTSUnavailableError,
    VoiceProfile,
    VoiceProfileRegistry,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, VoiceProfile]] = []
        self.stop_calls = 0

    def speak(self, text: str, profile: VoiceProfile, stop_event: threading.Event) -> None:
        self.calls.append((text, profile))

    def stop(self) -> None:
        self.stop_calls += 1


class BlockingProvider(RecordingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def speak(self, text: str, profile: VoiceProfile, stop_event: threading.Event) -> None:
        self.calls.append((text, profile))
        self.started.set()
        while not stop_event.wait(0.01):
            pass


class FailingProvider(RecordingProvider):
    def speak(self, text: str, profile: VoiceProfile, stop_event: threading.Event) -> None:
        raise ConnectionError("offline fixture")


class FakeEngine:
    def __init__(self) -> None:
        self.properties: dict[str, Any] = {}
        self.voices = [
            type("Voice", (), {"id": "voice-m", "name": "Test Male", "gender": "male"})(),
            type("Voice", (), {"id": "voice-f", "name": "Test Female", "gender": "female"})(),
        ]
        self.spoken: list[str] = []
        self.stopped = 0

    def setProperty(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def getProperty(self, key: str) -> Any:
        return self.voices if key == "voices" else self.properties.get(key)

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def runAndWait(self) -> None:
        return None

    def stop(self) -> None:
        self.stopped += 1


class FakeStreamingResponse:
    def __init__(self, payload: bytes = b"RIFF-fake-wav") -> None:
        self.payload = payload

    def __enter__(self) -> "FakeStreamingResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def stream_to_file(self, path: Path) -> None:
        Path(path).write_bytes(self.payload)


class FakeSpeechEndpoint:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.with_streaming_response = self

    def create(self, **kwargs: Any) -> FakeStreamingResponse:
        self.requests.append(kwargs)
        return FakeStreamingResponse()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.audio = type("Audio", (), {})()
        self.audio.speech = FakeSpeechEndpoint()


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []
        self.stopped = 0

    def play_file(self, path: Path, _stop_event: threading.Event) -> None:
        self.played.append(Path(path))
        self.last_bytes = Path(path).read_bytes()

    def stop(self) -> None:
        self.stopped += 1


class TTSProviderTests(unittest.TestCase):
    def test_registry_has_runtime_switchable_profiles(self) -> None:
        registry = VoiceProfileRegistry(DEFAULT_VOICE_PROFILES)
        self.assertIn("offline_male", registry.names())
        self.assertIn("offline_female", registry.names())
        self.assertEqual(registry.get("online_male").provider, "openai")

    def test_manager_switches_profile_without_restarting(self) -> None:
        registry = VoiceProfileRegistry(
            (
                VoiceProfile("male", "Male", "fake", gender_hint="male"),
                VoiceProfile("female", "Female", "fake", gender_hint="female"),
            )
        )
        provider = RecordingProvider()
        manager = TTSManager(registry, selected_profile="male")
        manager.register_provider("fake", provider)
        first = manager.speak("first")
        manager.set_profile("female")
        second = manager.speak("second")
        manager.close(wait=True)
        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual([profile.name for _, profile in provider.calls], ["male", "female"])

    def test_stop_interrupts_active_utterance(self) -> None:
        registry = VoiceProfileRegistry((VoiceProfile("test", "Test", "blocking"),))
        provider = BlockingProvider()
        manager = TTSManager(registry, selected_profile="test")
        manager.register_provider("blocking", provider)
        handle = manager.speak_async("long speech")
        self.assertTrue(provider.started.wait(1.0))
        self.assertGreaterEqual(manager.stop(), 1)
        result = handle.result(1.0)
        manager.close(wait=True)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.success)
        self.assertGreaterEqual(provider.stop_calls, 1)

    def test_missing_provider_returns_clear_failure(self) -> None:
        manager = TTSManager(
            VoiceProfileRegistry((VoiceProfile("missing", "Missing", "not-installed"),)),
            selected_profile="missing",
        )
        result = manager.speak("hello")
        manager.close(wait=True)
        self.assertFalse(result.success)
        self.assertIn("not configured", result.error or "")

    def test_configured_fallback_profile_recovers_provider_failure(self) -> None:
        registry = VoiceProfileRegistry(
            (
                VoiceProfile("online", "Online", "failing"),
                VoiceProfile("offline", "Offline", "fake"),
            )
        )
        manager = TTSManager(registry, selected_profile="online", fallback_profile="offline")
        manager.register_provider("failing", FailingProvider())
        fallback = RecordingProvider()
        manager.register_provider("fake", fallback)
        result = manager.speak("hello")
        manager.close(wait=True)
        self.assertTrue(result.success)
        self.assertEqual(result.profile_name, "offline")
        self.assertEqual(len(fallback.calls), 1)

    def test_sapi_profile_applies_speed_volume_and_gender(self) -> None:
        engine = FakeEngine()
        provider = SapiSpeechProvider(engine_factory=lambda: engine)
        profile = VoiceProfile(
            "female", "Female", "sapi", gender_hint="female", speed=1.25, volume=0.6
        )
        provider.speak("hello", profile, threading.Event())
        self.assertEqual(engine.spoken, ["hello"])
        self.assertEqual(engine.properties["rate"], 225)
        self.assertEqual(engine.properties["volume"], 0.6)
        self.assertEqual(engine.properties["voice"], "voice-f")

    def test_openai_provider_uses_profile_and_cleans_audio_file(self) -> None:
        client = FakeOpenAIClient()
        player = FakePlayer()
        provider = OpenAISpeechProvider(client=client, player=player)
        profile = VoiceProfile(
            "online", "Online", "openai", voice_id="cedar", speed=1.1, style="calm"
        )
        provider.speak("hello", profile, threading.Event())
        request = client.audio.speech.requests[0]
        self.assertEqual(request["model"], "gpt-4o-mini-tts")
        self.assertEqual(request["voice"], "cedar")
        self.assertEqual(request["input"], "hello")
        self.assertEqual(request["response_format"], "wav")
        self.assertEqual(request["speed"], 1.1)
        self.assertEqual(request["instructions"], "calm")
        self.assertEqual(player.last_bytes, b"RIFF-fake-wav")
        self.assertTrue(player.played)
        self.assertFalse(player.played[0].exists(), "temporary audio must be removed")

    def test_openai_provider_rejects_credential_like_text(self) -> None:
        provider = OpenAISpeechProvider(client=FakeOpenAIClient(), player=FakePlayer())
        profile = VoiceProfile("online", "Online", "openai", voice_id="cedar")
        with self.assertRaises(TTSError):
            provider.speak("password is do-not-send", profile, threading.Event())


if __name__ == "__main__":
    unittest.main()
