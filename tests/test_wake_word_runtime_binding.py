"""Deterministic standby wake-model and lifecycle-binding tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from asher.agent.controller import CompanionReply
from asher.config import AsherConfig
from asher.voice.capture import CapturedTurn
from asher.voice.runtime import (
    FileWakeWordVerifier,
    VoiceRuntime,
    WakeWordModelBinding,
    load_active_voiceguard_verifier,
    load_active_wake_word_binding,
)
from asher.voice.types import TranscriptResult
from asher.voiceguard import CalibratedVoiceGuardModel, EnrollmentManager, TrainingConfig
from tests.test_voiceguard_readiness import (
    _StaticTrainer,
    _fixture_training_result,
    _fixture_wake_training_result,
    _populate_ready_manager,
)


class _WakeVerifier:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.calls = 0

    def verify(self, _pcm16: bytes, _sample_rate: int) -> tuple[bool, float, str]:
        self.calls += 1
        return self.accepted, 0.91 if self.accepted else 0.09, "fixture wake result"


class _SpeakerVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def authenticate(self, _pcm16: bytes, _sample_rate: int):
        self.calls += 1
        return "owner-id", 0.93, "fixture speaker result"


class _FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak_async(self, text: str, **_kwargs):
        self.spoken.append(text)
        return object()

    @staticmethod
    def stop() -> int:
        return 0


class _ScriptedRuntime(VoiceRuntime):
    def __init__(self, *args, **kwargs) -> None:
        self.turns = [CapturedTurn(b"\x01\x00" * 320, 16_000, 1, True, True)]
        super().__init__(*args, **kwargs)

    def _capture_trigger(self, _frames, *, deadline=None):
        del deadline
        if self.turns:
            return self.turns.pop(0)
        self.stop()
        return None


def _run_one_turn(
    directory: str,
    transcript: str,
    binding: WakeWordModelBinding,
    speaker: _SpeakerVerifier,
):
    actor = SimpleNamespace(user_id="owner-id", role=SimpleNamespace(value="owner"))

    class Users:
        @staticmethod
        def get(user_id):
            return actor if user_id == actor.user_id else None

    class Controller:
        def __init__(self) -> None:
            self.config = AsherConfig.load(directory)
            self.users = Users()
            self.commands: list[tuple[str, object]] = []

        @staticmethod
        def create_voice_session(authenticated_actor):
            return SimpleNamespace(actor=authenticated_actor)

        @staticmethod
        def create_guest_session():
            return SimpleNamespace(actor=None)

        def handle_text(self, command, session):
            self.commands.append((command, session))
            return CompanionReply("fixture reply")

    class Transcriber:
        @staticmethod
        def transcribe(_audio, **_kwargs):
            return TranscriptResult(
                raw_text=transcript,
                normalized_text=transcript,
                acoustic_confidence=0.95,
                no_speech_probability=0.01,
            )

    class Backend:
        @staticmethod
        def frames(_cancellation=None):
            return iter(())

    controller = Controller()
    events = []
    runtime = _ScriptedRuntime(
        controller,
        backend=Backend(),
        transcriber=Transcriber(),
        wake_word_binding=binding,
        voiceguard=speaker,
        tts=_FakeTTS(),
        on_event=events.append,
    )
    runtime.run_forever()
    return controller, events


def _trained_manager(runtime_root: Path) -> EnrollmentManager:
    manager = EnrollmentManager(runtime_root / "voiceguard")
    _populate_ready_manager(manager, wake_word_classes=True)
    manager.retrain(trainer=_StaticTrainer(_fixture_training_result()))
    manager.retrain(
        trainer=_StaticTrainer(_fixture_wake_training_result()),
        config=TrainingConfig(task="wake_word"),
    )
    return manager


def _controller(runtime_root: Path):
    actors = (
        SimpleNamespace(
            user_id="private-owner-97",
            role=SimpleNamespace(value="owner"),
        ),
    )
    return SimpleNamespace(
        config=SimpleNamespace(runtime=SimpleNamespace(root=runtime_root)),
        users=SimpleNamespace(list_active=lambda: actors),
    )


class WakeWordRuntimeBindingTests(unittest.TestCase):
    def test_text_boundary_remains_mandatory_when_trained_model_accepts(self) -> None:
        wake = _WakeVerifier(True)
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "open chrome",
                WakeWordModelBinding(True, verifier=wake),
                speaker,
            )

        self.assertEqual(wake.calls, 0)
        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        self.assertNotIn("wake_detected", {event.kind for event in events})

    def test_wake_acceptance_precedes_separate_speaker_authentication(self) -> None:
        wake = _WakeVerifier(True)
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey asher open chrome",
                WakeWordModelBinding(True, verifier=wake),
                speaker,
            )

        self.assertEqual(wake.calls, 1)
        self.assertEqual(speaker.calls, 1)
        self.assertEqual([command for command, _session in controller.commands], ["open chrome"])
        kinds = [event.kind for event in events]
        self.assertLess(kinds.index("wake_detected"), kinds.index("authenticated"))

    def test_wake_rejection_never_reaches_speaker_authentication(self) -> None:
        wake = _WakeVerifier(False)
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey asher open chrome",
                WakeWordModelBinding(True, verifier=wake),
                speaker,
            )

        self.assertEqual(wake.calls, 1)
        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        self.assertIn("wake_rejected", {event.kind for event in events})

    def test_registry_loads_wake_and_speaker_models_separately(self) -> None:
        with TemporaryDirectory() as directory:
            runtime_root = Path(directory)
            _trained_manager(runtime_root)
            controller = _controller(runtime_root)

            speaker = load_active_voiceguard_verifier(controller)
            wake = load_active_wake_word_binding(controller)

            self.assertIsNotNone(speaker)
            self.assertTrue(wake.active_artifact)
            self.assertIsInstance(wake.verifier, FileWakeWordVerifier)
            assert speaker is not None
            assert isinstance(wake.verifier, FileWakeWordVerifier)
            self.assertNotEqual(speaker.model_path, wake.verifier.model_path)
            self.assertEqual(
                CalibratedVoiceGuardModel.load(speaker.model_path).task,
                "speaker_auth",
            )
            self.assertEqual(wake.verifier._model.task, "wake_word")

            class FixedExtractor:
                metadata = wake.verifier._model.extractor_metadata

                def __init__(self, value: float) -> None:
                    self.value = value

                def extract_wav(self, _path):
                    return (self.value,)

            extractor = FixedExtractor(1.0)
            wake.verifier.extractor = extractor
            accepted, _score, _reason = wake.verify(b"\0\0" * 40, 16_000)
            self.assertTrue(accepted)
            extractor.value = -1.0
            accepted, _score, _reason = wake.verify(b"\0\0" * 40, 16_000)
            self.assertFalse(accepted)

    def test_stale_registry_or_dataset_binding_rejects_without_inference(self) -> None:
        with TemporaryDirectory() as directory:
            runtime_root = Path(directory)
            manager = _trained_manager(runtime_root)
            binding = load_active_wake_word_binding(_controller(runtime_root))
            assert isinstance(binding.verifier, FileWakeWordVerifier)

            registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
            registry["wake_word_model_dataset_fingerprint"] = "0" * 64
            manager.registry_path.write_text(json.dumps(registry), encoding="utf-8")

            accepted, score, _reason = binding.verify(b"\0\0" * 40, 16_000)
            self.assertFalse(accepted)
            self.assertEqual(score, 0.0)
            fresh = load_active_wake_word_binding(_controller(runtime_root))
            self.assertTrue(fresh.active_artifact)
            self.assertIsNone(fresh.verifier)

        with TemporaryDirectory() as directory:
            runtime_root = Path(directory)
            manager = _trained_manager(runtime_root)
            binding = load_active_wake_word_binding(_controller(runtime_root))
            session_id = manager.list_users()[0].session_ids[0]
            manifest_path = manager.recordings_root / session_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["environment"] = "changed-after-wake-activation"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            accepted, score, _reason = binding.verify(b"\0\0" * 40, 16_000)
            self.assertFalse(accepted)
            self.assertEqual(score, 0.0)

    def test_revoked_preloaded_binding_fails_then_absent_artifact_allows_text_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            runtime_root = Path(directory)
            manager = _trained_manager(runtime_root)
            controller = _controller(runtime_root)
            binding = load_active_wake_word_binding(controller)
            manager.revoke_user("private-owner-97")

            accepted, score, _reason = binding.verify(b"\0\0" * 40, 16_000)
            self.assertFalse(accepted)
            self.assertEqual(score, 0.0)

            no_active_artifact = load_active_wake_word_binding(controller)
            self.assertFalse(no_active_artifact.active_artifact)
            accepted, score, _reason = no_active_artifact.verify(b"\0\0" * 40, 16_000)
            self.assertTrue(accepted)
            self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()
