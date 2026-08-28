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
    def __init__(self, *args, scripted_turns=None, **kwargs) -> None:
        self.turns = list(
            scripted_turns
            or [CapturedTurn(b"\x01\x00" * 320, 16_000, 1, True, True)]
        )
        super().__init__(*args, **kwargs)

    def _capture_trigger(self, _frames, *, deadline=None, vad_config=None):
        del deadline, vad_config
        if self.turns:
            return self.turns.pop(0)
        self.stop()
        return None


def _run_one_turn(
    directory: str,
    transcript: str,
    binding: WakeWordModelBinding,
    speaker: _SpeakerVerifier,
    *,
    turn_count: int = 1,
    acoustic_confidence: float = 0.95,
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
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, _audio, **_kwargs):
            self.calls += 1
            return TranscriptResult(
                raw_text=transcript,
                normalized_text=transcript,
                acoustic_confidence=acoustic_confidence,
                no_speech_probability=0.01,
            )

    class Backend:
        @staticmethod
        def frames(_cancellation=None):
            return iter(())

    controller = Controller()
    events = []
    transcriber = Transcriber()
    scripted_turns = [
        CapturedTurn(bytes([index + 1, 0]) * 320, 16_000, 1, True, True)
        for index in range(turn_count)
    ]
    runtime = _ScriptedRuntime(
        controller,
        backend=Backend(),
        transcriber=transcriber,
        wake_word_binding=binding,
        voiceguard=speaker,
        tts=_FakeTTS(),
        on_event=events.append,
        scripted_turns=scripted_turns,
    )
    runtime.run_forever()
    controller.transcriber_calls = transcriber.calls
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
    def test_trained_audio_wake_does_not_require_an_exact_whisper_phrase(self) -> None:
        wake = _WakeVerifier(True)
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "open chrome",
                WakeWordModelBinding(True, verifier=wake),
                speaker,
                turn_count=2,
            )

        self.assertEqual(wake.calls, 1)
        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.transcriber_calls, 1)
        self.assertEqual(
            [command for command, _session in controller.commands],
            ["open chrome"],
        )
        kinds = [event.kind for event in events]
        self.assertEqual(kinds.count("transcript"), 1)
        self.assertLess(kinds.index("wake_detected"), kinds.index("authenticated"))
        self.assertLess(kinds.index("authenticated"), kinds.index("transcribing"))

    def test_wake_acceptance_precedes_separate_speaker_authentication(self) -> None:
        wake = _WakeVerifier(True)
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey asher open chrome",
                WakeWordModelBinding(True, verifier=wake),
                speaker,
                turn_count=2,
            )

        self.assertEqual(wake.calls, 1)
        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.transcriber_calls, 1)
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
        self.assertEqual(controller.transcriber_calls, 0)
        self.assertEqual(controller.commands, [])
        self.assertIn("wake_rejected", {event.kind for event in events})

    def test_text_wake_is_an_explicit_fallback_only_without_an_audio_artifact(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey asher open chrome",
                WakeWordModelBinding(False),
                speaker,
            )

        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.transcriber_calls, 1)
        self.assertEqual(
            [command for command, _session in controller.commands],
            ["open chrome"],
        )
        kinds = [event.kind for event in events]
        self.assertIn("wake_fallback", kinds)
        self.assertIn("wake_detected", kinds)
        self.assertLess(kinds.index("transcribing"), kinds.index("wake_detected"))

    def test_low_confidence_exact_text_wake_still_activates_without_executing(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey asher open chrome",
                WakeWordModelBinding(False),
                speaker,
                acoustic_confidence=0.30,
            )

        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.transcriber_calls, 1)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertIn("wake_detected", kinds)
        self.assertIn("authenticated", kinds)
        self.assertIn("listening", kinds)
        self.assertNotIn("transcript", kinds)

    def test_low_confidence_non_wake_stays_in_standby(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "ordinary background speech",
                WakeWordModelBinding(False),
                speaker,
                acoustic_confidence=0.30,
            )

        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertNotIn("wake_detected", kinds)
        self.assertIn("wake_fallback_rejected", kinds)

    def test_exact_hey_asher_wakes(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey asher open chrome",
                WakeWordModelBinding(False),
                speaker,
            )
        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.commands, [("open chrome", SimpleNamespace(actor=SimpleNamespace(user_id="owner-id", role=SimpleNamespace(value="owner"))))])
        kinds = [event.kind for event in events]
        self.assertIn("wake_detected", kinds)

    def test_punctuation_variants_wake(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "Hey, Asher! open chrome",
                WakeWordModelBinding(False),
                speaker,
            )
        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.commands, [("open chrome", SimpleNamespace(actor=SimpleNamespace(user_id="owner-id", role=SimpleNamespace(value="owner"))))])
        kinds = [event.kind for event in events]
        self.assertIn("wake_detected", kinds)

    def test_close_wake_name_distortion_wakes(self) -> None:
        for distorted_phrase in ("hey ashir open chrome", "hey ashire open chrome", "hey ahsher open chrome", "hey usher open chrome"):
            speaker = _SpeakerVerifier()
            with TemporaryDirectory() as directory:
                controller, events = _run_one_turn(
                    directory,
                    distorted_phrase,
                    WakeWordModelBinding(False),
                    speaker,
                )
            self.assertEqual(speaker.calls, 1)
            self.assertEqual(controller.commands, [("open chrome", SimpleNamespace(actor=SimpleNamespace(user_id="owner-id", role=SimpleNamespace(value="owner"))))])
            kinds = [event.kind for event in events]
            self.assertIn("wake_detected", kinds)
            wake_detected_event = next(e for e in events if e.kind == "wake_detected")
            self.assertIn("text-boundary-fuzzy", wake_detected_event.message)

    def test_fuzzy_recovery_only_works_at_turn_start(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "please hey ashir open chrome",
                WakeWordModelBinding(False),
                speaker,
            )
        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertNotIn("wake_detected", kinds)


    def test_unrelated_speech_does_not_wake(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "open chrome",
                WakeWordModelBinding(False),
                speaker,
            )
        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertNotIn("wake_detected", kinds)

    def test_i_should_explain_machine_learning_does_not_wake(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "I should explain machine learning",
                WakeWordModelBinding(False),
                speaker,
            )
        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertNotIn("wake_detected", kinds)

    def test_they_ushered_us_into_the_room_does_not_wake(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "They ushered us into the room",
                WakeWordModelBinding(False),
                speaker,
            )
        self.assertEqual(speaker.calls, 0)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertNotIn("wake_detected", kinds)

    def test_aesthetic_confusions_whole_utterance_only(self) -> None:
        for confusion in ("ahshel", "yeah sir", "yeah i should", "he has it"):
            speaker = _SpeakerVerifier()
            with TemporaryDirectory() as directory:
                controller, events = _run_one_turn(
                    directory,
                    confusion,
                    WakeWordModelBinding(False),
                    speaker,
                )
            self.assertEqual(speaker.calls, 1)
            kinds = [event.kind for event in events]
            self.assertIn("wake_detected", kinds)

        for sentence in ("Yeah sir, that is correct", "I should explain machine learning", "Ashely open calculator"):
            speaker = _SpeakerVerifier()
            with TemporaryDirectory() as directory:
                controller, events = _run_one_turn(
                    directory,
                    sentence,
                    WakeWordModelBinding(False),
                    speaker,
                )
            self.assertEqual(speaker.calls, 0)
            kinds = [event.kind for event in events]
            self.assertNotIn("wake_detected", kinds)

    def test_low_confidence_fuzzy_text_wake_still_activates_without_executing(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "hey ashir open chrome",
                WakeWordModelBinding(False),
                speaker,
                acoustic_confidence=0.30,
            )

        self.assertEqual(speaker.calls, 1)
        self.assertEqual(controller.transcriber_calls, 1)
        self.assertEqual(controller.commands, [])
        kinds = [event.kind for event in events]
        self.assertIn("wake_detected", kinds)
        self.assertIn("authenticated", kinds)
        self.assertIn("listening", kinds)
        self.assertNotIn("transcript", kinds)

    def test_no_conversation_history_for_rejected_wake(self) -> None:
        speaker = _SpeakerVerifier()
        with TemporaryDirectory() as directory:
            controller, events = _run_one_turn(
                directory,
                "ordinary background speech",
                WakeWordModelBinding(False),
                speaker,
            )
        kinds = [event.kind for event in events]
        self.assertNotIn("transcript", kinds)

    def test_capture_trigger_suppresses_tts_and_cooldown(self) -> None:
        from asher.voice.runtime import AudioFrame, VoiceRuntime
        from asher.voice.capture import VadConfig
        from asher.voice.wakeword import TextWakeDetector

        class DummyController:
            config = SimpleNamespace(whisper_model="fixture", whisper_device="cpu", whisper_compute_type="auto")
            users = None
        
        class MockTTS:
            is_speaking = True
            def speak_async(self, text, interrupt=True):
                return None
            def stop(self):
                pass

        class MockBackend:
            def __init__(self):
                self.flushed = False
            def flush(self):
                self.flushed = True
            def frames(self, cancellation=None):
                return iter([])

        controller = DummyController()
        
        current_time = 1000.0
        def mock_clock():
            return current_time

        # Scenario 1: Suppress frames while TTS is speaking
        runtime1 = VoiceRuntime(
            controller,
            backend=MockBackend(),
            wake_detector=TextWakeDetector(),
            tts=MockTTS(),
            clock=mock_clock,
        )
        speech_frame = AudioFrame(b"\xff\x7f" * 320, 16_000)
        frames_list = [speech_frame] * 10
        result = runtime1._capture_trigger(iter(frames_list), vad_config=VadConfig(pre_roll_ms=0))
        self.assertIsNone(result)

        # Scenario 2: Suppress cooldown and trigger transition flushing
        backend2 = MockBackend()
        runtime2 = VoiceRuntime(
            controller,
            backend=backend2,
            wake_detector=TextWakeDetector(),
            tts=MockTTS(),
            clock=mock_clock,
        )
        
        def frame_generator():
            runtime2.tts.is_speaking = True
            for _ in range(5):
                yield speech_frame
            runtime2.tts.is_speaking = False
            yield speech_frame
            for _ in range(5):
                yield speech_frame
            nonlocal current_time
            current_time += 0.5
            for _ in range(15):
                yield speech_frame

        result = runtime2._capture_trigger(frame_generator(), vad_config=VadConfig(pre_roll_ms=0, start_consecutive_frames=2, end_silence_ms=500))
        self.assertTrue(backend2.flushed)
        self.assertIsNotNone(result)
        self.assertTrue(result.speech_started)


    def test_vad_rejects_ambient_bursts_but_accepts_speech(self) -> None:
        from asher.voice.capture import AudioFrame, VoiceActivityDetector, TurnCapture, VadConfig
        
        silence_frame = AudioFrame(b"\x00\x00" * 320, 16_000)
        speech_frame = AudioFrame(b"\xff\x7f" * 320, 16_000)
        
        vad = VoiceActivityDetector(VadConfig(absolute_threshold=0.012))
        capture = TurnCapture(vad)
        
        burst_frames = [silence_frame]*5 + [speech_frame]*3 + [silence_frame]*30
        result = capture.capture(burst_frames)
        self.assertFalse(result.speech_started)
        self.assertEqual(result.pcm16, b"")
        
        speech_frames = [silence_frame]*5 + [speech_frame]*10 + [silence_frame]*30
        result = capture.capture(speech_frames)
        self.assertTrue(result.speech_started)
        self.assertNotEqual(result.pcm16, b"")

    def test_emergency_stop_stops_tts_immediately(self) -> None:
        from asher.voice.runtime import VoiceRuntime, TextWakeDetector
        class MockTTS:
            def __init__(self):
                self.stopped = False
            def stop(self):
                self.stopped = True
        
        class DummyController:
            config = SimpleNamespace(whisper_model="fixture", whisper_device="cpu", whisper_compute_type="auto")
            users = None

        controller = DummyController()
        tts = MockTTS()
        runtime = VoiceRuntime(
            controller,
            wake_detector=TextWakeDetector(),
            tts=tts,
        )
        
        runtime.stop()
        self.assertTrue(tts.stopped)


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
