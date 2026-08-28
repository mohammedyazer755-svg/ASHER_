"""Wake boundary and dependency-free VAD tests."""

from __future__ import annotations

import array
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from asher.agent.controller import CompanionReply
from asher.config import AsherConfig
from asher.voice.capture import (
    AudioFrame,
    CapturedTurn,
    TurnCapture,
    VadConfig,
    VoiceActivityDetector,
)
from asher.voice.wakeword import (
    LazyOpenWakeWordDetector,
    TextWakeDetector,
    match_wake_phrase,
)
from asher.voice.runtime import COMMAND_TURN_VAD, VoiceRuntime, normalized_pcm16_rms
from asher.voice.types import TranscriptResult


def pcm(amplitude: float, samples: int = 320) -> bytes:
    value = max(-1.0, min(1.0, amplitude))
    data = array.array("h", [int(value * 32767)] * samples)
    return data.tobytes()


class WakeCaptureTests(unittest.TestCase):
    def test_pcm16_rms_is_normalized_bounded_and_tolerates_partial_samples(self) -> None:
        self.assertEqual(normalized_pcm16_rms(pcm(0.0)), 0.0)
        self.assertAlmostEqual(normalized_pcm16_rms(pcm(0.5)), 0.5, places=3)
        self.assertLessEqual(normalized_pcm16_rms(pcm(-1.0)), 1.0)
        self.assertGreater(normalized_pcm16_rms(pcm(-1.0)), 0.99)
        self.assertEqual(normalized_pcm16_rms(b"\x00"), 0.0)

    def test_runtime_level_comes_from_frames_without_callback_or_pcm_publication(self) -> None:
        class BlockingFrames:
            def __init__(self) -> None:
                self.calls = 0
                self.capture_waiting = threading.Event()
                self.release = threading.Event()

            def __iter__(self):
                return self

            def __next__(self):
                self.calls += 1
                if self.calls == 1:
                    return AudioFrame(pcm(0.5))
                self.capture_waiting.set()
                self.release.wait(1.0)
                raise StopIteration

        class QuietTTS:
            @staticmethod
            def stop() -> int:
                return 0

        with tempfile.TemporaryDirectory() as directory:
            controller = SimpleNamespace(config=AsherConfig.load(directory))
            events = []
            runtime = VoiceRuntime(
                controller,
                transcriber=object(),
                tts=QuietTTS(),
                on_event=events.append,
            )
            frames = BlockingFrames()
            worker = threading.Thread(target=runtime._capture_trigger, args=(frames,))
            worker.start()
            try:
                self.assertTrue(frames.capture_waiting.wait(1.0))
                self.assertAlmostEqual(runtime.microphone_level, 0.5, places=3)
                self.assertEqual(events, [])
                self.assertFalse(hasattr(runtime, "pcm16"))
            finally:
                frames.release.set()
                worker.join(1.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(runtime.microphone_level, 0.0)

    def test_runtime_level_decays_on_silence_and_resets_on_stop_and_error(self) -> None:
        class QuietTTS:
            @staticmethod
            def stop() -> int:
                return 0

        class FailingBackend:
            @staticmethod
            def frames(_cancellation=None):
                raise RuntimeError("fixture microphone failure")

        with tempfile.TemporaryDirectory() as directory:
            controller = SimpleNamespace(config=AsherConfig.load(directory))
            runtime = VoiceRuntime(
                controller,
                backend=FailingBackend(),
                transcriber=object(),
                tts=QuietTTS(),
            )
            runtime._observe_microphone_frame(pcm(0.8), speech_detected=True)
            previous = runtime.microphone_level
            runtime._observe_microphone_frame(pcm(0.0), speech_detected=False)
            self.assertLess(runtime.microphone_level, previous)
            for _ in range(8):
                runtime._observe_microphone_frame(pcm(0.0), speech_detected=False)
            self.assertEqual(runtime.microphone_level, 0.0)

            runtime._observe_microphone_frame(pcm(0.4), speech_detected=True)
            runtime.stop()
            self.assertEqual(runtime.microphone_level, 0.0)

            error_runtime = VoiceRuntime(
                controller,
                backend=FailingBackend(),
                transcriber=object(),
                tts=QuietTTS(),
            )
            error_runtime._observe_microphone_frame(pcm(0.4), speech_detected=True)
            with self.assertRaisesRegex(RuntimeError, "fixture microphone failure"):
                error_runtime.run_forever()
            self.assertEqual(error_runtime.microphone_level, 0.0)

    def test_wake_matching_uses_word_boundaries(self) -> None:
        self.assertFalse(match_wake_phrase("washer").detected)
        match = match_wake_phrase("Noise, Hey Asher: search the contact")
        self.assertTrue(match.detected)
        self.assertEqual(match.command, "search the contact")
        self.assertTrue(match_wake_phrase("hey, asher").detected)
        self.assertTrue(TextWakeDetector().detect("hey asher").detected)

    def test_optional_openwakeword_is_lazy(self) -> None:
        calls: list[Path] = []

        class FakeModel:
            def predict(self, _sample):
                return {"hey_asher": [0.2, 0.91]}

        def factory(path: Path):
            calls.append(path)
            return FakeModel()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wake.onnx"
            path.write_bytes(b"fixture")
            detector = LazyOpenWakeWordDetector(path, model_factory=factory)
            self.assertFalse(detector.loaded)
            self.assertTrue(detector.detect(b"audio").detected)
            self.assertTrue(detector.loaded)
            self.assertEqual(calls, [path])

    def test_vad_turn_capture_stops_after_silence(self) -> None:
        config = VadConfig(
            frame_duration_ms=20,
            start_consecutive_frames=2,
            end_silence_ms=60,
            pre_roll_ms=20,
            max_turn_ms=500,
        )
        vad = VoiceActivityDetector(config)
        vad.calibrate([AudioFrame(pcm(0.001)) for _ in range(3)])
        capture = TurnCapture(vad)
        frames = [
            AudioFrame(pcm(0.001)),
            AudioFrame(pcm(0.2)),
            AudioFrame(pcm(0.2)),
            AudioFrame(pcm(0.2)),
            AudioFrame(pcm(0.001)),
            AudioFrame(pcm(0.001)),
            AudioFrame(pcm(0.001)),
            AudioFrame(pcm(0.2)),  # must not be included after end silence
        ]
        result = capture.capture(frames)
        self.assertTrue(result.speech_started)
        self.assertTrue(result.ended_on_silence)
        self.assertLess(result.frame_count, len(frames))
        self.assertGreater(len(result.pcm16), 0)

    def test_vad_without_speech_is_empty(self) -> None:
        config = VadConfig(start_consecutive_frames=2)
        vad = VoiceActivityDetector(config)
        vad.calibrate([AudioFrame(pcm(0.001))])
        result = TurnCapture(vad).capture([AudioFrame(pcm(0.001)) for _ in range(5)])
        self.assertFalse(result.speech_started)
        self.assertEqual(result.pcm16, b"")

    def test_runtime_command_capture_survives_natural_pause_and_keeps_pre_roll(self) -> None:
        class QuietTTS:
            @staticmethod
            def stop() -> int:
                return 0

        with tempfile.TemporaryDirectory() as directory:
            controller = SimpleNamespace(config=AsherConfig.load(directory))
            runtime = VoiceRuntime(
                controller,
                transcriber=object(),
                tts=QuietTTS(),
            )
            frames = iter(
                [AudioFrame(pcm(0.001)) for _ in range(6)]
                + [AudioFrame(pcm(0.22)) for _ in range(4)]
                # 600 ms pause: longer than the old 550 ms endpoint, but
                # intentionally shorter than VOICE-2C command endpointing.
                + [AudioFrame(pcm(0.001)) for _ in range(30)]
                + [AudioFrame(pcm(0.31)) for _ in range(4)]
                + [AudioFrame(pcm(0.001)) for _ in range(COMMAND_TURN_VAD.end_silence_frames)]
                + [AudioFrame(pcm(0.45)) for _ in range(3)]
            )

            turn = runtime._capture_trigger(
                frames,
                vad_config=COMMAND_TURN_VAD,
            )

        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertTrue(turn.speech_started)
        self.assertTrue(turn.ended_on_silence)
        self.assertIn(pcm(0.22), turn.pcm16)
        self.assertIn(pcm(0.31), turn.pcm16)
        self.assertNotIn(pcm(0.45), turn.pcm16)

    def test_standalone_wake_keeps_next_utterance_active_and_speaks_reply(self) -> None:
        transcripts = iter(("hey asher", "open chrome"))

        class FakeTranscriber:
            def transcribe(self, _audio, **_kwargs):
                text = next(transcripts)
                return TranscriptResult(
                    raw_text=text,
                    normalized_text=text,
                    acoustic_confidence=0.95,
                    no_speech_probability=0.01,
                )

        class FakeTTS:
            def __init__(self) -> None:
                self.spoken: list[str] = []
                self.stop_calls = 0

            def speak_async(self, text: str, **_kwargs):
                self.spoken.append(text)
                return object()

            def stop(self) -> int:
                self.stop_calls += 1
                return 0

        class FakeUsers:
            @staticmethod
            def get(_user_id):
                return None

        class FakeController:
            def __init__(self, config) -> None:
                self.config = config
                self.users = FakeUsers()
                self.commands: list[tuple[str, object]] = []

            @staticmethod
            def create_guest_session():
                return SimpleNamespace(session_id="guest")

            def handle_text(self, command, session):
                self.commands.append((command, session))
                return CompanionReply("Opening Chrome in dry-run mode.")

        class EmptyBackend:
            @staticmethod
            def frames(_cancellation=None):
                return iter(())

        class ScriptedRuntime(VoiceRuntime):
            def __init__(self, *args, **kwargs):
                self.turns = [
                    CapturedTurn(b"\x01\x00" * 320, 16_000, 1, True, True),
                    CapturedTurn(b"\x02\x00" * 320, 16_000, 1, True, True),
                ]
                super().__init__(*args, **kwargs)

            def _capture_trigger(self, _frames, *, deadline=None, vad_config=None):
                del deadline, vad_config
                if self.turns:
                    return self.turns.pop(0)
                self.stop()
                return None

        with tempfile.TemporaryDirectory() as directory:
            config = AsherConfig.load(directory)
            controller = FakeController(config)
            tts = FakeTTS()
            events = []
            runtime = ScriptedRuntime(
                controller,
                backend=EmptyBackend(),
                transcriber=FakeTranscriber(),
                tts=tts,
                on_event=events.append,
            )
            runtime.run_forever()

        self.assertEqual([item[0] for item in controller.commands], ["open chrome"])
        self.assertIn("Yes?", tts.spoken)
        self.assertIn("Opening Chrome in dry-run mode.", tts.spoken)
        kinds = [event.kind for event in events]
        self.assertEqual(kinds.count("transcript"), 1)
        self.assertIn("listening", set(kinds))
        self.assertIn("reply", set(kinds))


if __name__ == "__main__":
    unittest.main()
