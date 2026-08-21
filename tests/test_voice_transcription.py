"""Fake-provider tests for lazy transcription and confidence gating."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asher.voice.evaluation import (
    VoicePrediction,
    evaluate_predictions,
    generate_non_private_fixtures,
    read_fixture_manifest,
    word_error_rate,
    write_fixture_manifest,
)
from asher.voice.pipeline import PipelineStatus, VoiceAccuracyPipeline
from asher.voice.transcription import (
    FasterWhisperTranscriber,
    TranscriptionConfig,
    TranscriptionError,
)
from asher.voice.types import TranscriptResult
from asher.voice.vocabulary import DynamicVocabulary


class FakeInfo:
    language = "en"
    language_probability = 0.99
    duration = 1.0


class FakeSegment:
    start = 0.0
    end = 1.0
    text = " search A V E R Y "
    avg_logprob = -0.1
    no_speech_prob = 0.02


class FakeModel:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return [FakeSegment()], FakeInfo()


class TranscriptionTests(unittest.TestCase):
    def test_import_and_model_load_are_lazy(self) -> None:
        created: list[tuple[str, str]] = []

        def factory(model, **kwargs):
            created.append((kwargs["device"], kwargs["compute_type"]))
            if kwargs["device"] == "cuda":
                raise RuntimeError("CUDA unavailable")
            return FakeModel()

        transcriber = FasterWhisperTranscriber(
            TranscriptionConfig(device="auto", model_size="fixture"),
            model_factory=factory,
            cuda_probe=lambda: True,
        )
        self.assertFalse(transcriber.loaded)
        result = transcriber.transcribe(b"audio", vocabulary=("Avery Stone",))
        self.assertEqual(transcriber.active_device, "cpu")
        self.assertEqual(created, [("cuda", "float16"), ("cpu", "int8")])
        self.assertEqual(result.raw_text, "search A V E R Y")
        self.assertEqual(result.normalized_text, "search A V E R Y")
        self.assertTrue(result.is_confident(0.5))

    def test_runtime_cuda_failure_retries_cpu(self) -> None:
        created: list[FakeModel] = []

        def factory(_model, **kwargs):
            model = FakeModel(fail=kwargs["device"] == "cuda")
            created.append(model)
            return model

        transcriber = FasterWhisperTranscriber(
            TranscriptionConfig(device="cuda"),
            model_factory=factory,
            cuda_probe=lambda: True,
        )
        result = transcriber.transcribe(b"audio")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(len(created), 2)

    def test_pipeline_blocks_low_confidence_and_preserves_raw(self) -> None:
        class FakeTranscriber:
            def transcribe(self, _audio, **_kwargs):
                return TranscriptResult(
                    raw_text="Send Hello WORLD to A V E R Y!",
                    normalized_text="Send Hello WORLD to A V E R Y",
                    acoustic_confidence=0.2,
                    no_speech_probability=0.05,
                )

        pipeline = VoiceAccuracyPipeline(
            FakeTranscriber(),
            DynamicVocabulary(contacts=("Avery Stone",)),
            confidence_threshold=0.6,
        )
        result = pipeline.process(b"audio", contact_expected=True)
        self.assertEqual(result.status, PipelineStatus.LOW_CONFIDENCE)
        self.assertIsNone(result.executable_command)
        self.assertEqual(result.transcript.raw_text, "Send Hello WORLD to A V E R Y!")

    def test_pipeline_resolves_spelled_contact_after_confidence_gate(self) -> None:
        class FakeTranscriber:
            def transcribe(self, _audio, **_kwargs):
                return TranscriptResult(
                    raw_text="search A-V-E-R-Y",
                    normalized_text="search A-V-E-R-Y",
                    acoustic_confidence=0.9,
                    no_speech_probability=0.01,
                )

        pipeline = VoiceAccuracyPipeline(
            FakeTranscriber(),
            DynamicVocabulary(contacts=("Avery Stone",)),
        )
        result = pipeline.process(b"audio", contact_expected=True)
        self.assertEqual(result.status, PipelineStatus.ACCEPTED)
        self.assertEqual(result.executable_command, "search Avery Stone")

    def test_pipeline_keeps_unknown_plain_search_as_query_without_context(self) -> None:
        class FakeTranscriber:
            def transcribe(self, _audio, **_kwargs):
                return TranscriptResult(
                    raw_text="search weather tomorrow",
                    normalized_text="search weather tomorrow",
                    acoustic_confidence=0.9,
                    no_speech_probability=0.01,
                )

        pipeline = VoiceAccuracyPipeline(
            FakeTranscriber(),
            DynamicVocabulary(contacts=("Avery Stone",)),
        )
        result = pipeline.process(b"audio")
        self.assertEqual(result.status, PipelineStatus.ACCEPTED)
        self.assertEqual(result.executable_command, "search weather tomorrow")

    def test_remote_fallback_is_opt_in_and_keeps_local_raw_transcript(self) -> None:
        class Local:
            def transcribe(self, _audio, **_kwargs):
                return TranscriptResult(
                    raw_text="local words",
                    normalized_text="local words",
                    acoustic_confidence=0.2,
                    no_speech_probability=0.1,
                    provider="local",
                )

        class Fallback:
            def __init__(self):
                self.calls = 0

            def transcribe(self, _audio, **_kwargs):
                self.calls += 1
                return TranscriptResult(
                    raw_text="search weather",
                    normalized_text="search weather",
                    acoustic_confidence=0.8,
                    no_speech_probability=0.1,
                    provider="fallback",
                )

        fallback = Fallback()
        pipeline = VoiceAccuracyPipeline(
            Local(),
            DynamicVocabulary(),
            fallback_transcriber=fallback,
            confidence_threshold=0.6,
        )
        blocked = pipeline.process(b"audio")
        self.assertEqual(blocked.status, PipelineStatus.LOW_CONFIDENCE)
        self.assertEqual(fallback.calls, 0)
        accepted = pipeline.process(b"audio", allow_remote_fallback=True)
        self.assertEqual(accepted.status, PipelineStatus.ACCEPTED)
        self.assertTrue(accepted.used_fallback)
        self.assertEqual(accepted.transcript.metadata["local_raw_text"], "local words")
        denied = pipeline.process(
            b"audio",
            allow_remote_fallback=True,
            security_sensitive=True,
        )
        self.assertEqual(denied.status, PipelineStatus.LOW_CONFIDENCE)


class EvaluationTests(unittest.TestCase):
    def test_wer_and_measured_report(self) -> None:
        self.assertAlmostEqual(word_error_rate("one two three", "one two"), 1 / 3)
        fixtures = generate_non_private_fixtures(4)
        predictions = [
            VoicePrediction(
                fixture_id=fixtures[0].fixture_id,
                transcript=fixtures[0].expected_transcript,
                intent=fixtures[0].expected_intent,
                contact=fixtures[0].expected_contact,
                latency_ms=10,
            ),
            VoicePrediction(
                fixture_id=fixtures[1].fixture_id,
                transcript="wrong command",
                intent="unknown",
                latency_ms=30,
            ),
        ]
        report = evaluate_predictions(fixtures, predictions)
        self.assertEqual(report.sample_count, 4)
        self.assertEqual(report.missing_prediction_count, 2)
        self.assertEqual(report.latency_sample_count, 2)
        self.assertIsNotNone(report.p95_latency_ms)

    def test_fixture_manifest_round_trip(self) -> None:
        fixtures = generate_non_private_fixtures(5)
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture_manifest(fixtures, Path(directory) / "fixtures.jsonl")
            self.assertEqual(read_fixture_manifest(path), fixtures)


if __name__ == "__main__":
    unittest.main()
