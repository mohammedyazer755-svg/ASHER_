from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluate_voice_accuracy import (
    SafeTaskEvaluator,
    corpus_inventory,
    evaluate_manifest,
    initialize_manifest,
    main,
)
from asher.voice.evaluation import VoiceFixture, write_fixture_manifest


class VoiceAccuracyWorkflowTests(unittest.TestCase):
    def test_safe_task_evaluator_uses_real_dry_run_and_confirmation_boundaries(self) -> None:
        fixtures = (
            VoiceFixture("open", "open notepad", "open_app"),
            VoiceFixture(
                "send",
                "send the project update to Avery Stone",
                "send_whatsapp",
                expected_contact="Avery Stone",
            ),
        )
        with SafeTaskEvaluator(fixtures) as evaluator:
            self.assertEqual(evaluator.interpret("open notepad"), ("open_app", None, True))
            intent, contact, success = evaluator.interpret(
                "send the project update to Avery Stone"
            )
            self.assertEqual((intent, contact, success), ("send_whatsapp", "Avery Stone", True))
            self.assertEqual(evaluator.interpret("unrecognized fixture"), (None, None, False))

    def test_initialize_and_check_are_aggregate_and_model_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "fixtures.jsonl"
            initialize_manifest(manifest, count=6)
            inventory = corpus_inventory(manifest)
            self.assertEqual(inventory["fixture_count"], 6)
            self.assertEqual(inventory["recorded_count"], 0)
            self.assertFalse(inventory["ready"])

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--manifest", str(manifest), "--json", "check"])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertNotIn(str(Path(directory)), output.getvalue())
            self.assertEqual(payload["missing_count"], 6)

    def test_manifest_evaluation_measures_task_success_and_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = (
                VoiceFixture("one", "open notepad", "open_app", audio_path="audio/one.wav", condition="quiet"),
                VoiceFixture("two", "toggle mute", "toggle_mute", audio_path="audio/two.wav", condition="fan-noise"),
            )
            manifest = write_fixture_manifest(fixtures, root / "fixtures.jsonl")
            (root / "audio").mkdir()
            (root / "audio" / "one.wav").write_bytes(b"fixture")
            (root / "audio" / "two.wav").write_bytes(b"fixture")
            transcripts = {"one.wav": "open notepad", "two.wav": "wrong"}

            report = evaluate_manifest(
                manifest,
                transcribe=lambda path: transcripts[path.name],
                interpret=lambda text: (
                    ("open_app", None, True)
                    if text == "open notepad"
                    else ("unknown", None, False)
                ),
            )
            self.assertEqual(report.sample_count, 2)
            self.assertEqual(report.task_case_count, 2)
            self.assertEqual(report.task_success_rate, 0.5)
            self.assertEqual(set(report.by_condition), {"quiet", "fan-noise"})

    def test_audio_path_traversal_is_rejected_without_reading_external_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_fixture_manifest(
                (
                    VoiceFixture(
                        "escape",
                        "open notepad",
                        "open_app",
                        audio_path="../outside.wav",
                        condition="quiet",
                    ),
                ),
                root / "fixtures.jsonl",
            )
            (root.parent / "outside.wav").write_bytes(b"do-not-read")
            called = False

            def transcribe(_path: Path) -> str:
                nonlocal called
                called = True
                return "open notepad"

            report = evaluate_manifest(
                manifest,
                transcribe=transcribe,
                interpret=lambda _text: ("open_app", None, True),
            )
            self.assertFalse(called)
            self.assertEqual(report.missing_prediction_count, 1)


if __name__ == "__main__":
    unittest.main()
