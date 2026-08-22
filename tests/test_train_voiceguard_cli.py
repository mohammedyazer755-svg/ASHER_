"""Privacy-safe command-line readiness behavior for VoiceGuard."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import train_voiceguard
from asher.voiceguard import EnrollmentManager


class TrainVoiceGuardCliTests(unittest.TestCase):
    def test_wake_training_payload_reports_separate_runtime_activation(self) -> None:
        report = SimpleNamespace(
            measured=True,
            accuracy=1.0,
            f1=1.0,
            false_accept_rate=0.0,
            false_reject_rate=0.0,
            authorized_identity_accuracy=None,
            authorized_identity_error_count=0,
            authorized_identity_sample_count=0,
            replay_acceptance_rate=None,
            unavailable_conditions=("replay",),
        )
        split = SimpleNamespace(
            train_sessions=("train",),
            validation_sessions=("validation",),
            test_sessions=("test",),
            train=(1,),
            validation=(2,),
            test=(3,),
        )
        result = SimpleNamespace(
            model=SimpleNamespace(
                task="wake_word",
                model_version="a" * 64,
                threshold=0.5,
            ),
            split=split,
            test_report=report,
            artifacts=None,
        )
        readiness = SimpleNamespace(to_dict=lambda: {"task": "wake_word", "ready": True})

        payload = train_voiceguard._training_payload(result, readiness)

        self.assertEqual(payload["runtime_activation"], "wake_word_active")

    def test_empty_private_runtime_returns_actionable_not_ready_status(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as temporary, redirect_stdout(output):
            status = train_voiceguard.main(["--runtime-dir", temporary, "--check"])
            rendered = output.getvalue()

        self.assertEqual(status, train_voiceguard.NOT_READY_EXIT)
        self.assertIn("NOT READY", rendered)
        self.assertIn("No finalized active recording samples", rendered)
        self.assertNotIn(temporary, rendered)

    def test_json_check_contains_aggregate_counts_and_no_private_paths(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as temporary, redirect_stdout(output):
            status = train_voiceguard.main(
                ["--runtime-dir", temporary, "--check", "--json"]
            )
            rendered = output.getvalue()

        self.assertEqual(status, train_voiceguard.NOT_READY_EXIT)
        payload = json.loads(rendered)
        self.assertFalse(payload["trained"])
        self.assertFalse(payload["readiness"]["ready"])
        self.assertEqual(payload["readiness"]["session_count"], 0)
        self.assertNotIn(temporary, rendered)
        self.assertNotIn("\\recordings\\", rendered.casefold())
        self.assertNotIn("/recordings/", rendered.casefold())

    def test_malformed_manifest_error_does_not_expose_private_session_id(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as temporary:
            manager = EnrollmentManager(Path(temporary) / "voiceguard")
            session = manager.begin_enrollment(
                "private-speaker-id",
                role="owner",
                environment="private-room",
                consent=True,
            )
            session.add_pcm16(
                [700, -700] * 32,
                contains_wake_phrase=False,
            )
            manager.finalize_enrollment(session)
            session.manifest_path.write_text("{invalid", encoding="utf-8")
            with redirect_stdout(output):
                status = train_voiceguard.main(
                    ["--runtime-dir", temporary, "--check", "--json"]
                )
        rendered = output.getvalue()
        self.assertEqual(status, 1)
        self.assertNotIn(session.manifest.session_id, rendered)
        self.assertNotIn("private-speaker-id", rendered)
        self.assertNotIn("private-room", rendered)


if __name__ == "__main__":
    unittest.main()
