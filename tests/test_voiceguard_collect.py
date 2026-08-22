from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import voiceguard_collect as collector
from asher.voiceguard import SpeakerRole


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.manifest = SimpleNamespace(session_id="session-1")

    def record_microphone(self, duration: float, **kwargs: object) -> None:
        self.calls.append({"duration": duration, **kwargs})


class _FakeManager:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.begin: dict[str, object] | None = None
        self.finalized_minimum: int | None = None
        self.readiness_config = None

    def begin_enrollment(self, speaker_id: str, *, role: SpeakerRole, environment: str, consent: bool):
        self.begin = {
            "speaker_id": speaker_id,
            "role": role,
            "environment": environment,
            "consent": consent,
        }
        return self.session

    def finalize_enrollment(self, session: _FakeSession, *, minimum_samples: int):
        self.finalized_minimum = minimum_samples
        return SimpleNamespace(session_ids=("a", "b", "c"))

    def assess_training_readiness(self, *, config=None):
        self.readiness_config = config
        return SimpleNamespace(
            ready=False,
            session_count=3,
            sample_count=12,
            class_count=2,
        )


class VoiceGuardCollectorTests(unittest.TestCase):
    def test_unknown_identity_is_never_authorized(self) -> None:
        config = SimpleNamespace()
        identity = collector.resolve_identity(config, speaker="unknown", speaker_id="unknown_pool")
        self.assertEqual(identity.role, SpeakerRole.UNKNOWN)
        self.assertFalse(identity.expected_authorized)
        self.assertEqual(identity.speaker_id, "unknown_pool")

    def test_collection_creates_one_session_with_multiple_clips(self) -> None:
        manager = _FakeManager()
        identity = collector.CollectionIdentity("owner-id", SpeakerRole.OWNER, True, "Owner")
        prompts: list[str] = []
        messages: list[str] = []

        session_id, session_count = collector.collect_session(
            manager,
            identity,
            environment="quiet_room",
            samples=4,
            duration=2.5,
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=messages.append,
        )

        self.assertEqual(session_id, "session-1")
        self.assertEqual(session_count, 3)
        self.assertEqual(len(manager.session.calls), 4)
        self.assertEqual(manager.finalized_minimum, 4)
        self.assertEqual(manager.begin["speaker_id"], "owner-id")
        self.assertTrue(manager.begin["consent"])
        self.assertTrue(all(call["expected_authorized"] for call in manager.session.calls))
        self.assertTrue(all(call["contains_wake_phrase"] is False for call in manager.session.calls))
        self.assertEqual(len(prompts), 4)

    def test_wake_positive_session_uses_literal_positive_prompts_and_labels(self) -> None:
        manager = _FakeManager()
        identity = collector.CollectionIdentity("owner-id", SpeakerRole.OWNER, True, "Owner")
        prompts: list[str] = []

        collector.collect_session(
            manager,
            identity,
            environment="quiet_room",
            samples=4,
            duration=2.5,
            task="wake_word",
            wake_label="positive",
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lambda _message: None,
        )

        self.assertTrue(
            all(call["contains_wake_phrase"] is True for call in manager.session.calls)
        )
        self.assertTrue(all("hey asher" in prompt.casefold() for prompt in prompts))

    def test_wake_negative_session_uses_literal_negative_prompts_and_labels(self) -> None:
        manager = _FakeManager()
        identity = collector.CollectionIdentity("owner-id", SpeakerRole.OWNER, True, "Owner")
        prompts: list[str] = []

        collector.collect_session(
            manager,
            identity,
            environment="quiet_room",
            samples=4,
            duration=2.5,
            task="wake_word",
            wake_label="negative",
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lambda _message: None,
        )

        self.assertTrue(
            all(call["contains_wake_phrase"] is False for call in manager.session.calls)
        )
        self.assertTrue(all("hey asher" not in prompt.casefold() for prompt in prompts))

    def test_wake_collection_requires_an_explicit_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "--wake-label is required"):
            collector.collection_prompt_plan("wake_word", None)
        with self.assertRaisesRegex(ValueError, "only valid"):
            collector.collection_prompt_plan("speaker_auth", "positive")

    def test_cli_uses_selected_task_for_labels_and_readiness(self) -> None:
        manager = _FakeManager()
        identity = collector.CollectionIdentity("owner-id", SpeakerRole.OWNER, True, "Owner")
        config = SimpleNamespace(runtime=SimpleNamespace(root=Path("fixture-runtime")))
        with (
            patch.object(collector, "load_dotenv"),
            patch.object(collector.AsherConfig, "load", return_value=config),
            patch.object(collector, "resolve_identity", return_value=identity),
            patch.object(collector, "EnrollmentManager", return_value=manager),
            patch.object(
                collector,
                "collect_session",
                return_value=("session-1", 3),
            ) as collect,
            patch("builtins.print"),
        ):
            result = collector.main(
                [
                    "--consent",
                    "--task",
                    "wake_word",
                    "--wake-label",
                    "positive",
                    "--samples",
                    "3",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(collect.call_args.kwargs["task"], "wake_word")
        self.assertEqual(collect.call_args.kwargs["wake_label"], "positive")
        self.assertEqual(manager.readiness_config.task, "wake_word")

    def test_collection_bounds_prevent_tiny_or_runaway_sessions(self) -> None:
        with self.assertRaises(ValueError):
            collector.validate_collection_settings(2, 3.0, "quiet")
        with self.assertRaises(ValueError):
            collector.validate_collection_settings(6, 20.0, "quiet")
        with self.assertRaises(ValueError):
            collector.validate_collection_settings(6, 3.0, "   ")

    def test_readiness_summary_is_aggregate_only(self) -> None:
        readiness = SimpleNamespace(
            ready=False,
            session_count=2,
            sample_count=12,
            class_count=1,
        )
        rendered = collector.readiness_summary(readiness)
        self.assertEqual(
            rendered,
            "VoiceGuard dataset: NOT READY; sessions=2, samples=12, classes=1.",
        )
        self.assertNotIn("speaker", rendered.casefold())


if __name__ == "__main__":
    unittest.main()
