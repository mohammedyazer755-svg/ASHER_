from __future__ import annotations

import unittest
from types import SimpleNamespace

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

    def test_collection_bounds_prevent_tiny_or_runaway_sessions(self) -> None:
        with self.assertRaises(ValueError):
            collector.validate_collection_settings(2, 3.0, "quiet")
        with self.assertRaises(ValueError):
            collector.validate_collection_settings(6, 20.0, "quiet")
        with self.assertRaises(ValueError):
            collector.validate_collection_settings(6, 3.0, "   ")


if __name__ == "__main__":
    unittest.main()
