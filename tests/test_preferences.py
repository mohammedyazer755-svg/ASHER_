from __future__ import annotations

import tempfile
import unittest

from asher.agent.controller import CompanionController
from asher.brain.deterministic import ContactResolver
from asher.config import AsherConfig
from asher.preferences import PreferenceStore
from asher.types import AuthMethod


class PreferenceCoreTests(unittest.TestCase):
    def _controller(self, directory: str) -> CompanionController:
        return CompanionController(
            AsherConfig.load(directory),
            contact_resolver=ContactResolver(("Sam Lee",), {"sam": "Sam Lee"}),
        )

    def test_preference_capture_is_opt_in_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)

            self.assertFalse(controller.preference_learning_enabled)
            controller.handle_text("open chrome", session)
            self.assertEqual(controller.list_preference_events(session), ())

            disabled = controller.handle_text("feedback shorter", session)
            self.assertIn("off", disabled.text.casefold())
            self.assertEqual(controller.list_preference_events(session), ())

            controller.handle_text("preference learning on", session)
            self.assertTrue(controller.preference_learning_enabled)
            controller.handle_text("open chrome", session)
            saved = controller.handle_text("feedback shorter", session)
            self.assertIn("saved", saved.text.casefold())

            events = controller.list_preference_events(session)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].feedback_kind, "shorter")
            self.assertEqual(events[0].dimensions, ("brevity",))
            self.assertEqual(events[0].user_text, "open chrome")
            self.assertIn("app.open", events[0].context["tool_names"])

    def test_preferred_reply_creates_a_real_pair_and_consumes_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            controller.handle_text("preference learning on", session)
            controller.handle_text("open chrome", session)

            reply = controller.handle_text(
                "feedback preferred: Chrome is open.",
                session,
            )
            self.assertIn("language_style", reply.text)
            event = controller.list_preference_events(session)[0]
            self.assertEqual(event.feedback_kind, "preferred_reply")
            self.assertEqual(event.preferred_text, "Chrome is open.")

            with self.assertRaises(LookupError):
                controller.handle_text("feedback shorter", session)

    def test_confirmation_and_safety_copy_are_not_trainable_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            controller.handle_text("preference learning on", session)
            controller.handle_text("open chrome", session)
            pending = controller.handle_text("send hello to Sam Lee", session)
            self.assertIsNotNone(pending.confirmation_id)

            with self.assertRaises(LookupError):
                controller.handle_text("feedback shorter", session)
            controller.reject(pending.confirmation_id, session)
            self.assertEqual(controller.list_preference_events(session), ())

    def test_guest_cannot_read_or_configure_owner_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            guest = controller.create_guest_session()

            response = controller.handle_text("preference learning on", guest)
            self.assertIn("owner", response.text.casefold())
            self.assertFalse(controller.preference_learning_enabled)
            with self.assertRaises(PermissionError):
                controller.list_preference_events(guest)

    def test_secret_like_content_is_rejected_by_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            controller.configure_preference_learning(session, enabled=True)
            store = PreferenceStore(controller.database)

            with self.assertRaises(ValueError):
                store.record(
                    controller.owner,
                    owner_id=controller.owner.user_id,
                    session_id=session.session_id,
                    user_text="remember this",
                    assistant_text="Your password is hunter2",
                    feedback_kind="accept_response",
                )
            self.assertEqual(controller.list_preference_events(session), ())

    def test_preference_event_can_be_deleted_without_audit_text_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            controller.handle_text("preference learning on", session)
            controller.handle_text("open chrome", session)
            controller.handle_text("feedback more direct", session)
            event = controller.list_preference_events(session)[0]

            self.assertTrue(controller.delete_preference_event(session, event.event_id))
            self.assertEqual(controller.list_preference_events(session), ())
            audit_text = controller.config.runtime.audit_log.read_text(encoding="utf-8")
            self.assertNotIn(event.user_text, audit_text)
            self.assertNotIn(event.assistant_text, audit_text)

    def test_candidate_generated_while_disabled_cannot_be_labeled_after_enabling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            self.assertFalse(controller.preference_learning_enabled)

            controller.handle_text("open chrome", session)

            controller.handle_text("preference learning on", session)
            self.assertTrue(controller.preference_learning_enabled)

            with self.assertRaises(LookupError):
                controller.handle_text("feedback shorter", session)
            self.assertEqual(controller.list_preference_events(session), ())


if __name__ == "__main__":
    unittest.main()
