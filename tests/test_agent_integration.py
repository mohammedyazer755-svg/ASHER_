from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asher.agent.controller import CompanionController
from asher.brain.deterministic import ContactResolver, DeterministicPlanner
from asher.config import AsherConfig
from asher.types import AuthMethod


class AgentIntegrationTests(unittest.TestCase):
    def _controller(self, directory: str) -> CompanionController:
        return CompanionController(
            AsherConfig.load(directory),
            contact_resolver=ContactResolver(
                ("Avery Stone", "Sam Lee"),
                {"avery": "Avery Stone", "sam": "Sam Lee"},
            ),
        )

    def test_common_tools_are_dry_run_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            opened = controller.handle_text("open chrome", session)
            closed = controller.handle_text("close chrome", session)
            self.assertIn("complete", {item.status for item in opened.updates})
            self.assertIn("complete", {item.status for item in closed.updates})

    def test_search_variants_and_ambiguous_contact_are_safe(self) -> None:
        planner = DeterministicPlanner(
            ContactResolver(("Sara", "Sarah"), {"sara": "Sara", "sarah": "Sarah"})
        )
        self.assertEqual(planner.plan("search Sara").steps[0].call.arguments["contact"], "Sara")
        self.assertEqual(planner.plan("search S A R A").steps[0].call.arguments["contact"], "Sara")
        self.assertIn("Which contact", planner.plan("search Sarra").response)
        self.assertEqual(
            planner.plan("search cats").steps[0].call.tool_name,
            "browser.search",
        )

    def test_compound_and_indirect_message_requires_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            compound = controller.handle_text("open chrome and search cats", session)
            self.assertEqual(
                [item.result.tool_name for item in compound.updates if item.result is not None],
                ["app.open", "browser.search"],
            )
            reply = controller.handle_text(
                "ask Sam Lee whether he is ready to hang out",
                session,
            )
            self.assertIsNotNone(reply.confirmation_id)
            self.assertIn("exact local preview", reply.text)
            approved = controller.approve(reply.confirmation_id, session)
            self.assertIn("complete", {item.status for item in approved.updates})
            self.assertIsNone(controller.loop.active)

    def test_guest_cannot_use_private_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            guest = controller.create_guest_session()
            reply = controller.handle_text("open chrome", guest)
            self.assertTrue(any(item.status in {"failed", "denied"} for item in reply.updates))
            self.assertIsNone(controller.loop.active)

    def test_failed_confirmation_approval_does_not_orphan_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            reply = controller.handle_text("send hello to Sam Lee", session)
            self.assertIsNotNone(reply.confirmation_id)
            wrong = controller.create_owner_session(AuthMethod.LOCAL_UI)
            denied = controller.approve(reply.confirmation_id, wrong)
            self.assertTrue(any(item.status == "denied" for item in denied.updates))
            self.assertIsNotNone(controller.loop.active)
            self.assertEqual(controller.loop.active.waiting_confirmation_id, reply.confirmation_id)
            self.assertTrue(controller.reject(reply.confirmation_id, session).updates)

    def test_pronoun_continuation_context_cannot_cross_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            first = controller.create_owner_session(AuthMethod.LOCAL_UI)
            prepared = controller.handle_text("send hello to Sam Lee", first)
            self.assertIsNotNone(prepared.confirmation_id)
            controller.reject(prepared.confirmation_id, first)

            second = controller.create_owner_session(AuthMethod.LOCAL_UI)
            isolated = controller.handle_text("send hello to her", second)
            self.assertIsNone(isolated.confirmation_id)
            self.assertIn("recipient", isolated.text.casefold())

            continued = controller.handle_text("send hello to her", first)
            self.assertIsNotNone(continued.confirmation_id)

    def test_disabled_long_term_memory_blocks_tools_without_weakening_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            controller.configure_memory(enabled=False, retention_days=90)
            reply = controller.handle_text("what is my project goal", session)
            self.assertIn("memory is disabled", reply.text.casefold())
            self.assertIsNone(controller.loop.active)
            self.assertFalse(controller.memory_enabled)
            self.assertEqual(controller.memory_retention_days, 90)

    def test_emotional_cue_is_practical_uncertain_and_non_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            controller.planner.openai = None
            controller.planner.ollama = None
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            reply = controller.handle_text("I am frustrated and stuck", session)
            lowered = reply.text.casefold()
            self.assertIn("frustrating", lowered)
            self.assertIn("practical step", lowered)
            for prohibited in ("diagnosis", "disorder", "definitely feel", "depend on me"):
                self.assertNotIn(prohibited, lowered)

    def test_search_follow_up_resolves_prior_topic_within_session_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session1 = controller.create_owner_session(AuthMethod.LOCAL_UI)
            reply1 = controller.handle_text("search YOLO pose estimation", session1)
            search1 = [
                u.result.evidence[0].data
                for u in reply1.updates
                if u.result and u.result.tool_name == "browser.search"
            ]
            self.assertEqual(len(search1), 1)
            self.assertEqual(search1[0]["query"], "YOLO pose estimation")
            self.assertEqual(search1[0]["engine"], "google")

            reply2 = controller.handle_text("Search that on YouTube", session1)
            search2 = [
                u.result.evidence[0].data
                for u in reply2.updates
                if u.result and u.result.tool_name == "browser.search"
            ]
            self.assertEqual(len(search2), 1)
            self.assertEqual(search2[0]["query"], "YOLO pose estimation")
            self.assertEqual(search2[0]["engine"], "youtube")

            session2 = controller.create_owner_session(AuthMethod.LOCAL_UI)
            reply_isolated = controller.handle_text("Search that on YouTube", session2)
            self.assertIn("Which search topic", reply_isolated.text)
            self.assertEqual(
                [u for u in reply_isolated.updates if u.result and u.result.tool_name == "browser.search"],
                [],
            )

    def test_no_tool_replies_appended_to_working_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            session = controller.create_owner_session(AuthMethod.LOCAL_UI)
            reply = controller.handle_text("hello", session)
            turns = controller.working_memory.recent(session.session_id)
            self.assertEqual(len(turns), 2)
            self.assertEqual(turns[0].role, "user")
            self.assertEqual(turns[0].text, "hello")
            self.assertEqual(turns[1].role, "assistant")
            self.assertEqual(turns[1].text, reply.text)


if __name__ == "__main__":
    unittest.main()
