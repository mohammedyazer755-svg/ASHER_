from __future__ import annotations

import tempfile
import unittest

from asher.agent.controller import CompanionController
from asher.brain.deterministic import ContactResolver
from asher.config import AsherConfig
from asher.core.state import AssistantState, StateStore
from asher.types import AuthMethod


class CinematicStateBridgeTests(unittest.TestCase):
    def test_state_event_publishes_previous_and_new_state(self) -> None:
        states = StateStore()
        events = []
        unsubscribe = states.subscribe(events.append)
        try:
            event = states.transition(AssistantState.THINKING, "Planning")
        finally:
            unsubscribe()

        self.assertEqual(event.previous_state, AssistantState.STANDBY)
        self.assertEqual(event.state, AssistantState.THINKING)
        self.assertEqual(events, [event])

    def test_complete_cinematic_state_vocabulary_is_available(self) -> None:
        expected = {
            "standby",
            "wake_detected",
            "authenticating",
            "authenticated",
            "listening",
            "transcribing",
            "thinking",
            "awaiting_confirmation",
            "executing",
            "observing",
            "speaking",
            "success",
            "error",
            "offline",
            "stopped",
            "locked",
        }
        values = {state.value for state in AssistantState}
        self.assertTrue(expected.issubset(values))

    def test_real_controller_orders_think_execute_observe_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CompanionController(
                AsherConfig.load(directory),
                contact_resolver=ContactResolver(
                    ("Avery Stone", "Sam Lee"),
                    {"avery": "Avery Stone", "sam": "Sam Lee"},
                ),
            )
            events = []
            unsubscribe = controller.loop.states.subscribe(events.append)
            try:
                session = controller.create_owner_session(AuthMethod.LOCAL_UI)
                reply = controller.handle_text("open chrome", session)
            finally:
                unsubscribe()

            self.assertIn("complete", {item.status for item in reply.updates})
            observed = [event.state for event in events]
            expected_order = [
                AssistantState.THINKING,
                AssistantState.EXECUTING,
                AssistantState.OBSERVING,
                AssistantState.SUCCESS,
            ]
            cursor = 0
            for state in observed:
                if cursor < len(expected_order) and state is expected_order[cursor]:
                    cursor += 1
            self.assertEqual(
                cursor,
                len(expected_order),
                f"Expected state subsequence {expected_order!r}; observed {observed!r}",
            )


if __name__ == "__main__":
    unittest.main()
