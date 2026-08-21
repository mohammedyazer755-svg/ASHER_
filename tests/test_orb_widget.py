from __future__ import annotations

import os
import unittest

from asher.core.state import AssistantState
from asher.ui import is_available
from asher.ui.orb_widget import visual_for_state


class OrbVisualContractTests(unittest.TestCase):
    def test_every_canonical_and_legacy_state_has_a_visual_profile(self) -> None:
        for state in AssistantState:
            visual = visual_for_state(state)
            self.assertTrue(visual.primary.startswith("#"), state)
            self.assertTrue(visual.secondary.startswith("#"), state)
            self.assertGreaterEqual(visual.particles, 0, state)
            self.assertGreater(visual.speed, 0.0, state)

    def test_stopped_and_locked_do_not_emit_particles(self) -> None:
        self.assertEqual(visual_for_state(AssistantState.STOPPED).particles, 0)
        self.assertEqual(visual_for_state(AssistantState.LOCKED).particles, 0)


@unittest.skipUnless(is_available(), "PySide6 is optional; install it for the offscreen orb smoke test")
class OrbQtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-orb-test"])

    def test_orb_renders_real_states_without_changing_them(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        orb.resize(420, 420)
        orb.show()
        for state in (
            AssistantState.STANDBY,
            AssistantState.THINKING,
            AssistantState.AWAITING_CONFIRMATION,
            AssistantState.EXECUTING,
            AssistantState.SPEAKING,
            AssistantState.SUCCESS,
            AssistantState.ERROR,
            AssistantState.STOPPED,
        ):
            orb.set_state(state)
            self.app.processEvents()
            self.assertEqual(orb.state, state)
            self.assertFalse(orb.grab().isNull())
        orb.close()


if __name__ == "__main__":
    unittest.main()
