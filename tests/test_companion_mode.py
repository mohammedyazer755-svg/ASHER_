from __future__ import annotations

import os
import unittest

from asher.core.state import AssistantState
from asher.ui import is_available
from asher.ui.window import should_use_companion_mode


class CompanionModeContractTests(unittest.TestCase):
    def test_companion_mode_requires_active_voice_session(self) -> None:
        self.assertFalse(should_use_companion_mode(AssistantState.STANDBY, True))
        self.assertFalse(should_use_companion_mode(AssistantState.THINKING, False))
        self.assertTrue(should_use_companion_mode(AssistantState.WAKE_DETECTED, True))
        self.assertTrue(should_use_companion_mode(AssistantState.LISTENING, True))
        self.assertTrue(should_use_companion_mode(AssistantState.THINKING, True))
        self.assertTrue(should_use_companion_mode(AssistantState.SPEAKING, True))


@unittest.skipUnless(is_available(), "PySide6 is optional; install it for companion-mode smoke tests")
class CompanionModeQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-companion-mode-test"])

    def test_window_keeps_workspace_and_immersive_scene_separate(self) -> None:
        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow

        window = AsherMainWindow(_controller())
        window.show()
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        self.assertEqual(window.mode_stack.count(), 2)
        window._set_companion_mode(True)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        window._set_companion_mode(False)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        window.close()

    def test_companion_orb_has_bounded_external_scale_hook(self) -> None:
        from asher.ui.orb_widget import AsherOrbWidget

        orb = AsherOrbWidget()
        orb.set_interactive_resize(True, initial_size=560)
        before = orb.state
        self.assertEqual(orb.set_display_size(1200), 900)
        self.assertEqual(orb.width(), 900)
        self.assertEqual(orb.set_display_size(100), 340)
        self.assertEqual(orb.width(), 340)
        self.assertIs(orb.state, before)
        orb.close()


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(is_available(), "PySide6 is optional; install it for companion-mode smoke tests")
class CompanionMinimalUiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-companion-minimal-test"])

    def test_companion_page_is_minimal_and_keeps_real_orb_and_stop(self) -> None:
        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow
        from PySide6.QtWidgets import QFrame

        window = AsherMainWindow(_controller())
        page = window.companion_mode
        self.assertTrue(page.orb._cinematic_mode)
        self.assertEqual(page.brand.text(), "ASHER")
        self.assertEqual(page.findChildren(QFrame, "companionHud"), [])
        self.assertTrue(page.presence.text())
        window.close()

    def test_fullscreen_is_scoped_to_companion_mode(self) -> None:
        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow

        window = AsherMainWindow(_controller())
        window.show()
        self.app.processEvents()
        window._set_companion_mode(True)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        self.assertTrue(window.isFullScreen())
        window._set_companion_mode(False)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        self.assertFalse(window.isFullScreen())
        window.close()

