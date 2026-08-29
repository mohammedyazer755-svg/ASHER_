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
        self.assertTrue(should_use_companion_mode(AssistantState.LOCKED, True))
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

    def test_locked_active_voice_status_stays_in_companion_until_standby(self) -> None:
        from dataclasses import replace
        from unittest.mock import patch

        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        window.show()
        self.app.processEvents()
        locked = replace(
            controller.status(),
            state=AssistantState.LOCKED,
            microphone_active=True,
        )
        with patch.object(controller, "status", return_value=locked):
            window._refresh_status()
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        self.assertTrue(window.isFullScreen())

        standby = replace(locked, state=AssistantState.STANDBY)
        with patch.object(controller, "status", return_value=standby):
            window._refresh_status()
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        self.assertFalse(window.isFullScreen())
        window.close()


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
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence

        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow

        window = AsherMainWindow(_controller())
        window.resize(1200, 740)
        window.show()
        self.app.processEvents()
        self.assertEqual(window._fullscreen_shortcut.key(), QKeySequence("F11"))
        self.assertEqual(window._windowed_shortcut.key(), QKeySequence("Esc"))
        self.assertEqual(
            window._fullscreen_shortcut.context(), Qt.ShortcutContext.WindowShortcut
        )
        self.assertEqual(
            window._windowed_shortcut.context(), Qt.ShortcutContext.WindowShortcut
        )
        window._set_companion_mode(True)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        self.assertTrue(window.isFullScreen())

        window._fullscreen_shortcut.activated.emit()
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        self.assertFalse(window.isFullScreen())
        window.resize(1240, 760)
        self.app.processEvents()
        self.assertEqual((window.width(), window.height()), (1240, 760))

        window._fullscreen_shortcut.activated.emit()
        self.app.processEvents()
        self.assertTrue(window.isFullScreen())
        window._windowed_shortcut.activated.emit()
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        self.assertFalse(window.isFullScreen())

        window._set_companion_mode(False)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        self.assertFalse(window.isFullScreen())
        window.close()

    def test_companion_window_toggle_restores_maximized_workspace(self) -> None:
        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow

        window = AsherMainWindow(_controller())
        window.showMaximized()
        self.app.processEvents()
        self.assertTrue(window.isMaximized())

        window._set_companion_mode(True)
        self.app.processEvents()
        self.assertTrue(window.isFullScreen())
        window._windowed_shortcut.activated.emit()
        self.app.processEvents()
        self.assertFalse(window.isFullScreen())
        self.assertTrue(window.isMaximized())

        window._fullscreen_shortcut.activated.emit()
        self.app.processEvents()
        self.assertTrue(window.isFullScreen())
        window._set_companion_mode(False)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        self.assertFalse(window.isFullScreen())
        self.assertTrue(window.isMaximized())
        window.close()

    def test_companion_window_toggle_restores_fullscreen_workspace(self) -> None:
        from tests.test_ui import _controller
        from asher.ui.window import AsherMainWindow

        window = AsherMainWindow(_controller())
        window.showFullScreen()
        self.app.processEvents()
        self.assertTrue(window.isFullScreen())

        window._set_companion_mode(True)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.companion_mode)
        window._windowed_shortcut.activated.emit()
        self.app.processEvents()
        self.assertFalse(window.isFullScreen())

        window._set_companion_mode(False)
        self.app.processEvents()
        self.assertIs(window.mode_stack.currentWidget(), window.workspace)
        self.assertTrue(window.isFullScreen())
        window.showNormal()
        window.close()

    def test_fullscreen_confirmation_shows_complete_plain_text_preview(self) -> None:
        import json
        from datetime import UTC, datetime, timedelta

        from tests.test_ui import _controller
        from asher.types import RiskLevel
        from asher.ui.controller import PendingAction
        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        page = window.companion_mode
        preview = {
            "contact": "Demo contact",
            "message": "First line\nSecond line <b>must stay plain text</b>",
            "body": "Exact consequential content",
        }
        pending = PendingAction(
            confirmation_id="confirmation-preview",
            action="whatsapp.send",
            target="Demo contact",
            effect="Send a WhatsApp message",
            preview=preview,
            risk=RiskLevel.EXTERNAL_COMMUNICATION,
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )

        window.show()
        controller.stage_confirmation(pending)
        # The ordinary status path must discover a runtime-created pending
        # action; production voice events do not call the page directly.
        window._refresh_status()
        window._set_companion_mode(True)
        self.app.processEvents()

        self.assertIs(window.mode_stack.currentWidget(), page)
        self.assertTrue(window.isFullScreen())
        self.assertTrue(page.confirm.isVisible())
        self.assertTrue(page.confirm_preview.isReadOnly())
        self.assertEqual(json.loads(page.confirm_preview.toPlainText()), preview)
        self.assertIn("<b>must stay plain text</b>", page.confirm_preview.toPlainText())
        self.assertEqual(page.cancel_button.text(), "Cancel")

        self.assertTrue(controller.reject_pending())
        window._refresh_status()
        self.assertTrue(page.confirm.isHidden())
        self.assertEqual(page.confirm_preview.toPlainText(), "")
        window.close()

    def test_companion_orb_and_confirmation_fit_all_target_viewports(self) -> None:
        from datetime import UTC, datetime, timedelta

        from tests.test_ui import _controller
        from asher.types import RiskLevel
        from asher.ui.controller import PendingAction
        from asher.ui.window import AsherMainWindow

        from PySide6.QtWidgets import QSizePolicy

        window = AsherMainWindow(_controller())
        page = window.companion_mode
        pending = PendingAction(
            confirmation_id="responsive-confirmation",
            action="files.write_text",
            target="fixture.txt",
            effect="Replace the selected file",
            preview={"path": "fixture.txt", "content": "exact content"},
            risk=RiskLevel.SENSITIVE,
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )

        for active_pending in (None, pending):
            page.refresh_pending(active_pending)
            for width, height in ((1366, 768), (1920, 1080), (2560, 1440)):
                with self.subTest(
                    width=width,
                    height=height,
                    confirmation=active_pending is not None,
                ):
                    page.resize(width, height)
                    self.app.processEvents()
                    applied = page._fit_orb_to_viewport()
                    top_height = max(
                        page.brand.sizeHint().height(),
                        page.stop_button.sizeHint().height(),
                    )
                    confirmation_height = (
                        page.confirm.sizeHint().height()
                        if active_pending is not None
                        else 0
                    )
                    usable_short_edge = min(
                        width,
                        max(1, height - top_height - confirmation_height - 24),
                    )
                    self.assertEqual(applied, usable_short_edge)
                    self.assertEqual(page.orb.minimumWidth(), 1)
                    self.assertEqual(page.orb.minimumHeight(), 1)
                    self.assertGreater(page.orb.maximumWidth(), 900)
                    self.assertGreater(page.orb.maximumHeight(), 900)
                    self.assertEqual(
                        page.orb.sizePolicy().horizontalPolicy(),
                        QSizePolicy.Policy.Expanding,
                    )
                    self.assertEqual(
                        page.orb.sizePolicy().verticalPolicy(),
                        QSizePolicy.Policy.Expanding,
                    )
        window.close()

