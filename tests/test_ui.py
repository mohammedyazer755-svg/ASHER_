"""Controller and optional offscreen smoke tests for the ASHER UI."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from asher.types import RiskLevel
from asher.ui import is_available
from asher.ui.controller import ControllerUnavailable, DesktopController, PendingAction
from asher.voice.tts import TTSManager, VoiceProfile, VoiceProfileRegistry


def _controller() -> DesktopController:
    registry = VoiceProfileRegistry(
        (
            VoiceProfile("offline_male", "Male", "fake", gender_hint="male"),
            VoiceProfile("offline_female", "Female", "fake", gender_hint="female"),
        )
    )
    tts = TTSManager(registry, selected_profile="offline_male")
    tts.register_provider("fake", type("Provider", (), {"speak": lambda *_args: None, "stop": lambda *_args: None})())
    return DesktopController(tts=tts)


class ControllerTests(unittest.TestCase):
    def test_text_and_memory_crud_are_observable(self) -> None:
        controller = _controller()
        response = controller.submit_text("hello Asher")
        self.assertEqual(response.sender, "Asher")
        created = controller.create_memory("Project", "VoiceGuard", "goal", "private")
        self.assertEqual(controller.list_memories()[0].value, "VoiceGuard")
        updated = controller.update_memory(created.memory_id, "VoiceGuard v2", "goal", "private")
        self.assertEqual(updated.value, "VoiceGuard v2")
        self.assertTrue(controller.delete_memory(created.memory_id))
        self.assertTrue(any(item.event == "memory_delete" for item in controller.audit_records()))

    def test_confirmation_preview_requires_local_approval(self) -> None:
        controller = _controller()
        pending = PendingAction(
            "confirmation-1",
            "send_message",
            "Test contact",
            "Send the prepared message",
            {"message": "hello"},
            RiskLevel.EXTERNAL_COMMUNICATION,
        )
        controller.stage_confirmation(pending)
        self.assertEqual(controller.pending_action(), pending)
        self.assertTrue(controller.approve_pending())
        self.assertIsNone(controller.pending_action())

    def test_guest_sensitive_permission_cannot_be_enabled(self) -> None:
        controller = _controller()
        with self.assertRaises(ValueError):
            controller.set_permission("guest", "private_memory", True)

    def test_emergency_stop_cancels_pending_and_can_reset(self) -> None:
        controller = _controller()
        controller.stage_confirmation(
            PendingAction("stop-me", "delete_file", "fixture", "Delete fixture", {}, RiskLevel.SENSITIVE)
        )
        status = controller.emergency_stop()
        self.assertTrue(status.emergency_stopped)
        self.assertIsNone(controller.pending_action())
        self.assertEqual(controller.reset_emergency_stop().emergency_stopped, False)

    def test_sensitive_confirmation_requires_strong_auth_adapter(self) -> None:
        controller = _controller()
        controller.stage_confirmation(
            PendingAction("secure", "delete_file", "fixture", "Delete fixture", {}, RiskLevel.SENSITIVE)
        )
        self.assertFalse(controller.approve_pending())
        self.assertIsNotNone(controller.pending_action())

    def test_voiceguard_absence_is_explicit(self) -> None:
        controller = _controller()
        user = controller.enroll_user("Guest test", "guest")
        with self.assertRaises(ControllerUnavailable):
            controller.capture_voice_sample(user.user_id)

    def test_credential_like_memory_is_rejected(self) -> None:
        controller = _controller()
        with self.assertRaises(ValueError):
            controller.create_memory("password", "not stored", "semantic", "sensitive")

    def test_sensitive_memory_requires_explicit_consent(self) -> None:
        controller = _controller()
        with self.assertRaises(PermissionError):
            controller.create_memory("health note", "private detail", "semantic", "sensitive")
        record = controller.create_memory(
            "health note", "private detail", "semantic", "sensitive", consented=True
        )
        self.assertEqual(record.sensitivity, "sensitive")


@unittest.skipUnless(is_available(), "PySide6 is optional; install it for the offscreen smoke test")
class QtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(["asher-ui-test"])

    def test_window_renders_all_required_views(self) -> None:
        from asher.ui.window import AsherMainWindow

        window = AsherMainWindow(_controller())
        window.show()
        self.app.processEvents()
        self.assertTrue(window.isVisible())
        self.assertGreaterEqual(window.stack.count(), 9)
        self.assertFalse(window.grab().isNull())
        window.select_page("Memory")
        self.app.processEvents()
        self.assertEqual(window.stack.currentWidget(), window.memory_page)
        window.close()


    def test_local_provider_is_not_presented_as_no_connectivity(self) -> None:
        from dataclasses import replace

        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        window.show()
        self.app.processEvents()

        local_status = replace(controller.status(), offline=True, api_configured=False)
        with patch.object(controller, "status", return_value=local_status):
            window._refresh_status()
        self.app.processEvents()

        self.assertEqual(window.header_offline.text(), "LOCAL MODE")
        self.assertIn("local", window.home.provider.text().casefold())
        self.assertNotIn("offline", window.home.provider.text().casefold())
        self.assertEqual(window.companion_mode.presence.text(), "LOCAL")
        window.close()

    def test_memory_export_uses_workspace_save_dialog_and_stays_out_of_companion(self) -> None:
        from PySide6.QtWidgets import QPushButton

        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        window.show()
        window.select_page("Memory")
        self.app.processEvents()

        self.assertTrue(window.memory_page.export_button.isVisible())
        self.assertNotIn(
            "Export JSON",
            [button.text() for button in window.companion_mode.findChildren(QPushButton)],
        )
        with tempfile.TemporaryDirectory() as directory:
            selected = os.path.join(directory, "selected-memory-export")
            with (
                patch(
                    "asher.ui.window.QFileDialog.getSaveFileName",
                    return_value=(selected, "JSON files (*.json)"),
                ),
                patch.object(window, "_run") as run,
            ):
                window.memory_page.export_button.click()
                self.app.processEvents()

            run.assert_called_once()
            args, kwargs = run.call_args
            self.assertEqual(args[0], controller.export_memories)
            self.assertEqual(args[1], selected + ".json")
            self.assertEqual(kwargs["on_result"], window._memory_export_result)

        window.close()

    def test_reauthentication_control_and_expired_status_are_workspace_only(self) -> None:
        from dataclasses import replace

        from PySide6.QtWidgets import QPushButton

        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        window.show()
        self.app.processEvents()

        expired = replace(controller.status(), owner_session_active=False)
        with patch.object(controller, "status", return_value=expired):
            window._refresh_status()
        self.assertEqual(window.header_session.text(), "SESSION EXPIRED")
        self.assertTrue(window.reauthenticate_button.isVisible())
        self.assertNotIn(
            "Re-authenticate",
            [button.text() for button in window.companion_mode.findChildren(QPushButton)],
        )

        with patch.object(window, "_run") as run:
            window.reauthenticate_button.click()
            self.app.processEvents()
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], controller.reauthenticate_owner)
        self.assertEqual(kwargs["on_result"], window._reauthentication_result)
        window.close()

    def test_real_microphone_level_reaches_both_orbs_and_resets_without_tts_fabrication(self) -> None:
        from dataclasses import replace

        from asher.core.state import AssistantState
        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        window.show()
        self.app.processEvents()

        listening = replace(
            controller.status(),
            state=AssistantState.LISTENING,
            microphone_active=True,
            microphone_level=0.37,
        )
        with patch.object(controller, "status", return_value=listening):
            window._refresh_status()
            self.app.processEvents()
            self.assertAlmostEqual(window.home.orb._audio_level, 0.37)
            # The visible Companion orb may consume one presentation-decay tick
            # while processEvents() paints it; the real sample must remain present
            # and bounded, not equal an animation-frame-dependent exact value.
            self.assertGreater(window.companion_mode.orb._audio_level, 0.0)
            self.assertLessEqual(window.companion_mode.orb._audio_level, 0.37)
            self.assertFalse(window.home.orb.isVisible())
            self.assertFalse(window.home.orb._timer.isActive())

        # A speaking state never manufactures amplitude: without active real
        # microphone input, even a stale/non-zero scalar is forced to zero.
        speaking = replace(
            listening,
            state=AssistantState.SPEAKING,
            microphone_active=False,
            microphone_level=0.91,
        )
        with patch.object(controller, "status", return_value=speaking):
            window._refresh_status()
        self.assertEqual(window.home.orb._audio_level, 0.0)
        self.assertEqual(window.companion_mode.orb._audio_level, 0.0)
        window.close()

    def test_presentation_audio_timer_directly_refreshes_both_orbs(self) -> None:
        from dataclasses import replace

        from asher.core.state import AssistantState
        from asher.ui.window import AsherMainWindow

        controller = _controller()
        window = AsherMainWindow(controller)
        try:
            self.assertEqual(window._audio_timer.interval(), 66)
            self.assertFalse(window._audio_timer.isActive())
            listening = replace(
                controller.status(),
                state=AssistantState.LISTENING,
                microphone_active=True,
                microphone_level=0.42,
            )
            with patch.object(controller, "status", return_value=listening):
                window._refresh_status()
                self.assertTrue(window._audio_timer.isActive())
                window._refresh_orb_audio()

            self.assertAlmostEqual(window.home.orb._audio_level, 0.42)
            self.assertAlmostEqual(window.companion_mode.orb._audio_level, 0.42)

            inactive = replace(
                listening,
                microphone_active=False,
                microphone_level=0.42,
            )
            with patch.object(controller, "status", return_value=inactive):
                window._refresh_orb_audio()
            self.assertFalse(window._audio_timer.isActive())
            self.assertEqual(window.home.orb._audio_level, 0.0)
            self.assertEqual(window.companion_mode.orb._audio_level, 0.0)

            window._audio_poll_enabled = True
            window._sync_audio_timer()
            with patch.object(controller, "status", side_effect=RuntimeError("offline")):
                window._refresh_orb_audio()
            self.assertFalse(window._audio_poll_enabled)
            self.assertFalse(window._audio_timer.isActive())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
