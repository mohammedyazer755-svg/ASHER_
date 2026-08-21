"""Controller and optional offscreen smoke tests for the ASHER UI."""

from __future__ import annotations

import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
