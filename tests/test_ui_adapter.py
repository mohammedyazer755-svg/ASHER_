from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from asher.agent.controller import CompanionController
from asher.brain.deterministic import ContactResolver
from asher.config import AsherConfig
from asher.security.strong_auth import StrongAuthResult
from asher.types import AuthMethod, RiskLevel
from asher.ui.companion_adapter import CompanionDesktopController
from asher.ui.controller import PendingAction


class _Denied:
    def verify(self, _prompt: str) -> StrongAuthResult:
        return StrongAuthResult(False, "test", "denied")


class _Verified:
    def verify(self, _prompt: str) -> StrongAuthResult:
        return StrongAuthResult(True, "test", "verified")


class _FakeVoiceRuntime:
    def __init__(self, _controller, *, tts, on_event) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.on_event = on_event

    def run_forever(self) -> None:
        self.started.set()
        self.stopped.wait(2)

    def stop(self) -> None:
        self.stopped.set()


class UIAdapterTests(unittest.TestCase):
    def _adapter(self, directory: str, *, verified: bool = False) -> CompanionDesktopController:
        controller = CompanionController(
            AsherConfig.load(directory),
            contact_resolver=ContactResolver(("Sam Lee",), {"sam": "Sam Lee"}),
        )
        return CompanionDesktopController(
            controller,
            strong_authenticator=_Verified() if verified else _Denied(),
        )

    def test_ui_uses_real_controller_and_persists_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory, verified=True)
            response = adapter.submit_text("open chrome")
            self.assertEqual(response.sender, "Asher")
            record = adapter.create_memory("demo goal", "VoiceGuard", "goal", "private")
            self.assertEqual(adapter.list_memories()[0].memory_id, record.memory_id)
            adapter.update_memory(record.memory_id, "VoiceGuard verified", "goal", "private")
            self.assertEqual(adapter.list_memories()[0].value, "VoiceGuard verified")
            self.assertTrue(adapter.delete_memory(record.memory_id))
            self.assertTrue(any(item.event == "ui_memory_delete" for item in adapter.audit_records()))

    def test_ui_confirmation_is_session_bound_and_sensitive_denies_without_device_auth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            adapter.stage_confirmation(
                PendingAction(
                    "manual",
                    "files.delete",
                    "fixture.txt",
                    "Move fixture to recycle bin",
                    {"path": "fixture.txt"},
                    RiskLevel.SENSITIVE,
                )
            )
            self.assertFalse(adapter.approve_pending())
            self.assertIsNotNone(adapter.pending_action())
            self.assertTrue(adapter.reject_pending())
            self.assertIsNone(adapter.pending_action())

    def test_ui_whatsapp_preview_can_be_approved_only_by_local_ui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            reply = adapter.submit_text("send hello to Sam Lee")
            pending = adapter.pending_action()
            self.assertIsNotNone(pending)
            self.assertEqual(pending.risk, RiskLevel.EXTERNAL_COMMUNICATION)
            self.assertTrue(adapter.approve_pending())
            self.assertIsNone(adapter.pending_action())

    def test_emergency_stop_latches_real_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            adapter.stage_confirmation(
                PendingAction("stop", "x", "fixture", "effect", {}, RiskLevel.EXTERNAL_COMMUNICATION)
            )
            self.assertTrue(adapter.emergency_stop().emergency_stopped)
            self.assertIsNone(adapter.pending_action())
            self.assertFalse(adapter.reset_emergency_stop().emergency_stopped)

    def test_offline_setting_disables_remote_planner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            adapter.companion.planner.online_enabled = True
            adapter.update_settings(offline_only=True, api_enabled=False)
            self.assertFalse(adapter.companion.planner.online_enabled)

    def test_listening_toggle_owns_a_background_voice_runtime_without_emergency_latch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            holder: list[_FakeVoiceRuntime] = []

            def factory(controller, *, tts, on_event):
                runtime = _FakeVoiceRuntime(controller, tts=tts, on_event=on_event)
                holder.append(runtime)
                return runtime

            controller = CompanionController(AsherConfig.load(directory))
            adapter = CompanionDesktopController(
                controller,
                strong_authenticator=_Verified(),
                voice_runtime_factory=factory,
            )
            self.assertFalse(adapter.status().emergency_stopped)
            self.assertEqual(adapter.toggle_listening().state.value, "listening")
            deadline = time.monotonic() + 1.0
            while not holder and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(holder and holder[0].started.wait(1.0))
            adapter.toggle_listening()
            self.assertTrue(holder[0].stopped.is_set())
            self.assertFalse(adapter.status().emergency_stopped)


if __name__ == "__main__":
    unittest.main()
