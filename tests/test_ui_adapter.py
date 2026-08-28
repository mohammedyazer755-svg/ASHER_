from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from asher.agent.controller import CompanionController
from asher.brain.deterministic import ContactResolver
from asher.config import AsherConfig
from asher.core.state import AssistantState
from asher.security.strong_auth import StrongAuthResult
from asher.types import AuthMethod, RiskLevel, Role
from asher.ui.companion_adapter import CompanionDesktopController
from asher.ui.controller import PendingAction
from asher.voice.runtime import VoiceRuntimeEvent
from asher.voice.types import TranscriptResult


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
        self.microphone_level = 0.42

    def run_forever(self) -> None:
        self.started.set()
        self.stopped.wait(2)

    def stop(self) -> None:
        self.microphone_level = 0.0
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

    def test_voice_history_persists_only_canonical_final_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory, verified=True)
            rejected = TranscriptResult(
                raw_text="hey usher maybe",
                normalized_text="hey usher maybe",
                acoustic_confidence=0.41,
                no_speech_probability=0.01,
            )
            adapter._on_voice_event(
                VoiceRuntimeEvent(
                    "wake_rejected",
                    "Wake phrase not detected",
                    rejected,
                )
            )
            self.assertEqual(adapter.conversation(), ())

            accepted = TranscriptResult(
                raw_text="hey asher open chrome",
                normalized_text="hey asher open chrome",
                acoustic_confidence=0.95,
                no_speech_probability=0.01,
            )
            adapter._on_voice_event(
                VoiceRuntimeEvent(
                    "transcript",
                    "open chrome",
                    accepted,
                )
            )

            turns = adapter.conversation()
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0].sender, "You")
            self.assertEqual(turns[0].message, "open chrome")
            adapter.close()

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

    def test_owner_memory_export_is_atomic_authenticated_and_privacy_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory, verified=True)
            private_value = "export verification value"
            adapter.create_memory("export fixture", private_value, "goal", "private")
            destination = Path(directory) / "selected" / "memories.json"

            exported = adapter.export_memories(destination)

            self.assertEqual(exported, destination)
            self.assertFalse(destination.with_suffix(".json.tmp").exists())
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["memories"]), 1)
            self.assertEqual(payload["memories"][0]["value"], private_value)
            export_events = [
                item for item in adapter.audit_records() if item.event == "ui_memory_export"
            ]
            self.assertEqual(export_events[-1].result, "complete")
            self.assertEqual(
                dict(export_events[-1].details),
                {"format": "json", "record_count": 1},
            )
            raw_audit = adapter.companion.config.runtime.audit_log.read_text(encoding="utf-8")
            self.assertNotIn(str(destination), raw_audit)
            self.assertNotIn(destination.name, raw_audit)
            self.assertNotIn(private_value, raw_audit)

    def test_memory_export_denies_without_device_auth_or_owner_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            adapter.create_memory("export fixture", "private value", "goal", "private")
            denied_destination = Path(directory) / "denied.json"

            with self.assertRaises(PermissionError):
                adapter.export_memories(denied_destination)
            self.assertFalse(denied_destination.exists())
            denied_event = next(
                item
                for item in adapter.audit_records()
                if item.event == "ui_memory_export"
            )
            self.assertEqual(denied_event.result, "denied")
            self.assertEqual(denied_event.details.get("reason"), "device_auth_denied")
            self.assertNotIn("path", denied_event.details)

            trusted = adapter.companion.users.create("Trusted export test", Role.TRUSTED)
            adapter._owner_session = adapter.companion.sessions.create(
                trusted,
                AuthMethod.LOCAL_UI,
            )
            owner_only_destination = Path(directory) / "owner-only.json"
            with self.assertRaises(PermissionError):
                adapter.export_memories(owner_only_destination)
            self.assertFalse(owner_only_destination.exists())
            owner_event = next(
                item
                for item in adapter.audit_records()
                if item.event == "ui_memory_export"
            )
            self.assertEqual(owner_event.details.get("reason"), "owner_required")

    def test_expired_owner_session_requires_device_reauthentication_without_rebinding_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory, verified=True)
            old_session = adapter._owner_session
            adapter.submit_text("send hello to Sam Lee")
            pending = adapter.pending_action()
            self.assertIsNotNone(pending)
            confirmation = adapter.companion.registry.confirmations.get(
                pending.confirmation_id
            )
            self.assertIsNotNone(confirmation)
            self.assertEqual(confirmation.session_id, old_session.session_id)

            adapter.companion.sessions.invalidate(old_session.session_id)
            self.assertFalse(adapter.status().owner_session_active)
            with self.assertRaises(PermissionError):
                adapter.list_memories()

            status = adapter.reauthenticate_owner()

            fresh_session = adapter._owner_session
            self.assertNotEqual(fresh_session.session_id, old_session.session_id)
            self.assertEqual(fresh_session.actor.user_id, adapter.companion.owner.user_id)
            self.assertIs(fresh_session.actor.role, Role.OWNER)
            self.assertIs(fresh_session.auth_method, AuthMethod.DEVICE_CREDENTIAL)
            self.assertTrue(status.owner_session_active)
            self.assertFalse(adapter.approve_pending())
            self.assertIsNone(adapter.pending_action())
            self.assertIsNone(
                adapter.companion.registry.confirmations.get(pending.confirmation_id)
            )
            self.assertIsNone(adapter.companion.loop.active)
            event = next(
                item
                for item in adapter.audit_records()
                if item.event == "ui_owner_reauthentication"
            )
            self.assertEqual(event.result, "complete")
            self.assertEqual(
                dict(event.details),
                {"method": AuthMethod.DEVICE_CREDENTIAL.value},
            )

    def test_reauthentication_click_without_device_proof_leaves_session_expired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            expired_session = adapter._owner_session
            adapter.companion.sessions.invalidate(expired_session.session_id)

            with self.assertRaises(PermissionError):
                adapter.reauthenticate_owner()

            self.assertEqual(
                adapter._owner_session.session_id,
                expired_session.session_id,
            )
            self.assertFalse(adapter.status().owner_session_active)
            self.assertEqual(adapter.status().state, AssistantState.LOCKED)
            event = next(
                item
                for item in adapter.audit_records()
                if item.event == "ui_owner_reauthentication"
            )
            self.assertEqual(event.result, "denied")
            self.assertEqual(event.details.get("reason"), "device_auth_denied")
            self.assertNotIn("message", event.details)
            self.assertNotIn("value", event.details)

    def test_reauthentication_does_not_replace_an_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory, verified=True)
            active_session_id = adapter._owner_session.session_id

            with self.assertRaises(RuntimeError):
                adapter.reauthenticate_owner()

            self.assertEqual(adapter._owner_session.session_id, active_session_id)
            self.assertTrue(adapter.status().owner_session_active)
            event = next(
                item
                for item in adapter.audit_records()
                if item.event == "ui_owner_reauthentication"
            )
            self.assertEqual(event.details.get("reason"), "session_still_active")

    def test_reject_pending_is_false_when_session_mismatch_leaves_action_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory, verified=True)
            original_session = adapter._owner_session
            adapter.submit_text("send hello to Sam Lee")
            pending = adapter.pending_action()
            self.assertIsNotNone(pending)

            # Simulate a different still-valid desktop session without using
            # the reauthentication flow, which now safely cancels stale work.
            adapter._owner_session = adapter.companion.sessions.create(
                adapter.companion.owner,
                AuthMethod.DEVICE_CREDENTIAL,
            )

            self.assertFalse(adapter.reject_pending())
            self.assertEqual(
                adapter.pending_action().confirmation_id,
                pending.confirmation_id,
            )
            self.assertEqual(
                adapter.companion.loop.active.session.session_id,
                original_session.session_id,
            )
            adapter.companion.loop.cancel("test cleanup")

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

    def test_microphone_environment_default_accepts_index_and_device_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"ASHER_MIC_INDEX": " 7 "}):
                self.assertEqual(AsherConfig.load(directory).microphone_index, 7)
            with patch.dict("os.environ", {"ASHER_MIC_INDEX": " Studio USB microphone "}):
                self.assertEqual(
                    AsherConfig.load(directory).microphone_index,
                    "Studio USB microphone",
                )

    def test_configured_microphone_is_lazy_and_ui_selection_overrides_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"ASHER_MIC_INDEX": "Studio USB microphone"}):
                controller = CompanionController(AsherConfig.load(directory))
            adapter = CompanionDesktopController(
                controller,
                strong_authenticator=_Verified(),
            )

            configured_runtime = adapter._build_voice_runtime()
            self.assertEqual(configured_runtime.backend.device, "Studio USB microphone")

            adapter.update_settings(microphone_index=4)
            selected_runtime = adapter._build_voice_runtime()
            self.assertEqual(selected_runtime.backend.device, 4)

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
            started_status = adapter.toggle_listening()
            self.assertEqual(started_status.state.value, "standby")
            self.assertTrue(started_status.microphone_active)
            # The microphone runtime being active must not mask a deeper
            # assistant state such as thinking/executing/speaking.
            controller.loop.states.transition(
                AssistantState.THINKING,
                "Thinking test",
            )
            self.assertEqual(adapter.status().state.value, "thinking")
            self.assertTrue(adapter.status().microphone_active)
            deadline = time.monotonic() + 1.0
            while not holder and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(holder and holder[0].started.wait(1.0))
            self.assertAlmostEqual(adapter.status().microphone_level, 0.42)
            holder[0].microphone_level = 2.5
            self.assertEqual(adapter.status().microphone_level, 1.0)
            stopped_status = adapter.toggle_listening()
            self.assertTrue(holder[0].stopped.is_set())
            self.assertFalse(stopped_status.microphone_active)
            self.assertEqual(stopped_status.microphone_level, 0.0)
            self.assertEqual(stopped_status.state.value, "standby")
            self.assertFalse(adapter.status().emergency_stopped)

    def test_emergency_stop_cannot_leave_a_concurrently_starting_runtime_orphaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            factory_entered = threading.Event()
            release_factory = threading.Event()
            stop_attempted = threading.Event()
            holder: list[_FakeVoiceRuntime] = []
            errors: list[BaseException] = []

            def factory(controller, *, tts, on_event):
                factory_entered.set()
                self.assertTrue(release_factory.wait(1.0))
                runtime = _FakeVoiceRuntime(controller, tts=tts, on_event=on_event)
                holder.append(runtime)
                return runtime

            controller = CompanionController(AsherConfig.load(directory))
            adapter = CompanionDesktopController(
                controller,
                strong_authenticator=_Verified(),
                voice_runtime_factory=factory,
            )

            def start_listening() -> None:
                try:
                    adapter.toggle_listening()
                except BaseException as error:  # captured for deterministic assertion
                    errors.append(error)

            def stop_everything() -> None:
                stop_attempted.set()
                try:
                    adapter.emergency_stop()
                except BaseException as error:  # captured for deterministic assertion
                    errors.append(error)

            starter = threading.Thread(target=start_listening)
            stopper = threading.Thread(target=stop_everything)
            starter.start()
            self.assertTrue(factory_entered.wait(1.0))
            stopper.start()
            self.assertTrue(stop_attempted.wait(1.0))
            release_factory.set()
            starter.join(2.0)
            stopper.join(2.0)

            self.assertFalse(starter.is_alive())
            self.assertFalse(stopper.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(holder and holder[0].started.is_set())
            self.assertTrue(holder[0].stopped.is_set())
            self.assertTrue(adapter.status().emergency_stopped)
            self.assertFalse(adapter.status().microphone_active)
            self.assertIsNone(adapter._voice_runtime)
            self.assertIsNone(adapter._voice_thread)

    def test_stale_runtime_error_cannot_replace_new_runtime_or_publish_error_state(self) -> None:
        class DelayedFailure:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def run_forever(self) -> None:
                self.started.set()
                self.release.wait(1.0)
                raise RuntimeError("stale fixture failure")

        with tempfile.TemporaryDirectory() as directory:
            controller = CompanionController(AsherConfig.load(directory))
            adapter = CompanionDesktopController(
                controller,
                strong_authenticator=_Verified(),
            )
            stale = DelayedFailure()
            newer = _FakeVoiceRuntime(
                controller,
                tts=adapter._tts,
                on_event=adapter._on_voice_event,
            )
            worker = threading.Thread(
                target=adapter._run_voice_runtime,
                args=(stale,),
            )
            worker.start()
            self.assertTrue(stale.started.wait(1.0))
            controller.loop.states.transition(AssistantState.THINKING, "New runtime state")
            with adapter._lock:
                adapter._voice_runtime = newer
                adapter._voice_thread = threading.current_thread()
                adapter._listening = True

            stale.release.set()
            worker.join(1.0)

            self.assertFalse(worker.is_alive())
            self.assertIs(adapter._voice_runtime, newer)
            self.assertTrue(adapter.status().microphone_active)
            self.assertIs(adapter.status().state, AssistantState.THINKING)
            self.assertFalse(
                any(item.event == "ui_voice_error" for item in adapter.audit_records())
            )
            adapter.close()

    def test_close_cannot_orphan_a_concurrently_starting_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            factory_entered = threading.Event()
            release_factory = threading.Event()
            close_attempted = threading.Event()
            holder: list[_FakeVoiceRuntime] = []
            errors: list[BaseException] = []

            def factory(controller, *, tts, on_event):
                factory_entered.set()
                self.assertTrue(release_factory.wait(1.0))
                runtime = _FakeVoiceRuntime(controller, tts=tts, on_event=on_event)
                holder.append(runtime)
                return runtime

            controller = CompanionController(AsherConfig.load(directory))
            adapter = CompanionDesktopController(
                controller,
                strong_authenticator=_Verified(),
                voice_runtime_factory=factory,
            )

            def start_listening() -> None:
                try:
                    adapter.toggle_listening()
                except BaseException as error:
                    errors.append(error)

            def close_adapter() -> None:
                close_attempted.set()
                try:
                    adapter.close()
                except BaseException as error:
                    errors.append(error)

            starter = threading.Thread(target=start_listening)
            closer = threading.Thread(target=close_adapter)
            starter.start()
            self.assertTrue(factory_entered.wait(1.0))
            closer.start()
            self.assertTrue(close_attempted.wait(1.0))
            release_factory.set()
            starter.join(2.0)
            closer.join(2.0)

            self.assertFalse(starter.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(holder and holder[0].started.is_set())
            self.assertTrue(holder[0].stopped.is_set())
            self.assertIsNone(adapter._voice_runtime)
            self.assertIsNone(adapter._voice_thread)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                adapter.toggle_listening()


if __name__ == "__main__":
    unittest.main()
