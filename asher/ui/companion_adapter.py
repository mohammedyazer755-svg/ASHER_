"""Composition root adapter that connects the real ASHER runtime to Qt.

The Qt layer deliberately depends on a small protocol.  This module is the
bridge between that protocol and :class:`CompanionController`; keeping the
bridge here prevents widgets from reaching into SQLite, policy, or tool
objects directly.  It also gives the UI one stable owner session, so a
confirmation created by a text request cannot accidentally be approved from a
different actor or session.
"""

from __future__ import annotations

import importlib.util
import math
import os
import threading
from dataclasses import replace
from datetime import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from asher.agent.controller import CompanionController
from asher.core.redaction import redact_text
from asher.core.state import AssistantState, StateEvent
from asher.memory.store import MemoryRecord as StoreMemoryRecord, MemoryStore
from asher.security.strong_auth import DenyStrongAuthenticator, StrongAuthenticator
from asher.types import AuthMethod, RiskLevel, Role, utc_now
from asher.ui.controller import (
    AuditRecord,
    ConversationTurn,
    ControllerUnavailable,
    DesktopSettings,
    DesktopStatus,
    DiagnosticReport,
    LiveStep,
    MemoryRecord,
    PendingAction,
    PermissionRecord,
    UserRecord,
)
from asher.voice.tts import TTSManager, build_default_tts


class CompanionDesktopController:
    """Thread-safe UI facade over the authenticated companion runtime.

    ``CompanionController`` remains the source of truth for authorization and
    tool execution.  UI operations are local, explicit user actions; memory
    writes/deletes therefore pass ``confirmed=True`` only after the user has
    clicked the corresponding button in the local UI.
    """

    def __init__(
        self,
        companion: CompanionController,
        *,
        tts: TTSManager | None = None,
        strong_authenticator: StrongAuthenticator | None = None,
        voiceguard: Any | None = None,
        voice_runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.companion = companion
        self._lock = threading.RLock()
        self._reauth_lock = threading.Lock()
        self._owner_session = companion.create_owner_session(AuthMethod.LOCAL_UI)
        self._tts = tts or build_default_tts(selected_profile=companion.config.voice_profile)
        self._strong_auth = strong_authenticator or companion.strong_authenticator or DenyStrongAuthenticator()
        if voiceguard is None:
            try:
                from asher.ui.voiceguard_adapter import VoiceGuardDesktopAdapter

                voiceguard = VoiceGuardDesktopAdapter(companion.config.runtime.root)
            except Exception:
                # The optional adapter itself is safe to omit; the UI will
                # report a precise unavailable-capability message on capture.
                voiceguard = None
        self._voiceguard = voiceguard
        self._voice_runtime_factory = voice_runtime_factory
        self._voice_runtime: Any | None = None
        self._voice_thread: threading.Thread | None = None
        self._closed = False
        self._memory = MemoryStore(companion.database)
        self._listening = False
        self._message = "Ready for Hey Asher"
        self._conversation: list[ConversationTurn] = []
        self._steps: list[LiveStep] = []
        self._manual_pending: PendingAction | None = None
        self._enrollment: dict[str, tuple[str, int]] = {}
        self._settings = DesktopSettings(
            voice_profile=self._tts.selected_profile_name,
            speech_speed=self._tts.selected_profile.speed,
            speech_style=self._tts.selected_profile.style,
            offline_only=not companion.config.openai_enabled,
            api_enabled=companion.config.openai_enabled,
            microphone_index=(
                companion.config.microphone_index
                if isinstance(companion.config.microphone_index, int)
                else None
            ),
        )
        self._unsubscribe_state = companion.loop.states.subscribe(self._on_state_event)

    # ------------------------------------------------------------------ state
    def _on_state_event(self, event: StateEvent) -> None:
        """Mirror the canonical core-state reason into the desktop status text."""

        message = redact_text(event.message).strip()
        if not message:
            return
        with self._lock:
            self._message = message

    def subscribe_state(self, callback: Callable[[StateEvent], None]) -> Callable[[], None]:
        """Expose the existing canonical state stream for the future orb bridge."""

        return self.companion.loop.states.subscribe(callback)

    def _state(self) -> AssistantState:
        if self.companion.emergency_stopped:
            return AssistantState.STOPPED
        active = self.companion.loop.active
        if active is not None and active.waiting_confirmation_id:
            return AssistantState.AWAITING_CONFIRMATION
        return self.companion.loop.states.current

    def status(self) -> DesktopStatus:
        with self._lock:
            state = self._state()
            runtime = self._voice_runtime
            microphone_active = self._listening
            microphone_level = 0.0
            if microphone_active and runtime is not None:
                try:
                    microphone_level = float(
                        getattr(runtime, "microphone_level", 0.0)
                    )
                except (TypeError, ValueError, OverflowError):
                    microphone_level = 0.0
                if not math.isfinite(microphone_level):
                    microphone_level = 0.0
                microphone_level = max(0.0, min(1.0, microphone_level))
            active_owner_session = self.companion.sessions.get(
                self._owner_session.session_id
            )
            owner_session_active = bool(
                active_owner_session is not None
                and active_owner_session.actor.role is Role.OWNER
                and active_owner_session.actor.user_id == self.companion.owner.user_id
            )
            return DesktopStatus(
                state=state,
                message=self._message,
                offline=self._settings.offline_only,
                api_configured=self.companion.config.openai_enabled
                and importlib.util.find_spec("openai") is not None,
                emergency_stopped=self.companion.emergency_stopped,
                microphone_active=microphone_active,
                owner_session_active=owner_session_active,
                microphone_level=microphone_level,
            )

    def toggle_listening(self) -> DesktopStatus:
        runtime_to_stop: Any | None = None
        thread_to_join: threading.Thread | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("The desktop voice controller is closed")
            if self.companion.emergency_stopped:
                raise RuntimeError("Reset the emergency stop before listening")
            starting = not self._listening
            if starting:
                runtime = self._build_voice_runtime()
                thread = threading.Thread(
                    target=self._run_voice_runtime,
                    args=(runtime,),
                    name="asher-voice-runtime",
                    daemon=True,
                )
                self._voice_runtime = runtime
                self._voice_thread = thread
                self._listening = True
                self._message = "Listening for Hey Asher"
                self.companion.loop.states.transition(
                    AssistantState.STANDBY,
                    "Listening for Hey Asher",
                    microphone_active=True,
                )
            else:
                runtime_to_stop = self._voice_runtime
                thread_to_join = self._voice_thread
                self._listening = False
                self._voice_runtime = None
                self._voice_thread = None
                self._message = "Listening paused"
                self.companion.loop.states.transition(
                    AssistantState.STANDBY,
                    "Listening paused",
                    microphone_active=False,
                )
            self.companion.audit.append(
                "ui_listening",
                actor_id=self._owner_session.actor.user_id,
                session_id=self._owner_session.session_id,
                outcome="started" if starting else "stopped",
            )
            if starting:
                # Publish and start the runtime under the same lock used by
                # emergency_stop(). This closes the window where an emergency
                # stop could observe no runtime, latch the stop, and then have
                # this worker start an orphan microphone thread afterwards.
                thread.start()
        if not starting and runtime_to_stop is not None:
            runtime_to_stop.stop()
            if thread_to_join is not None and thread_to_join is not threading.current_thread():
                thread_to_join.join(timeout=2.0)
        return self.status()

    def _build_voice_runtime(self) -> Any:
        factory = self._voice_runtime_factory
        if factory is not None:
            return factory(
                self.companion,
                tts=self._tts,
                on_event=self._on_voice_event,
            )
        from asher.voice.runtime import (
            SoundDeviceBackend,
            VoiceRuntime,
            load_active_voiceguard_verifier,
        )

        # A concrete UI selection takes precedence. Otherwise use the
        # environment-backed application default, which may be either a
        # sounddevice index or an exact device name.
        device = self._settings.microphone_index
        if device is None:
            device = self.companion.config.microphone_index
        return VoiceRuntime(
            self.companion,
            backend=SoundDeviceBackend(device=device),
            tts=self._tts,
            voiceguard=load_active_voiceguard_verifier(self.companion),
            on_event=self._on_voice_event,
        )

    def _run_voice_runtime(self, runtime: Any) -> None:
        try:
            runtime.run_forever()
        except Exception as error:
            owned_runtime = False
            with self._lock:
                if self._voice_runtime is runtime:
                    owned_runtime = True
                    self._listening = False
                    self._message = f"Voice input unavailable: {type(error).__name__}"
                    self._voice_runtime = None
                    self._voice_thread = None
                    self.companion.loop.states.transition(
                        AssistantState.ERROR,
                        self._message,
                        error=type(error).__name__,
                    )
            if owned_runtime:
                self.companion.audit.append(
                    "ui_voice_error",
                    actor_id=self._owner_session.actor.user_id,
                    session_id=self._owner_session.session_id,
                    outcome=type(error).__name__,
                )
        finally:
            with self._lock:
                if self._voice_runtime is runtime:
                    self._voice_runtime = None
                    self._voice_thread = None
                    self._listening = False

    def _on_voice_event(self, event: Any) -> None:
        """Consume runtime events without touching Qt objects from its thread."""

        with self._lock:
            message = redact_text(getattr(event, "message", ""))
            self._message = message or self._message
            transcript = getattr(event, "transcript", None)
            if getattr(event, "kind", "") == "transcript":
                # VoiceRuntime emits this event only for one accepted final
                # command. Prefer its canonical post-wake/post-normalization
                # message so a decoded wake prefix never pollutes history.
                text = redact_text(getattr(event, "message", "")).strip()
                if not text and transcript is not None:
                    text = redact_text(getattr(transcript, "normalized_text", "")).strip()
                if text:
                    self._conversation.append(ConversationTurn("You", text))
            reply = getattr(event, "reply", None)
            if getattr(event, "kind", "") == "reply" and reply is not None:
                text = redact_text(getattr(reply, "text", "")).strip()
                if text:
                    self._conversation.append(ConversationTurn("Asher", text))
                for update in getattr(reply, "updates", ()):
                    status = str(getattr(update, "status", "step"))
                    self._steps.append(
                        LiveStep(
                            uuid4().hex,
                            redact_text(getattr(update, "message", "Voice task")),
                            status,
                        )
                    )

    # --------------------------------------------------------------- requests
    def _active_owner_session(self):
        active = self.companion.sessions.get(self._owner_session.session_id)
        if active is None or active.actor.user_id != self._owner_session.actor.user_id:
            raise PermissionError(
                "The desktop session expired. Re-authenticate locally before continuing."
            )
        return active

    def _audit_reauthentication(
        self,
        outcome: str,
        *,
        session_id: str,
        reason: str | None = None,
    ) -> None:
        details = {"method": AuthMethod.DEVICE_CREDENTIAL.value}
        if reason:
            details["reason"] = reason
        self.companion.audit.append(
            "ui_owner_reauthentication",
            actor_id=self.companion.owner.user_id,
            session_id=session_id,
            outcome=outcome,
            details=details,
        )

    def _lock_expired_owner_session(self, message: str, *, reason: str) -> None:
        with self._lock:
            self._message = message
        self.companion.loop.states.transition(
            AssistantState.LOCKED,
            message,
            reason=reason,
            actor_id=self.companion.owner.user_id,
        )

    def reauthenticate_owner(self) -> DesktopStatus:
        """Create a new owner session only after fresh strong device proof."""

        with self._reauth_lock:
            expected_owner_id = self.companion.owner.user_id
            old_session = self._owner_session
            old_session_id = old_session.session_id
            owner = self.companion.users.get(expected_owner_id)
            if (
                owner is None
                or owner.role is not Role.OWNER
                or owner.user_id != expected_owner_id
            ):
                message = "The persistent owner identity is unavailable; session remains locked"
                self._lock_expired_owner_session(
                    message,
                    reason="persistent_owner_unavailable",
                )
                self._audit_reauthentication(
                    "denied",
                    session_id=old_session_id,
                    reason="persistent_owner_unavailable",
                )
                raise PermissionError(message)

            if self.companion.sessions.get(old_session_id) is not None:
                with self._lock:
                    self._message = "Owner session is already active"
                self._audit_reauthentication(
                    "denied",
                    session_id=old_session_id,
                    reason="session_still_active",
                )
                raise RuntimeError(
                    "The owner session is still active; re-authentication is not required"
                )

            try:
                authentication = self._strong_auth.verify(
                    "Re-authenticate the ASHER owner session"
                )
            except Exception as error:
                message = "Device authentication could not be completed; session remains locked"
                self._lock_expired_owner_session(message, reason="device_auth_error")
                self._audit_reauthentication(
                    "failed",
                    session_id=old_session_id,
                    reason="device_auth_error",
                )
                raise PermissionError(message) from error
            if not authentication.verified:
                message = "Device authentication was not verified; session remains locked"
                self._lock_expired_owner_session(message, reason="device_auth_denied")
                self._audit_reauthentication(
                    "denied",
                    session_id=old_session_id,
                    reason="device_auth_denied",
                )
                raise PermissionError(message)

            # Re-read after the device prompt so a changed/revoked identity is
            # never authenticated using stale actor data.
            verified_owner = self.companion.users.get(expected_owner_id)
            if (
                verified_owner is None
                or verified_owner.role is not Role.OWNER
                or verified_owner.user_id != owner.user_id
            ):
                message = "The persistent owner identity changed; session remains locked"
                self._lock_expired_owner_session(
                    message,
                    reason="persistent_owner_changed",
                )
                self._audit_reauthentication(
                    "denied",
                    session_id=old_session_id,
                    reason="persistent_owner_changed",
                )
                raise PermissionError(message)

            # A confirmation is deliberately bound to the session that
            # created it and must never be rebound to the fresh session. Once
            # strong device proof succeeds, cancel an old-session checkpoint
            # so it cannot become an invisible, permanently busy agent plan.
            active = self.companion.loop.active
            if (
                active is not None
                and active.waiting_confirmation_id
                and active.session.session_id == old_session_id
                and active.session.actor.user_id == expected_owner_id
            ):
                self.companion.loop.reject(
                    active.waiting_confirmation_id,
                    old_session,
                )
                # The confirmation can expire between inspection and reject.
                # In that case the exact same plan remains paused; cancel it
                # rather than leaving the agent permanently busy.
                if self.companion.loop.active is active:
                    self.companion.loop.cancel(
                        "Expired owner-session confirmation cancelled during re-authentication"
                    )

            fresh_session = self.companion.sessions.create(
                verified_owner,
                AuthMethod.DEVICE_CREDENTIAL,
            )
            self.companion.sessions.invalidate(old_session_id)
            with self._lock:
                self._owner_session = fresh_session
                self._manual_pending = None
                self._message = "Owner session re-authenticated with device credentials"
            self.companion.loop.states.transition(
                AssistantState.AUTHENTICATED,
                "Owner session re-authenticated",
                actor_id=verified_owner.user_id,
                auth_method=AuthMethod.DEVICE_CREDENTIAL.value,
            )
            self._audit_reauthentication(
                "complete",
                session_id=fresh_session.session_id,
            )
            return self.status()

    def submit_text(self, text: str) -> ConversationTurn:
        clean = str(text).strip()
        if not clean:
            raise ValueError("Enter a message first")
        if self.companion.emergency_stopped:
            raise RuntimeError("Reset the emergency stop before sending a command")
        step_id = uuid4().hex
        with self._lock:
            self._message = "Understanding your request"
            self._steps.append(LiveStep(step_id, "Understand and authorize request", "running"))
            self._conversation.append(ConversationTurn("You", redact_text(clean)))
        try:
            session = self._active_owner_session()
            reply = self.companion.handle_text(clean, session)
            response = ConversationTurn("Asher", redact_text(reply.text))
            with self._lock:
                self._conversation.append(response)
                detail = "Confirmation required" if reply.confirmation_id else "Request processed"
                status = "awaiting_confirmation" if reply.confirmation_id else "complete"
                self._replace_step(step_id, status, detail)
                self._message = reply.text
            self._speak_reply(
                reply.text,
                return_state=(
                    AssistantState.AWAITING_CONFIRMATION
                    if reply.confirmation_id
                    else AssistantState.SUCCESS
                ),
            )
            return response
        except Exception as error:
            with self._lock:
                self._replace_step(step_id, "error", f"{type(error).__name__}")
                self._message = "Request failed safely"
            raise

    def _speak_reply(self, text: str, *, return_state: AssistantState) -> None:
        clean = str(text).strip()
        if not clean:
            return
        try:
            self.companion.loop.states.transition(
                AssistantState.SPEAKING,
                "Speaking",
                voice_profile=self._tts.selected_profile_name,
            )
            handle = self._tts.speak_async(clean, interrupt=True)
        except Exception:
            # Text mode remains usable when no local speech provider exists.
            self.companion.loop.states.transition(
                return_state,
                "Ready",
            )
            return
        wait = getattr(handle, "wait", None)
        if not callable(wait):
            self.companion.loop.states.transition(return_state, "Ready")
            return

        def finish_state() -> None:
            try:
                result = handle.wait()
            except Exception as error:
                self.companion.loop.states.transition(
                    AssistantState.ERROR,
                    "Speech output failed",
                    error=type(error).__name__,
                )
                return
            if result is not None and not getattr(result, "success", False):
                if getattr(result, "cancelled", False):
                    return
                self.companion.loop.states.transition(
                    AssistantState.ERROR,
                    "Speech output failed",
                    error=getattr(result, "error", None),
                )
                return
            if not self.companion.emergency_stopped:
                self.companion.loop.states.transition(return_state, "Ready")

        threading.Thread(
            target=finish_state,
            name="asher-ui-tts-state",
            daemon=True,
        ).start()

    def _replace_step(self, step_id: str, status: str, detail: str = "") -> None:
        self._steps = [
            replace(item, status=status, detail=redact_text(detail)) if item.step_id == step_id else item
            for item in self._steps
        ]

    def conversation(self) -> tuple[ConversationTurn, ...]:
        with self._lock:
            return tuple(self._conversation)

    def live_steps(self) -> tuple[LiveStep, ...]:
        with self._lock:
            return tuple(self._steps)

    # ------------------------------------------------------------ confirmations
    def _runtime_pending(self) -> PendingAction | None:
        active = self.companion.loop.active
        if active is None or not active.waiting_confirmation_id:
            return None
        confirmation = self.companion.registry.confirmations.get(active.waiting_confirmation_id)
        if confirmation is None:
            return None
        return PendingAction(
            confirmation_id=confirmation.confirmation_id,
            action=confirmation.tool_name,
            target=confirmation.target,
            effect=confirmation.effect,
            preview=confirmation.preview,
            risk=confirmation.risk,
            expires_at=confirmation.expires_at,
        )

    def pending_action(self) -> PendingAction | None:
        pending = self._manual_pending or self._runtime_pending()
        if pending is not None and pending.expires_at <= utc_now():
            self._manual_pending = None
            return None
        return pending

    def stage_confirmation(self, action: PendingAction) -> None:
        if action.expires_at <= utc_now():
            raise ValueError("Confirmation has already expired")
        with self._lock:
            self._manual_pending = action
            self._message = "Review the pending action"

    def _strongly_authenticate(self, pending: PendingAction) -> bool:
        if pending.risk < RiskLevel.SENSITIVE:
            return True
        result = self._strong_auth.verify(
            f"Authorize ASHER action: {pending.action} targeting {pending.target}"
        )
        return bool(result.verified)

    def approve_pending(self) -> bool:
        pending = self.pending_action()
        if pending is None:
            return False
        runtime_pending = self._manual_pending is not pending
        # The real controller owns the OS-bound verification for runtime
        # confirmations.  Manual previews (used by an injected UI adapter)
        # still need the same verifier here.
        if (not runtime_pending) and not self._strongly_authenticate(pending):
            with self._lock:
                self._message = "Device authentication was not verified; action remains pending"
            return False
        if self._manual_pending is pending:
            with self._lock:
                self._manual_pending = None
                self._message = "Action approved in the local UI"
            return True
        # CompanionController performs the session/actor-bound confirmation
        # check and, for sensitive actions, invokes its OS-bound verifier.  A
        # boolean here is only a request to begin that flow, never proof by
        # itself.
        session = self._active_owner_session()
        reply = self.companion.approve(
            pending.confirmation_id,
            session,
            device_authenticated=pending.risk >= RiskLevel.SENSITIVE,
        )
        with self._lock:
            self._message = redact_text(reply.text)
        return bool(reply.updates and any(item.status in {"complete", "step"} for item in reply.updates))

    def reject_pending(self) -> bool:
        pending = self.pending_action()
        if pending is None:
            return False
        if self._manual_pending is pending:
            self._manual_pending = None
            self._message = "Action rejected"
            return True
        session = self._active_owner_session()
        reply = self.companion.reject(pending.confirmation_id, session)
        with self._lock:
            self._message = redact_text(reply.text)
        # ``denied`` means the cancellation request itself was refused and the
        # action is still pending. Report True only when the underlying plan
        # was actually rejected/cancelled.
        return any(item.status in {"failed", "cancelled"} for item in reply.updates)

    # ----------------------------------------------------------------- memory
    @property
    def _memory_store(self) -> MemoryStore:
        return self._memory

    def _require_strong_auth(self, action: str, target: str) -> None:
        result = self._strong_auth.verify(f"Authorize ASHER {action}: {target}")
        if not result.verified:
            raise PermissionError("Device authentication was not verified; no change was made")

    def _ui_memory(self, record: StoreMemoryRecord) -> MemoryRecord:
        # The storage layer intentionally has only normal/sensitive semantics;
        # the UI's public/private labels are presentation choices.  Values are
        # still masked by MemoryPage until the user explicitly reveals them.
        return MemoryRecord(
            memory_id=record.memory_id,
            key=record.key,
            value=record.value,
            memory_type=self._ui_memory_type(record.memory_type),
            sensitivity="sensitive" if record.sensitivity == "sensitive" else "private",
            updated_at=record.updated_at,
        )

    def list_memories(self) -> tuple[MemoryRecord, ...]:
        session = self._active_owner_session()
        records = self._memory_store.list(
            session.actor,
            owner_id=session.actor.user_id,
            include_sensitive=True,
        )
        return tuple(self._ui_memory(item) for item in records)

    @staticmethod
    def _storage_sensitivity(value: str) -> str:
        if value == "sensitive":
            return "sensitive"
        return "normal"

    @staticmethod
    def _storage_memory_type(value: str) -> str:
        return {
            "preference": "preference_routine",
            "relationship": "people_relationship",
            "goal": "project_goal",
        }.get(value, value)

    @staticmethod
    def _ui_memory_type(value: str) -> str:
        return {
            "preference_routine": "preference",
            "people_relationship": "relationship",
            "project_goal": "goal",
        }.get(value, value)

    def create_memory(
        self,
        key: str,
        value: str,
        memory_type: str = "semantic",
        sensitivity: str = "private",
        consented: bool = False,
    ) -> MemoryRecord:
        session = self._active_owner_session()
        if sensitivity == "sensitive":
            self._require_strong_auth("save sensitive memory", key)
        record = self._memory_store.put(
            session.actor,
            owner_id=session.actor.user_id,
            memory_type=self._storage_memory_type(memory_type),
            key=key,
            value=value,
            source="desktop_ui",
            sensitivity=self._storage_sensitivity(sensitivity),
            consented=consented,
            confirmed=True,
        )
        self.companion.audit.append(
            "ui_memory_create",
            actor_id=session.actor.user_id,
            session_id=session.session_id,
            outcome="complete",
            details={"memory_id": record.memory_id, "sensitivity": sensitivity},
        )
        return self._ui_memory(record)

    def update_memory(
        self,
        memory_id: str,
        value: str,
        memory_type: str = "semantic",
        sensitivity: str = "private",
        consented: bool = False,
    ) -> MemoryRecord:
        session = self._active_owner_session()
        current = self._memory_store.get(session.actor, memory_id)
        if current is None:
            raise KeyError("Memory was not found")
        if sensitivity == "sensitive" or current.sensitivity == "sensitive":
            self._require_strong_auth("update sensitive memory", current.key)
        record = self._memory_store.update(
            session.actor,
            memory_id,
            memory_type=self._storage_memory_type(memory_type),
            value=value,
            source="desktop_ui",
            sensitivity=self._storage_sensitivity(sensitivity),
            consented=consented,
            confirmed=True,
        )
        self.companion.audit.append(
            "ui_memory_update",
            actor_id=session.actor.user_id,
            session_id=session.session_id,
            outcome="complete",
            details={"memory_id": record.memory_id, "sensitivity": sensitivity},
        )
        return self._ui_memory(record)

    def delete_memory(self, memory_id: str) -> bool:
        session = self._active_owner_session()
        current = self._memory_store.get(session.actor, memory_id)
        if current is None:
            return False
        self._require_strong_auth("delete memory", current.key)
        deleted = self._memory_store.delete(session.actor, memory_id, confirmed=True)
        if deleted:
            self.companion.audit.append(
                "ui_memory_delete",
                actor_id=session.actor.user_id,
                session_id=session.session_id,
                outcome="complete",
                details={"memory_id": memory_id},
            )
        return deleted

    def _audit_memory_export(
        self,
        session: Any,
        outcome: str,
        *,
        record_count: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Audit export metadata without retaining values or destination data."""

        details: dict[str, Any] = {"format": "json"}
        if record_count is not None:
            details["record_count"] = int(record_count)
        if reason:
            details["reason"] = reason
        self.companion.audit.append(
            "ui_memory_export",
            actor_id=session.actor.user_id,
            session_id=session.session_id,
            outcome=outcome,
            details=details,
        )

    def export_memories(self, destination: str | Path) -> Path:
        """Export owner memories atomically after fresh device authentication."""

        session = self._active_owner_session()
        if (
            session.actor.role is not Role.OWNER
            or session.actor.user_id != self.companion.owner.user_id
        ):
            self._audit_memory_export(session, "denied", reason="owner_required")
            raise PermissionError("Only the owner can export private memory")

        destination_text = str(destination).strip()
        selected_path = Path(destination_text).expanduser() if destination_text else None
        if (
            selected_path is None
            or not selected_path.is_absolute()
            or selected_path.suffix.casefold() != ".json"
        ):
            self._audit_memory_export(session, "denied", reason="invalid_json_destination")
            raise ValueError("Select an absolute JSON destination path")

        try:
            self._require_strong_auth(
                "export private memories",
                "the selected local JSON file",
            )
        except Exception as error:
            self._audit_memory_export(
                session,
                "denied" if isinstance(error, PermissionError) else "failed",
                reason="device_auth_denied" if isinstance(error, PermissionError) else "device_auth_error",
            )
            raise

        records = self._memory_store.list(
            session.actor,
            owner_id=self.companion.owner.user_id,
            include_sensitive=True,
            limit=1000,
        )
        try:
            exported = self._memory_store.export_json(
                session.actor,
                owner_id=self.companion.owner.user_id,
                destination=selected_path,
            )
        except Exception:
            self._audit_memory_export(
                session,
                "failed",
                record_count=len(records),
                reason="write_failed",
            )
            raise

        self._audit_memory_export(session, "complete", record_count=len(records))
        with self._lock:
            self._message = f"Exported {len(records)} memories to the selected JSON file"
        return exported

    # --------------------------------------------------------------- users/auth
    def _user_record(self, actor: Any) -> UserRecord:
        status, samples = self._enrollment.get(actor.user_id, ("not_enrolled", 0))
        if actor.user_id == self._owner_session.actor.user_id and status == "not_enrolled":
            status = "owner session"
        return UserRecord(actor.user_id, actor.display_name, actor.role.value, status, samples)

    def list_users(self) -> tuple[UserRecord, ...]:
        self._active_owner_session()
        return tuple(self._user_record(actor) for actor in self.companion.users.list_active())

    def enroll_user(self, display_name: str, role: str) -> UserRecord:
        session = self._active_owner_session()
        self._require_strong_auth("enroll a user", display_name)
        actor = self.companion.users.create(display_name, Role(role))
        self._enrollment[actor.user_id] = ("collecting_samples", 0)
        begin_user = getattr(self._voiceguard, "begin_user", None)
        if callable(begin_user):
            # Consent is collected by the UI dialog before the first sample;
            # keep enrollment pending until that explicit acknowledgement.
            self._enrollment[actor.user_id] = ("awaiting_recording_consent", 0)
        self.companion.audit.append(
            "ui_user_enroll",
            actor_id=session.actor.user_id,
            session_id=session.session_id,
            outcome="started",
            details={"user_id": actor.user_id, "role": role},
        )
        return self._user_record(actor)

    def confirm_voice_recording_consent(self, user_id: str) -> UserRecord:
        """Record the user's explicit consent before the first microphone use."""

        actor = self.companion.users.get(user_id)
        if actor is None:
            raise KeyError("User was not found")
        self._active_owner_session()
        self._require_strong_auth("confirm voice recording consent", actor.display_name)
        begin_user = getattr(self._voiceguard, "begin_user", None)
        if not callable(begin_user):
            raise ControllerUnavailable("VoiceGuard recording service is not connected")
        recovered = begin_user(user_id, actor.role.value, consent=True)
        samples = self._enrollment.get(user_id, ("collecting_samples", 0))[1]
        if isinstance(recovered, int) and not isinstance(recovered, bool):
            samples = max(0, recovered)
        self._enrollment[user_id] = ("collecting_samples", samples)
        return self._user_record(actor)

    def capture_voice_sample(self, user_id: str) -> UserRecord:
        if self._voiceguard is None:
            raise ControllerUnavailable(
                "VoiceGuard recording service is not connected. Enroll through the documented recorder first."
            )
        actor = self.companion.users.get(user_id)
        if actor is None:
            raise KeyError("User was not found")
        self._active_owner_session()
        self._require_strong_auth("capture a VoiceGuard sample", actor.display_name)
        count = int(self._voiceguard.capture_sample(user_id))
        self._enrollment[user_id] = ("collecting_samples", count)
        return self._user_record(actor)

    def train_voiceguard(self, user_id: str) -> UserRecord:
        if self._voiceguard is None:
            raise ControllerUnavailable(
                "VoiceGuard training service is not connected. Configure a real dataset/model adapter first."
            )
        actor = self.companion.users.get(user_id)
        if actor is None:
            raise KeyError("User was not found")
        self._active_owner_session()
        self._require_strong_auth("train VoiceGuard", actor.display_name)
        self._voiceguard.train(user_id)
        samples = self._enrollment.get(user_id, ("collecting_samples", 0))[1]
        self._enrollment[user_id] = ("trained", samples)
        return self._user_record(actor)

    def revoke_user(self, user_id: str) -> bool:
        session = self._active_owner_session()
        actor = self.companion.users.get(user_id)
        if actor is None:
            return False
        self._require_strong_auth("revoke user", actor.display_name)
        result = self.companion.users.revoke(user_id)
        if result:
            self.companion.sessions.invalidate_user(user_id)
            if self._voiceguard is not None:
                try:
                    self._voiceguard.revoke(user_id)
                except Exception:
                    # Account revocation remains authoritative even if an old
                    # recording registry has no entry; audit the safe state.
                    pass
            self._enrollment.pop(user_id, None)
            self.companion.audit.append(
                "ui_user_revoke",
                actor_id=session.actor.user_id,
                session_id=session.session_id,
                outcome="complete",
                details={"user_id": user_id},
            )
        return result

    # ------------------------------------------------------------ permissions
    def _capability_risks(self) -> dict[str, RiskLevel]:
        risks = {"conversation": RiskLevel.CONVERSATION}
        for definition in self.companion.registry._tools.values():
            risks.setdefault(definition.policy.capability, definition.policy.risk)
        return risks

    def list_permissions(self) -> tuple[PermissionRecord, ...]:
        self._active_owner_session()
        risks = self._capability_risks()
        active = self.companion.users.list_active()
        values: list[PermissionRecord] = []
        for actor in active:
            role = actor.role
            for capability, risk in sorted(risks.items()):
                allowed = role is Role.OWNER
                if role is Role.GUEST:
                    allowed = risk is RiskLevel.CONVERSATION
                elif role is Role.TRUSTED:
                    allowed = capability in actor.permissions
                values.append(PermissionRecord(role.value, capability, risk, allowed, actor.user_id))
        return tuple(values)

    def set_permission(
        self,
        role: str,
        capability: str,
        allowed: bool,
        actor_id: str | None = None,
    ) -> PermissionRecord:
        session = self._active_owner_session()
        self._require_strong_auth("change permissions", actor_id or role)
        role_enum = Role(role)
        risks = self._capability_risks()
        if capability not in risks:
            raise KeyError("Permission capability was not found")
        if role_enum is Role.GUEST and allowed and risks[capability] is not RiskLevel.CONVERSATION:
            raise ValueError("Guests cannot receive private or action permissions")
        if role_enum is Role.OWNER:
            # Owner permissions are intentionally non-revocable in this UI.
            if not allowed:
                raise ValueError("The owner cannot be denied core permissions")
            return PermissionRecord(role, capability, risks[capability], True, session.actor.user_id)
        if role_enum is Role.GUEST:
            if allowed and risks[capability] is not RiskLevel.CONVERSATION:
                raise ValueError("Guests cannot receive private or action permissions")
            return PermissionRecord(role, capability, risks[capability], risks[capability] is RiskLevel.CONVERSATION, "guest")
        if not actor_id:
            raise ValueError("Select one trusted user before changing a permission")
        actor = self.companion.users.get(actor_id)
        if actor is None or actor.role is not role_enum:
            raise KeyError("The selected user is not an active trusted user")
        permissions = set(actor.permissions)
        if allowed:
            permissions.add(capability)
        else:
            permissions.discard(capability)
        updated = self.companion.users.set_permissions(actor.user_id, permissions)
        self.companion.audit.append(
            "ui_permission_change",
            actor_id=session.actor.user_id,
            session_id=session.session_id,
            outcome="complete",
            details={"subject_user_id": updated.user_id, "capability": capability, "allowed": allowed},
        )
        return PermissionRecord(role, capability, risks[capability], capability in updated.permissions, updated.user_id)

    # ------------------------------------------------------------------ audit
    def audit_records(self) -> tuple[AuditRecord, ...]:
        records: list[AuditRecord] = []
        for item in self.companion.audit.read_recent(200):
            try:
                timestamp = datetime.fromisoformat(str(item.get("timestamp")))
            except (TypeError, ValueError):
                timestamp = utc_now()
            records.append(
                AuditRecord(
                    timestamp=timestamp,
                    event=str(item.get("event", "event")),
                    result=str(item.get("outcome", "")),
                    details=item.get("details", {}) if isinstance(item.get("details", {}), dict) else {},
                )
            )
        return tuple(reversed(records))

    # --------------------------------------------------------------- settings
    def settings(self) -> DesktopSettings:
        with self._lock:
            return self._settings

    def voice_profiles(self) -> tuple[tuple[str, str], ...]:
        return tuple((item.name, item.label) for item in self._tts.registry.all())

    def update_settings(self, **changes: Any) -> DesktopSettings:
        session = self._active_owner_session()
        allowed = {"voice_profile", "speech_speed", "speech_style", "offline_only", "api_enabled", "microphone_index"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown setting: {sorted(unknown)[0]}")
        with self._lock:
            updated = replace(self._settings, **changes)
            profile = self._tts.registry.get(updated.voice_profile)
            if updated.offline_only and profile.provider != "sapi":
                raise ValueError("Select an offline voice while offline-only mode is enabled")
            api_ready = self.companion.config.openai_enabled and importlib.util.find_spec("openai") is not None
            if updated.api_enabled and not api_ready:
                raise ValueError("OpenAI API key and package are not configured")
            if profile.provider == "openai" and not updated.api_enabled:
                raise ValueError("Enable the configured API before selecting an online voice")
            if not 0.5 <= float(updated.speech_speed) <= 2.0:
                raise ValueError("Speech speed must be between 0.5 and 2.0")
            self._tts.set_profile(updated.voice_profile)
            self._tts.registry.register(
                profile.adjusted(speed=updated.speech_speed, style=updated.speech_style),
                replace_existing=True,
            )
            # This is a global privacy control, not just a speech preference:
            # an offline-only/API-disabled UI must not route unsupported text
            # requests to the remote planner either.
            self.companion.planner.online_enabled = bool(
                updated.api_enabled and not updated.offline_only
            )
            self._settings = updated
            self.companion.audit.append(
                "ui_settings_change",
                actor_id=session.actor.user_id,
                session_id=session.session_id,
                outcome="complete",
                details={"fields": sorted(changes)},
            )
            return updated

    def run_diagnostics(self) -> DiagnosticReport:
        checks = {
            "Desktop UI": "available" if importlib.util.find_spec("PySide6") else "PySide6 not installed",
            "Microphone backend": "available" if importlib.util.find_spec("sounddevice") else "sounddevice not installed",
            "Faster-Whisper": "available" if importlib.util.find_spec("faster_whisper") else "not installed",
            "Offline TTS": "Windows SAPI provider registered" if "sapi" in self._tts.provider_names() else "not registered",
            "OpenAI API": "configured" if self.companion.config.openai_enabled and importlib.util.find_spec("openai") else "not configured",
            "Ollama fallback": "configured (checked on command)" if self.companion.config.ollama_url else "not configured",
            "VoiceGuard": "adapter connected" if self._voiceguard is not None else "enrollment adapter not connected",
        }
        healthy = checks["Desktop UI"] == "available" and checks["Offline TTS"] != "not registered"
        return DiagnosticReport(checks, "Core desktop services are ready" if healthy else "One or more optional services need attention", healthy)

    # --------------------------------------------------------------- stop/reset
    def emergency_stop(self) -> DesktopStatus:
        runtime: Any | None
        thread: threading.Thread | None
        # Serialize the latch/snapshot with toggle_listening(). Once this lock
        # is released, no new runtime can pass its emergency-stop check.
        with self._lock:
            self.companion.emergency_stop()
            runtime = self._voice_runtime
            thread = self._voice_thread
            self._listening = False
            self._voice_runtime = None
            self._voice_thread = None
            self._manual_pending = None
            self._message = "Emergency stop is active"
            self._steps = [
                replace(item, status="cancelled", detail="Emergency stop")
                if item.status in {"queued", "running", "awaiting_confirmation"}
                else item
                for item in self._steps
            ]
        if runtime is not None:
            runtime.stop()
        self._tts.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        return self.status()

    def close(self) -> None:
        """Stop background voice/TTS work when the desktop window closes."""

        # Share the lifecycle lock with toggle_listening() so a closing window
        # cannot lose a runtime that is concurrently being constructed.
        with self._lock:
            self._closed = True
            runtime = self._voice_runtime
            thread = self._voice_thread
            self._voice_runtime = None
            self._voice_thread = None
            self._listening = False
        if runtime is not None:
            runtime.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._tts.stop()

    def reset_emergency_stop(self) -> DesktopStatus:
        if not self.companion.reset_emergency_stop(local_ui_confirmed=True):
            raise RuntimeError("Emergency stop could not be reset")
        with self._lock:
            self._message = "Emergency stop reset locally"
        return self.status()


__all__ = ["CompanionDesktopController"]
