"""UI-facing controller contracts and a safe local desktop controller.

Qt widgets depend on this protocol rather than importing agent, memory, or
security implementations. A production composition root can inject an adapter;
the included controller keeps every view interactive without performing real
external actions or pretending that VoiceGuard training has occurred.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from asher.core.cancellation import EmergencyStop
from asher.core.redaction import contains_prohibited_secret, redact_mapping, redact_text
from asher.core.state import AssistantState
from asher.types import RiskLevel, utc_now
from asher.voice.tts import TTSManager, build_default_tts


class ControllerUnavailable(RuntimeError):
    """A UI capability has no configured backend."""


SUPPORTED_MEMORY_TYPES = frozenset(
    {
        "episodic",
        "semantic",
        "preference",
        "relationship",
        "goal",
        "task",
        "interaction_preference",
        "people_relationship",
        "preference_routine",
        "project_goal",
    }
)
SUPPORTED_SENSITIVITIES = frozenset({"public", "private", "sensitive"})


@dataclass(frozen=True, slots=True)
class DesktopStatus:
    state: AssistantState
    message: str
    offline: bool
    api_configured: bool
    emergency_stopped: bool


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    sender: str
    message: str
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class LiveStep:
    step_id: str
    description: str
    status: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PendingAction:
    confirmation_id: str
    action: str
    target: str
    effect: str
    preview: Mapping[str, Any]
    risk: RiskLevel
    expires_at: datetime = field(default_factory=lambda: utc_now() + timedelta(seconds=90))


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    key: str
    value: str
    memory_type: str = "semantic"
    sensitivity: str = "private"
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class UserRecord:
    user_id: str
    display_name: str
    role: str
    enrollment_status: str = "not_enrolled"
    samples: int = 0


@dataclass(frozen=True, slots=True)
class PermissionRecord:
    role: str
    capability: str
    risk: RiskLevel
    allowed: bool
    actor_id: str = ""


@dataclass(frozen=True, slots=True)
class AuditRecord:
    timestamp: datetime
    event: str
    result: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DesktopSettings:
    voice_profile: str = "asher_male"
    speech_speed: float = 1.0
    speech_style: str = "warm, natural, practical, and concise"
    offline_only: bool = True
    api_enabled: bool = False
    microphone_index: int | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    checks: Mapping[str, str]
    summary: str
    healthy: bool


@runtime_checkable
class VoiceGuardAdapter(Protocol):
    def capture_sample(self, user_id: str, *, contains_wake_phrase: bool = False) -> int: ...

    def train(self, user_id: str) -> Mapping[str, Any]: ...

    def revoke(self, user_id: str) -> None: ...


@runtime_checkable
class DesktopControllerProtocol(Protocol):
    def status(self) -> DesktopStatus: ...

    def toggle_listening(self) -> DesktopStatus: ...

    def submit_text(self, text: str) -> ConversationTurn: ...

    def conversation(self) -> tuple[ConversationTurn, ...]: ...

    def live_steps(self) -> tuple[LiveStep, ...]: ...

    def pending_action(self) -> PendingAction | None: ...

    def stage_confirmation(self, action: PendingAction) -> None: ...

    def approve_pending(self) -> bool: ...

    def reject_pending(self) -> bool: ...

    def list_memories(self) -> tuple[MemoryRecord, ...]: ...

    def create_memory(self, key: str, value: str, memory_type: str = "semantic", sensitivity: str = "private", consented: bool = False) -> MemoryRecord: ...

    def update_memory(self, memory_id: str, value: str, memory_type: str = "semantic", sensitivity: str = "private", consented: bool = False) -> MemoryRecord: ...

    def delete_memory(self, memory_id: str) -> bool: ...

    def list_users(self) -> tuple[UserRecord, ...]: ...

    def enroll_user(self, display_name: str, role: str) -> UserRecord: ...

    def capture_voice_sample(self, user_id: str) -> UserRecord: ...

    def train_voiceguard(self, user_id: str) -> UserRecord: ...

    def revoke_user(self, user_id: str) -> bool: ...

    def list_permissions(self) -> tuple[PermissionRecord, ...]: ...

    def set_permission(self, role: str, capability: str, allowed: bool, actor_id: str | None = None) -> PermissionRecord: ...

    def audit_records(self) -> tuple[AuditRecord, ...]: ...

    def settings(self) -> DesktopSettings: ...

    def voice_profiles(self) -> tuple[tuple[str, str], ...]: ...

    def update_settings(self, **changes: Any) -> DesktopSettings: ...

    def run_diagnostics(self) -> DiagnosticReport: ...

    def emergency_stop(self) -> DesktopStatus: ...

    def reset_emergency_stop(self) -> DesktopStatus: ...


class DesktopController:
    """Thread-safe default implementation suitable for the desktop shell."""

    def __init__(
        self,
        *,
        command_handler: Callable[[str], str] | None = None,
        tts: TTSManager | None = None,
        emergency_stop: EmergencyStop | None = None,
        voiceguard: VoiceGuardAdapter | None = None,
        strong_auth: Callable[[PendingAction], bool] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._handler = command_handler
        self._tts = tts or build_default_tts()
        self._emergency = emergency_stop or EmergencyStop()
        self._voiceguard = voiceguard
        self._strong_auth = strong_auth
        self._state = AssistantState.STANDBY
        self._message = "Ready for Hey Asher"
        self._conversation: list[ConversationTurn] = []
        self._steps: list[LiveStep] = []
        self._pending: PendingAction | None = None
        self._memories: dict[str, MemoryRecord] = {}
        self._users: dict[str, UserRecord] = {
            "owner": UserRecord("owner", "Owner", "owner")
        }
        self._permissions: dict[tuple[str, str], PermissionRecord] = {}
        self._audit: list[AuditRecord] = []
        default_profile = self._tts.selected_profile.name
        api_ready = bool(
            os.getenv("OPENAI_API_KEY", "").strip()
            and importlib.util.find_spec("openai")
        )
        self._settings = DesktopSettings(
            voice_profile=default_profile,
            api_enabled=api_ready,
            offline_only=not api_ready,
        )
        self._seed_permissions()

    def _seed_permissions(self) -> None:
        capabilities = (
            ("conversation", RiskLevel.CONVERSATION),
            ("private_memory", RiskLevel.SENSITIVE),
            ("open_app", RiskLevel.HARMLESS_LOCAL),
            ("send_message", RiskLevel.EXTERNAL_COMMUNICATION),
            ("files", RiskLevel.SENSITIVE),
            ("security_changes", RiskLevel.FINANCIAL_OR_SECURITY),
        )
        for role in ("owner", "trusted", "guest"):
            for capability, risk in capabilities:
                allowed = role == "owner" or (
                    role == "trusted" and risk <= RiskLevel.HARMLESS_LOCAL
                ) or (role == "guest" and risk == RiskLevel.CONVERSATION)
                self._permissions[(role, capability)] = PermissionRecord(
                    role, capability, risk, allowed
                )

    def _record(self, event: str, result: str, **details: Any) -> None:
        clean_details = redact_mapping(details)
        self._audit.append(
            AuditRecord(utc_now(), event, result, clean_details if isinstance(clean_details, dict) else {})
        )

    def status(self) -> DesktopStatus:
        with self._lock:
            return DesktopStatus(
                self._state,
                self._message,
                self._settings.offline_only,
                bool(os.getenv("OPENAI_API_KEY", "").strip() and importlib.util.find_spec("openai")),
                self._emergency.latched,
            )

    def toggle_listening(self) -> DesktopStatus:
        with self._lock:
            if self._emergency.latched:
                raise RuntimeError("Reset the emergency stop before listening")
            if self._state == AssistantState.LISTENING:
                self._state = AssistantState.STANDBY
                self._message = "Listening paused"
            else:
                self._state = AssistantState.LISTENING
                self._message = "Listening for a command"
            self._record("listening", "updated", state=self._state.value)
            return self.status()

    def submit_text(self, text: str) -> ConversationTurn:
        clean = str(text).strip()
        if not clean:
            raise ValueError("Enter a message first")
        with self._lock:
            if self._emergency.latched:
                raise RuntimeError("Reset the emergency stop before sending a command")
            self._state = AssistantState.UNDERSTANDING
            self._message = "Understanding your request"
            self._conversation.append(ConversationTurn("You", redact_text(clean)))
            step = LiveStep(uuid4().hex, "Understand text request", "running")
            self._steps.append(step)

        try:
            if self._handler is None:
                reply = "Text received. The desktop shell is ready for the ASHER runtime adapter."
            else:
                # The UI adapter never forwards raw credential-like text to a
                # model/tool handler; local rendering still keeps the request
                # understandable through the redaction marker.
                reply = str(self._handler(redact_text(clean))).strip() or "Request completed."
            response = ConversationTurn("Asher", redact_text(reply))
            with self._lock:
                self._conversation.append(response)
                for index, current in enumerate(self._steps):
                    if current.step_id == step.step_id:
                        self._steps[index] = replace(step, status="complete", detail="Request processed")
                        break
                self._state = AssistantState.COMPLETE
                self._message = "Request complete"
                self._record("text_request", "complete", request=redact_text(clean))
            return response
        except Exception as error:
            with self._lock:
                for index, current in enumerate(self._steps):
                    if current.step_id == step.step_id:
                        self._steps[index] = replace(step, status="error", detail=str(error))
                        break
                self._state = AssistantState.ERROR
                self._message = "Request failed"
                self._record("text_request", "error", error=str(error))
            raise

    def conversation(self) -> tuple[ConversationTurn, ...]:
        with self._lock:
            return tuple(self._conversation)

    def live_steps(self) -> tuple[LiveStep, ...]:
        with self._lock:
            return tuple(self._steps)

    def pending_action(self) -> PendingAction | None:
        with self._lock:
            if self._pending is not None and self._pending.expires_at <= utc_now():
                expired = self._pending
                self._pending = None
                self._state = AssistantState.STANDBY
                self._message = "The confirmation expired"
                self._record("confirmation", "expired", confirmation_id=expired.confirmation_id)
            return self._pending

    def stage_confirmation(self, action: PendingAction) -> None:
        if action.expires_at <= utc_now():
            raise ValueError("Confirmation has already expired")
        with self._lock:
            self._pending = action
            self._state = AssistantState.AWAITING_CONFIRMATION
            self._message = "Review the pending action"
            self._record("confirmation", "pending", action=action.action, target=action.target)

    def approve_pending(self) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            action = self._pending
            if action.expires_at <= utc_now():
                self._pending = None
                self._state = AssistantState.STANDBY
                self._message = "The confirmation expired"
                self._record("confirmation", "expired", confirmation_id=action.confirmation_id)
                return False
        if action.risk >= RiskLevel.SENSITIVE:
            authenticated = self._strong_auth is not None and self._strong_auth(action)
            if not authenticated:
                with self._lock:
                    self._message = "Device authentication is required for this action"
                    self._record(
                        "confirmation", "denied", reason="strong_auth_required", risk=action.risk.name
                    )
                return False
        with self._lock:
            if self._pending is None or self._pending.confirmation_id != action.confirmation_id:
                return False
            self._pending = None
            self._state = AssistantState.COMPLETE
            self._message = "Action approved in the local UI"
            self._record(
                "confirmation", "approved", confirmation_id=action.confirmation_id
            )
            return True

    def reject_pending(self) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            action = self._pending
            self._pending = None
            self._state = AssistantState.STANDBY
            self._message = "Action rejected"
            self._record(
                "confirmation", "rejected", confirmation_id=action.confirmation_id
            )
            return True

    def list_memories(self) -> tuple[MemoryRecord, ...]:
        with self._lock:
            return tuple(sorted(self._memories.values(), key=lambda item: item.key.casefold()))

    def create_memory(
        self, key: str, value: str, memory_type: str = "semantic", sensitivity: str = "private", consented: bool = False
    ) -> MemoryRecord:
        clean_key, clean_value = key.strip(), value.strip()
        if not clean_key or not clean_value:
            raise ValueError("Memory key and value are required")
        if memory_type not in SUPPORTED_MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        if sensitivity not in SUPPORTED_SENSITIVITIES:
            raise ValueError(f"Unsupported sensitivity: {sensitivity}")
        sensitive_key = clean_key.casefold().replace("-", "_").replace(" ", "_")
        prohibited_key = sensitive_key in {
            "password",
            "passcode",
            "pin",
            "api_key",
            "secret",
            "token",
            "credential",
        }
        if prohibited_key or contains_prohibited_secret(clean_key) or contains_prohibited_secret(clean_value):
            raise ValueError("Credential-like values are never stored as personal memory")
        if sensitivity == "sensitive" and not consented:
            raise PermissionError("Sensitive memory requires explicit consent")
        with self._lock:
            if any(item.key.casefold() == clean_key.casefold() for item in self._memories.values()):
                raise ValueError("A memory with that key already exists")
            record = MemoryRecord(
                uuid4().hex, clean_key, clean_value, memory_type, sensitivity
            )
            self._memories[record.memory_id] = record
            self._record("memory_create", "complete", memory_id=record.memory_id)
            return record

    def update_memory(
        self, memory_id: str, value: str, memory_type: str = "semantic", sensitivity: str = "private", consented: bool = False
    ) -> MemoryRecord:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("Memory value is required")
        if memory_type not in SUPPORTED_MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        if sensitivity not in SUPPORTED_SENSITIVITIES:
            raise ValueError(f"Unsupported sensitivity: {sensitivity}")
        if sensitivity == "sensitive" and not consented:
            raise PermissionError("Sensitive memory requires explicit consent")
        if contains_prohibited_secret(clean_value):
            raise ValueError("Credential-like values are never stored as personal memory")
        with self._lock:
            current = self._memories.get(memory_id)
            if current is None:
                raise KeyError("Memory was not found")
            updated = replace(
                current,
                value=clean_value,
                memory_type=memory_type,
                sensitivity=sensitivity,
                updated_at=utc_now(),
            )
            self._memories[memory_id] = updated
            self._record("memory_update", "complete", memory_id=memory_id)
            return updated

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            removed = self._memories.pop(memory_id, None)
            if removed is None:
                return False
            self._record("memory_delete", "complete", memory_id=memory_id)
            return True

    def list_users(self) -> tuple[UserRecord, ...]:
        with self._lock:
            return tuple(self._users.values())

    def enroll_user(self, display_name: str, role: str) -> UserRecord:
        name = display_name.strip()
        if not name:
            raise ValueError("Display name is required")
        if role not in {"owner", "trusted", "guest"}:
            raise ValueError("Unknown user role")
        with self._lock:
            if role == "owner" and any(item.role == "owner" for item in self._users.values()):
                raise ValueError("An owner is already enrolled")
            record = UserRecord(uuid4().hex, name, role, "collecting_samples", 0)
            self._users[record.user_id] = record
            self._record("user_enroll", "started", user_id=record.user_id, role=role)
            return record

    def capture_voice_sample(self, user_id: str) -> UserRecord:
        if self._voiceguard is None:
            raise ControllerUnavailable("VoiceGuard recording service is not connected")
        with self._lock:
            current = self._users.get(user_id)
        if current is None:
            raise KeyError("User was not found")
        count = int(self._voiceguard.capture_sample(user_id))
        updated = replace(current, samples=count, enrollment_status="collecting_samples")
        with self._lock:
            self._users[user_id] = updated
            self._record("voice_sample", "captured", user_id=user_id, samples=count)
        return updated

    def train_voiceguard(self, user_id: str) -> UserRecord:
        if self._voiceguard is None:
            raise ControllerUnavailable("VoiceGuard training service is not connected")
        with self._lock:
            current = self._users.get(user_id)
        if current is None:
            raise KeyError("User was not found")
        metadata = self._voiceguard.train(user_id)
        updated = replace(current, enrollment_status="trained")
        with self._lock:
            self._users[user_id] = updated
            self._record("voiceguard_train", "complete", user_id=user_id, metadata=metadata)
        return updated

    def revoke_user(self, user_id: str) -> bool:
        if user_id == "owner":
            raise ValueError("The built-in owner cannot be removed")
        with self._lock:
            if user_id not in self._users:
                return False
        if self._voiceguard is not None:
            self._voiceguard.revoke(user_id)
        with self._lock:
            self._users.pop(user_id, None)
            self._record("user_revoke", "complete", user_id=user_id)
        return True

    def list_permissions(self) -> tuple[PermissionRecord, ...]:
        with self._lock:
            return tuple(self._permissions.values())

    def set_permission(
        self,
        role: str,
        capability: str,
        allowed: bool,
        actor_id: str | None = None,
    ) -> PermissionRecord:
        key = (role, capability)
        with self._lock:
            current = self._permissions.get(key)
            if current is None:
                raise KeyError("Permission was not found")
            if role == "guest" and current.risk > RiskLevel.CONVERSATION and allowed:
                raise ValueError("Guests cannot receive private or action permissions")
            updated = replace(current, allowed=bool(allowed))
            self._permissions[key] = updated
            self._record(
                "permission_change",
                "complete",
                role=role,
                capability=capability,
                allowed=allowed,
            )
            return updated

    def audit_records(self) -> tuple[AuditRecord, ...]:
        with self._lock:
            return tuple(reversed(self._audit))

    def settings(self) -> DesktopSettings:
        with self._lock:
            return self._settings

    def voice_profiles(self) -> tuple[tuple[str, str], ...]:
        return tuple((profile.name, profile.label) for profile in self._tts.registry.all())

    def update_settings(self, **changes: Any) -> DesktopSettings:
        allowed = {
            "voice_profile",
            "speech_speed",
            "speech_style",
            "offline_only",
            "api_enabled",
            "microphone_index",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown setting: {sorted(unknown)[0]}")
        with self._lock:
            updated = replace(self._settings, **changes)
            selected = self._tts.registry.get(updated.voice_profile)
            if updated.offline_only and selected.provider != "sapi":
                raise ValueError("Select an offline voice while offline-only mode is enabled")
            api_key_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
            api_ready = api_key_configured and bool(importlib.util.find_spec("openai"))
            if selected.provider == "openai" and (not updated.api_enabled or not api_ready):
                raise ValueError("An API key and API-enabled mode are required for an online voice")
            if updated.api_enabled and not api_ready:
                raise ValueError("OpenAI API key and optional package are not configured")
            if not 0.5 <= updated.speech_speed <= 2.0:
                raise ValueError("Speech speed must be between 0.5 and 2.0")
            self._tts.set_profile(updated.voice_profile)
            self._settings = updated
            self._record("settings_change", "complete", fields=sorted(changes))
            return updated

    def run_diagnostics(self) -> DiagnosticReport:
        settings = self.settings()
        checks = {
            "Desktop UI": "available" if importlib.util.find_spec("PySide6") else "PySide6 not installed",
            "Microphone index": "automatic" if settings.microphone_index is None else str(settings.microphone_index),
            "Speech recognition": "available" if importlib.util.find_spec("speech_recognition") else "not installed",
            "Local transcription": "available" if importlib.util.find_spec("whisper") else "not installed",
            "Offline TTS": (
                "configured"
                if os.name == "nt" and "sapi" in self._tts.provider_names()
                else "Windows SAPI unavailable on this platform"
            ),
            "OpenAI API": (
                "configured"
                if os.getenv("OPENAI_API_KEY", "").strip()
                and importlib.util.find_spec("openai")
                else "not configured (key or package missing)"
            ),
        }
        healthy = checks["Offline TTS"] == "configured"
        summary = "Core desktop audio is configured" if healthy else "Audio setup needs attention"
        with self._lock:
            self._record("diagnostics", "complete", healthy=healthy)
        return DiagnosticReport(checks, summary, healthy)

    def emergency_stop(self) -> DesktopStatus:
        cancelled = self._emergency.trigger("Emergency stop activated from local UI")
        self._tts.stop()
        with self._lock:
            self._state = AssistantState.STOPPED
            self._message = "Emergency stop is active"
            self._pending = None
            self._steps = [
                replace(step, status="cancelled", detail="Emergency stop")
                if step.status in {"queued", "running"}
                else step
                for step in self._steps
            ]
            self._record("emergency_stop", "activated", cancelled_plans=cancelled)
            return self.status()

    def reset_emergency_stop(self) -> DesktopStatus:
        if not self._emergency.reset(local_ui_confirmed=True):
            raise RuntimeError("Emergency stop could not be reset")
        with self._lock:
            self._state = AssistantState.STANDBY
            self._message = "Emergency stop reset locally"
            self._record("emergency_stop", "reset")
            return self.status()


__all__ = [
    "AuditRecord",
    "AsherUIController",
    "ControllerUnavailable",
    "ConversationTurn",
    "DesktopController",
    "DesktopControllerProtocol",
    "DesktopSettings",
    "DesktopStatus",
    "DiagnosticReport",
    "LiveStep",
    "MemoryRecord",
    "PendingAction",
    "PermissionRecord",
    "UserRecord",
    "UIControllerProtocol",
    "VoiceGuardAdapter",
    "SUPPORTED_MEMORY_TYPES",
    "SUPPORTED_SENSITIVITIES",
]

# Discoverable names for composition roots that prefer a UI-specific prefix.
UIControllerProtocol = DesktopControllerProtocol
AsherUIController = DesktopController
