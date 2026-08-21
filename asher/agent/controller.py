"""High-level companion controller shared by CLI and desktop UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import threading
from typing import Any

from asher.agent.loop import AgentLoop, ExecutionUpdate
from asher.brain.deterministic import ContactResolver, DeterministicPlanner
from asher.brain.plans import as_execution_plan
from asher.brain.providers import HybridPlanner, OllamaProvider, OpenAIResponsesProvider
from asher.config import AsherConfig
from asher.core.cancellation import EmergencyStop
from asher.core.redaction import contains_prohibited_secret, redact_text
from asher.memory.retrieval import MemoryRetriever
from asher.memory.store import MemoryStore
from asher.memory.working import WorkingMemory
from asher.brain.personality import infer_emotional_context
from asher.security.audit import AuditLog
from asher.security.sessions import SessionManager
from asher.security.strong_auth import StrongAuthenticator, WindowsHelloAuthenticator
from asher.security.users import UserStore, guest_actor
from asher.storage import Database
from asher.tools.catalog import build_registry
from asher.types import Actor, AuthMethod, Role, SessionContext


@dataclass(frozen=True)
class CompanionReply:
    text: str
    updates: tuple[ExecutionUpdate, ...] = ()
    offline: bool = True
    provider: str = "deterministic"
    confirmation_id: str | None = None


class CompanionController:
    def __init__(
        self,
        config: AsherConfig | None = None,
        *,
        database: Database | None = None,
        contact_resolver: ContactResolver | None = None,
        strong_authenticator: StrongAuthenticator | None = None,
    ) -> None:
        self.config = config or AsherConfig.load()
        self.config.runtime.ensure()
        self.database = database or Database(self.config.runtime.database)
        self.memory_store = MemoryStore(self.database)
        self.memory_retriever = MemoryRetriever(self.memory_store)
        self.working_memory = WorkingMemory()
        self.audit = AuditLog(self.config.runtime.audit_log)
        self.users = UserStore(self.database)
        self.owner = self.users.ensure_owner(self.config.owner_name)
        self.sessions = SessionManager(self.config.session_minutes)
        self.resolver = contact_resolver or _load_contact_resolver()
        self.registry = build_registry(self.config, self.database, contact_resolver=self.resolver)
        self._emergency_stop = EmergencyStop()
        # A caller can inject Windows Hello (or another OS-bound verifier),
        # but the secure default is explicit denial rather than treating a
        # boolean CLI/UI flag as proof of device authentication.
        # Windows Hello is the production default. Its adapter fails closed
        # when the optional WinRT binding or a configured device credential is
        # unavailable; tests and deployments may inject another OS-bound
        # authenticator explicitly.
        self.strong_authenticator = strong_authenticator or WindowsHelloAuthenticator()
        self.loop = AgentLoop(self.registry, self.registry.confirmations, self.audit, emergency_stop=self._emergency_stop)
        self.planner = HybridPlanner(
            DeterministicPlanner(self.resolver),
            self.registry,
            openai=OpenAIResponsesProvider(self.config.openai_routine_model, reasoning_model=self.config.openai_reasoning_model) if self.config.openai_enabled else None,
            ollama=OllamaProvider(
                self.config.ollama_url,
                self.config.ollama_model,
                offline=self.config.ollama_local,
            ),
        )
        # Continuation hints are authentication-session scoped. A controller-
        # global "last contact" could otherwise leak a private name from an
        # owner turn into a trusted user's pronoun-based request.
        self._context_lock = threading.RLock()
        self._session_context: dict[str, dict[str, str]] = {}

    def create_owner_session(self, method: AuthMethod = AuthMethod.LOCAL_UI) -> SessionContext:
        return self.sessions.create(self.owner, method)

    @property
    def emergency_stopped(self) -> bool:
        """Whether the process-wide stop latch is active."""

        return self._emergency_stop.latched

    def create_voice_session(self, actor: Actor, *, suspicious: bool = False) -> SessionContext:
        """Create a voice-authenticated session for an explicitly identified actor.

        A missing VoiceGuard decision must become a guest session at the caller;
        silently defaulting an absent actor to the owner would turn an integration
        mistake into an authorization bypass.
        """

        if actor.role is Role.GUEST:
            raise ValueError("Guest voice access must use create_guest_session()")
        return self.sessions.create(actor, AuthMethod.VOICE, suspicious=suspicious)

    def create_guest_session(self) -> SessionContext:
        return self.sessions.create(guest_actor(), AuthMethod.NONE)

    def _resolve_session(self, session: SessionContext) -> SessionContext | None:
        current = self.sessions.get(session.session_id)
        if current is None or current != session:
            return None
        actor = self.users.get(current.actor.user_id)
        if session.actor.role is Role.GUEST:
            return current if current.actor.user_id == "guest" and current.actor.role is Role.GUEST else None
        if actor is None or actor != current.actor:
            return None
        return current

    def _active_session(self, session: SessionContext) -> bool:
        return self._resolve_session(session) is not None

    def handle_text(self, text: str, session: SessionContext) -> CompanionReply:
        text = str(text).strip()
        if not text:
            return CompanionReply("Please tell me what you would like me to do.")
        if contains_prohibited_secret(text) or re.search(
            r"\b(?:password|passcode|pin|api[ _-]?key|api[ _-]?token|credential|credentials)\b",
            text.casefold(),
        ):
            return CompanionReply(
                "I do not store, reveal, or transmit passwords, PINs, API keys, tokens, or credentials.",
                offline=True,
                provider="local-safety",
            )
        lowered = text.casefold().rstrip(".,!?;:")
        if lowered in {"emergency stop", "stop everything", "asher emergency stop"}:
            updates = tuple(self.loop.trigger_emergency_stop())
            return CompanionReply("Emergency stop activated. Any active plan was cancelled.", updates)
        if lowered in {"cancel", "stop", "never mind", "nevermind"}:
            updates = tuple(self.loop.cancel())
            return CompanionReply("Cancelled.", updates)
        active_session = self._resolve_session(session)
        if active_session is None:
            return CompanionReply(
                "Your authenticated session has expired or is no longer valid. Please authenticate again.",
                offline=self.planner.offline,
                provider=self.planner.last_provider,
            )
        session = active_session
        context: dict[str, Any] = {"session": session}
        with self._context_lock:
            continuation = dict(self._session_context.get(session.session_id, {}))
        # Only send continuation context when the wording needs it. This keeps
        # unrelated contact/app names out of remote provider prompts.
        last_contact = continuation.get("last_contact", "")
        last_app = continuation.get("last_app", "")
        if last_contact and any(token in lowered for token in ("him", "her", "them", "same person", "that contact")):
            context["last_contact"] = last_contact
        if last_app and any(token in lowered for token in ("it", "that window", "same app", "close it")):
            context["last_app"] = last_app
        self.working_memory.append(session.session_id, "user", redact_text(text))
        plan = self.planner.plan(text, context=context)
        emotional = infer_emotional_context(text)
        if emotional is not None and not plan.steps and plan.response:
            response = {
                "frustrated": "That sounds frustrating. I can help break it into the next practical step.",
                "tired": "You sound tired, so I’ll keep this brief. What is the smallest useful next step?",
                "excited": "That sounds exciting. I’m ready to help with the next step.",
                "urgent": "I hear the urgency. I’ll start with the fastest safe option.",
            }.get(emotional.cue, plan.response)
            self.working_memory.append(session.session_id, "assistant", redact_text(response))
            return CompanionReply(response, offline=plan.offline, provider=plan.provider)
        if not plan.steps:
            return CompanionReply(plan.response or "I’m not sure how to help with that yet.", offline=plan.offline, provider=plan.provider)
        updates = tuple(self.loop.start(as_execution_plan(plan), session, dry_run=self.config.dry_run))
        self._update_context(updates, session)
        confirmation_id = next((item.confirmation_id for item in updates if item.confirmation_id), None)
        text_reply = next((item.message for item in reversed(updates) if item.message), plan.response or "Working on it.")
        if (
            confirmation_id is None
            and plan.steps
            and plan.steps[0].call.tool_name == "memory.search"
            and session.actor.role in {Role.OWNER, Role.TRUSTED}
        ):
            query = str(plan.steps[0].call.arguments.get("query", ""))
            records = self.memory_retriever.retrieve(
                session.actor,
                owner_id=self.owner.user_id,
                query=query,
                include_sensitive=False,
                limit=3,
            )
            if records:
                text_reply = " ".join(
                    f"I remember {item.record.key}: {redact_text(item.record.value)}."
                    for item in records
                )
            else:
                text_reply = "I do not have a non-sensitive memory matching that yet."
        self.working_memory.append(session.session_id, "assistant", redact_text(text_reply))
        return CompanionReply(text_reply, updates, offline=plan.offline, provider=plan.provider, confirmation_id=confirmation_id)

    def approve(self, confirmation_id: str, session: SessionContext, *, device_authenticated: bool = False) -> CompanionReply:
        active_session = self._resolve_session(session)
        if active_session is None:
            update = ExecutionUpdate(
                "denied",
                "Approval denied because the authenticated session is expired or invalid.",
            )
            return CompanionReply(
                update.message,
                (update,),
                offline=self.planner.offline,
                provider=self.planner.last_provider,
            )
        session = active_session
        pending = self.registry.confirmations.get(confirmation_id)
        if pending is None:
            update = ExecutionUpdate("denied", "That confirmation is missing or expired.")
            return CompanionReply(update.message, (update,), offline=self.planner.offline, provider=self.planner.last_provider)
        if pending.actor_id != session.actor.user_id or pending.session_id != session.session_id:
            update = ExecutionUpdate("denied", "That confirmation belongs to a different authenticated session.")
            return CompanionReply(update.message, (update,), offline=self.planner.offline, provider=self.planner.last_provider)
        if device_authenticated:
            auth = self.strong_authenticator.verify(
                f"Authorize ASHER action: {pending.tool_name} targeting {pending.target}"
            )
            if not auth.verified:
                self.audit.append(
                    "strong_auth_failed",
                    actor_id=session.actor.user_id,
                    session_id=session.session_id,
                    tool_name=pending.tool_name,
                    target=pending.target,
                    outcome="denied",
                )
                update = ExecutionUpdate("denied", "Device authentication was not verified; the action remains pending.")
                return CompanionReply(update.message, (update,), offline=self.planner.offline, provider=self.planner.last_provider)
        method = AuthMethod.DEVICE_CREDENTIAL if device_authenticated else AuthMethod.LOCAL_UI
        updates = tuple(self.loop.approve_and_resume(confirmation_id, session, method))
        return CompanionReply(next((item.message for item in reversed(updates) if item.message), "Approval processed."), updates, offline=self.planner.offline, provider=self.planner.last_provider)

    def reject(self, confirmation_id: str, session: SessionContext) -> CompanionReply:
        active_session = self._resolve_session(session)
        if active_session is None:
            update = ExecutionUpdate(
                "denied",
                "Rejection denied because the authenticated session is expired or invalid.",
            )
            return CompanionReply(update.message, (update,))
        updates = tuple(self.loop.reject(confirmation_id, active_session))
        return CompanionReply("The action was rejected.", updates)

    def emergency_stop(self) -> CompanionReply:
        updates = tuple(self.loop.trigger_emergency_stop())
        return CompanionReply("Emergency stop activated.", updates)

    def reset_emergency_stop(self, *, local_ui_confirmed: bool) -> bool:
        return self._emergency_stop.reset(local_ui_confirmed=local_ui_confirmed)

    def _update_context(
        self,
        updates: tuple[ExecutionUpdate, ...],
        session: SessionContext,
    ) -> None:
        changed: dict[str, str] = {}
        for update in updates:
            if update.result is None:
                continue
            for evidence in update.result.evidence:
                if evidence.kind == "window_observed":
                    value = str(evidence.data.get("window_title", "")).strip()
                    if value:
                        changed["last_app"] = value
                if evidence.kind in {"chat_prepared", "dry_run_chat"}:
                    value = str(evidence.data.get("contact", "")).strip()
                    if value:
                        changed["last_contact"] = value
        if changed:
            with self._context_lock:
                self._session_context.setdefault(session.session_id, {}).update(changed)


def _load_contact_resolver() -> ContactResolver:
    try:
        from asher.voice.vocabulary import DynamicVocabulary
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        vocabulary = DynamicVocabulary(
            contacts_path=project_root / "data" / "voice_contacts.json",
            applications_path=project_root / "data" / "apps.json",
        )
        contacts, aliases = vocabulary.contacts_and_aliases()
        return ContactResolver(tuple(contacts), aliases)
    except Exception:
        return ContactResolver()
