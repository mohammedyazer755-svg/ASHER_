from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from asher.core.cancellation import CancellationToken, EmergencyStop
from asher.memory.store import MemoryStore
from asher.security.audit import AuditLog
from asher.security.confirmations import ConfirmationStore
from asher.security.policy import DecisionKind, PolicyEngine, ToolPolicy
from asher.security.sessions import SessionManager
from asher.security.users import UserStore, guest_actor
from asher.storage import Database
from asher.types import AuthMethod, RiskLevel, Role, utc_now


class SecurityTests(unittest.TestCase):
    def test_roles_sessions_and_confirmation_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "asher.db")
            users = UserStore(db)
            owner = users.ensure_owner("Test Owner")
            guest = guest_actor()
            manager = SessionManager(ttl_minutes=1)
            owner_session = manager.create(owner, AuthMethod.LOCAL_UI)
            guest_session = manager.create(guest, AuthMethod.NONE)
            self.assertIsNotNone(manager.get(owner_session.session_id))
            self.assertIsNotNone(manager.get(guest_session.session_id))
            self.assertIsNone(manager.get(owner_session.session_id, now=owner_session.expires_at + timedelta(seconds=1)))

            confirmations = ConfirmationStore(ttl_seconds=90)
            pending = confirmations.create(
                owner_session,
                tool_name="whatsapp.send",
                target="fixture contact",
                effect="send one message",
                preview={"message_length": 5},
                risk=RiskLevel.EXTERNAL_COMMUNICATION,
                arguments={"contact": "fixture contact", "message": "hello"},
            )
            with self.assertRaises(PermissionError):
                confirmations.approve(pending.confirmation_id, owner_session, AuthMethod.VOICE)
            approved = confirmations.approve(pending.confirmation_id, owner_session, AuthMethod.LOCAL_UI)
            self.assertTrue(approved.approved)
            self.assertEqual(approved.actor_id, owner.user_id)

    def test_confirmation_rejects_same_length_payload_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "asher.db")
            owner = UserStore(db).ensure_owner("Test Owner")
            session = SessionManager().create(owner, AuthMethod.LOCAL_UI)
            confirmations = ConfirmationStore()
            original = {"contact": "Fixture", "message": "hello"}
            preview = {"contact": "Fixture", "message": "hello"}
            pending = confirmations.create(
                session,
                tool_name="whatsapp.send",
                target="Fixture",
                effect="Send a message",
                preview=preview,
                risk=RiskLevel.EXTERNAL_COMMUNICATION,
                arguments=original,
            )
            approved = confirmations.approve(
                pending.confirmation_id,
                session,
                AuthMethod.LOCAL_UI,
            )
            with self.assertRaises(PermissionError):
                confirmations.consume(
                    approved,
                    session,
                    tool_name="whatsapp.send",
                    target="Fixture",
                    effect="Send a message",
                    preview=preview,
                    risk=RiskLevel.EXTERNAL_COMMUNICATION,
                    arguments={"contact": "Fixture", "message": "world"},
                )
            consumed = confirmations.consume(
                approved,
                session,
                tool_name="whatsapp.send",
                target="Fixture",
                effect="Send a message",
                preview=preview,
                risk=RiskLevel.EXTERNAL_COMMUNICATION,
                arguments=original,
            )
            self.assertEqual(consumed.argument_digest, approved.argument_digest)

    def test_expired_or_revoked_controller_session_cannot_run_tools(self) -> None:
        from asher.agent.controller import CompanionController
        from asher.brain.deterministic import ContactResolver
        from asher.config import AsherConfig

        with tempfile.TemporaryDirectory() as directory:
            controller = CompanionController(
                AsherConfig.load(directory),
                contact_resolver=ContactResolver(("Demo",), {"demo": "Demo"}),
            )
            session = controller.create_owner_session()
            controller.sessions.invalidate(session.session_id)
            reply = controller.handle_text("open chrome", session)
            self.assertIn("session", reply.text.casefold())
            trusted = controller.users.create("Trusted Demo", Role.TRUSTED, {"open_app"})
            fresh = controller.create_voice_session(trusted)
            controller.users.revoke(trusted.user_id)
            denied = controller.handle_text("open chrome", fresh)
            self.assertIn("session", denied.text.casefold())
            with self.assertRaises(TypeError):
                controller.create_voice_session()  # type: ignore[call-arg]

    def test_policy_denies_guest_and_suspicious_sessions(self) -> None:
        guest = guest_actor()
        guest_session = SessionManager().create(guest, AuthMethod.NONE)
        policy = PolicyEngine()
        self.assertEqual(
            policy.evaluate(ToolPolicy("private_files", RiskLevel.SENSITIVE), guest_session).kind,
            DecisionKind.DENY,
        )
        owner = UserStore(Database(Path(tempfile.mkdtemp()) / "x.db")).ensure_owner("Owner")
        suspicious = SessionManager().create(owner, AuthMethod.VOICE, suspicious=True)
        self.assertEqual(
            policy.evaluate(ToolPolicy("open_app", RiskLevel.HARMLESS_LOCAL), suspicious).kind,
            DecisionKind.REQUIRE_STRONG_AUTH,
        )

    def test_memory_is_typed_private_persistent_and_guest_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "asher.db")
            owner = UserStore(db).ensure_owner("Owner")
            owner_session = SessionManager().create(owner, AuthMethod.LOCAL_UI)
            store = MemoryStore(db)
            record = store.put(
                owner,
                owner_id=owner.user_id,
                memory_type="project_goal",
                key="demo goal",
                value="Build a safe companion",
                source="test",
                confidence=0.9,
                confirmed=True,
            )
            self.assertEqual(store.get(owner, record.memory_id).value, "Build a safe companion")
            with self.assertRaises(ValueError):
                store.put(
                    owner,
                    owner_id=owner.user_id,
                    memory_type="semantic",
                    key="api key",
                    value="secret-token",
                    source="test",
                    confirmed=True,
                )
            self.assertEqual(store.list(guest_actor(), owner_id=owner.user_id), [])
            with self.assertRaises(PermissionError):
                store.put(
                    guest_actor(),
                    owner_id=owner.user_id,
                    memory_type="semantic",
                    key="x",
                    value="y",
                    source="guest",
                    confirmed=True,
                )
            updated = store.update(
                owner,
                record.memory_id,
                memory_type="project_goal",
                value="Build and verify a safe companion",
                source="test",
                confirmed=True,
            )
            self.assertEqual(updated.memory_id, record.memory_id)
            self.assertTrue(store.delete(owner, record.memory_id, confirmed=True))
            self.assertIsNone(store.get(owner, record.memory_id))

    def test_audit_redacts_payload_keys_and_emergency_stop_is_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "audit.jsonl"
            audit = AuditLog(log_path)
            audit.append(
                "test_event",
                actor_id="owner",
                details={"body": "private message", "value": "private memory", "api_key": "secret"},
            )
            raw = log_path.read_text(encoding="utf-8")
            self.assertNotIn("private message", raw)
            self.assertNotIn("private memory", raw)
            self.assertNotIn("secret", raw)
            self.assertEqual(audit.read_recent()[0]["details"]["body"], "[CONTENT OMITTED]")

        stop = EmergencyStop()
        token = CancellationToken()
        stop.register(token)
        self.assertEqual(stop.trigger("test stop"), 1)
        self.assertTrue(token.cancelled)
        self.assertFalse(stop.reset(local_ui_confirmed=False))
        self.assertTrue(stop.reset(local_ui_confirmed=True))


if __name__ == "__main__":
    unittest.main()
