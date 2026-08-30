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
from asher.types import Actor, AuthMethod, RiskLevel, Role, utc_now


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

    def test_memory_disable_and_retention_are_persistent_and_fail_closed(self) -> None:
        from datetime import UTC, datetime, timedelta

        from asher.memory.retrieval import MemoryRetriever

        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "asher.db")
            store = MemoryStore(database)
            owner = Actor("owner", "Owner", Role.OWNER)
            store.configure(enabled=True, retention_days=30)
            before = datetime.now(UTC)
            record = store.put(
                owner,
                owner_id=owner.user_id,
                memory_type="semantic",
                key="retained project",
                value="voice evaluation",
                source="test",
                confirmed=True,
            )
            self.assertIsNotNone(record.expires_at)
            self.assertGreaterEqual(record.expires_at, before + timedelta(days=29))
            self.assertEqual(MemoryStore(database).retention_days, 30)

            store.configure(enabled=False, retention_days=30)
            self.assertFalse(MemoryStore(database).enabled)
            self.assertEqual(
                MemoryRetriever(store).retrieve(
                    owner,
                    owner_id=owner.user_id,
                    query="project",
                ),
                [],
            )
            with self.assertRaises(PermissionError):
                store.put(
                    owner,
                    owner_id=owner.user_id,
                    memory_type="semantic",
                    key="blocked",
                    value="blocked",
                    source="test",
                    confirmed=True,
                )
            # Disabling capture/retrieval must not hold existing records
            # hostage: the owner can still inspect, export, or delete them.
            self.assertEqual(store.list(owner, owner_id=owner.user_id)[0].memory_id, record.memory_id)
            self.assertTrue(store.delete(owner, record.memory_id, confirmed=True))

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

    def test_financial_or_security_policy_matrix(self) -> None:
        from asher.types import Confirmation

        policy_engine = PolicyEngine()
        financial_policy = ToolPolicy("bank.transfer", RiskLevel.FINANCIAL_OR_SECURITY)
        owner = Actor("owner-1", "Owner", Role.OWNER)
        trusted = Actor("trusted-1", "Trusted", Role.TRUSTED, permissions=("bank.transfer",))
        guest = Actor("guest-1", "Guest", Role.GUEST)

        manager = SessionManager(ttl_minutes=5)
        owner_session = manager.create(owner, AuthMethod.LOCAL_UI)
        trusted_session = manager.create(trusted, AuthMethod.LOCAL_UI)
        guest_session = manager.create(guest, AuthMethod.NONE)

        # 1. Guest is denied
        d_guest = policy_engine.evaluate(financial_policy, guest_session)
        self.assertEqual(d_guest.kind, DecisionKind.DENY)

        # 2. Trusted user is denied even with permission
        d_trusted = policy_engine.evaluate(financial_policy, trusted_session)
        self.assertEqual(d_trusted.kind, DecisionKind.DENY)

        # 3. Owner without confirmation requires strong auth
        d_owner_unconfirmed = policy_engine.evaluate(financial_policy, owner_session)
        self.assertEqual(d_owner_unconfirmed.kind, DecisionKind.REQUIRE_STRONG_AUTH)

        # 4. Owner with unapproved confirmation requires strong auth
        expires = utc_now() + timedelta(minutes=2)
        unapproved = Confirmation(
            confirmation_id="c1",
            tool_name="bank.transfer",
            target="target",
            effect="effect",
            preview={},
            risk=RiskLevel.FINANCIAL_OR_SECURITY,
            expires_at=expires,
            session_id=owner_session.session_id,
            actor_id=owner.user_id,
            approved=False,
        )
        d_owner_unapproved = policy_engine.evaluate(financial_policy, owner_session, unapproved)
        self.assertEqual(d_owner_unapproved.kind, DecisionKind.REQUIRE_STRONG_AUTH)

        # 5. Owner with VOICE approval is denied
        voice_approved = Confirmation(
            confirmation_id="c2",
            tool_name="bank.transfer",
            target="target",
            effect="effect",
            preview={},
            risk=RiskLevel.FINANCIAL_OR_SECURITY,
            expires_at=expires,
            session_id=owner_session.session_id,
            actor_id=owner.user_id,
            approved=True,
            method=AuthMethod.VOICE,
        )
        d_owner_voice = policy_engine.evaluate(financial_policy, owner_session, voice_approved)
        self.assertEqual(d_owner_voice.kind, DecisionKind.DENY)

        # 6. Owner with standard LOCAL_UI approval still requires DEVICE_CREDENTIAL
        local_approved = Confirmation(
            confirmation_id="c3",
            tool_name="bank.transfer",
            target="target",
            effect="effect",
            preview={},
            risk=RiskLevel.FINANCIAL_OR_SECURITY,
            expires_at=expires,
            session_id=owner_session.session_id,
            actor_id=owner.user_id,
            approved=True,
            method=AuthMethod.LOCAL_UI,
        )
        d_owner_local = policy_engine.evaluate(financial_policy, owner_session, local_approved)
        self.assertEqual(d_owner_local.kind, DecisionKind.REQUIRE_STRONG_AUTH)

        # 7. Owner with DEVICE_CREDENTIAL approval is allowed
        device_approved = Confirmation(
            confirmation_id="c4",
            tool_name="bank.transfer",
            target="target",
            effect="effect",
            preview={},
            risk=RiskLevel.FINANCIAL_OR_SECURITY,
            expires_at=expires,
            session_id=owner_session.session_id,
            actor_id=owner.user_id,
            approved=True,
            method=AuthMethod.DEVICE_CREDENTIAL,
        )
        d_owner_device = policy_engine.evaluate(financial_policy, owner_session, device_approved)
        self.assertEqual(d_owner_device.kind, DecisionKind.ALLOW)

    def test_trusted_memory_capability_aliases(self) -> None:
        policy_engine = PolicyEngine()
        read_policy = ToolPolicy("private_memory", RiskLevel.HARMLESS_LOCAL)
        write_policy = ToolPolicy("private_memory_write", RiskLevel.SENSITIVE)

        trusted_read = Actor("t1", "Trusted Read", Role.TRUSTED, permissions=("memory.read",))
        session_read = SessionManager().create(trusted_read, AuthMethod.LOCAL_UI)
        d_read = policy_engine.evaluate(read_policy, session_read)
        self.assertEqual(d_read.kind, DecisionKind.ALLOW)

        trusted_write = Actor("t2", "Trusted Write", Role.TRUSTED, permissions=("private_memory_write",))
        session_write = SessionManager().create(trusted_write, AuthMethod.LOCAL_UI)
        d_write_unconf = policy_engine.evaluate(write_policy, session_write)
        self.assertEqual(d_write_unconf.kind, DecisionKind.REQUIRE_STRONG_AUTH)


if __name__ == "__main__":
    unittest.main()
