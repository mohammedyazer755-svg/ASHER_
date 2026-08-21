from __future__ import annotations

import time
import tempfile
import unittest
from pathlib import Path

from asher.core.cancellation import CancellationToken
from asher.security.audit import AuditLog
from asher.security.confirmations import ConfirmationStore
from asher.security.policy import PolicyEngine, ToolPolicy
from asher.security.sessions import SessionManager
from asher.tools.files import FileTools
from asher.tools.registry import ToolContext, ToolDefinition, ToolRegistry, successful_result
from asher.tools.whatsapp import WhatsAppTools
from asher.types import Actor, AuthMethod, Evidence, RiskLevel, Role, ToolCall


class ToolSafetyTests(unittest.TestCase):
    def test_file_allow_list_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = FileTools([directory])
            with self.assertRaises(PermissionError):
                tools._resolve(str(Path(directory).parent / "outside.txt"))

    def test_registry_timeout_returns_without_waiting_for_handler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                policy=PolicyEngine(),
                confirmations=ConfirmationStore(),
                audit=AuditLog(Path(directory) / "audit.jsonl"),
            )

            def slow(arguments, context):
                time.sleep(0.25)
                return successful_result(
                    context.metadata["call_id"],
                    context.metadata["tool_name"],
                    "late",
                    (Evidence("late", "late"),),
                )

            registry.register(
                ToolDefinition(
                    "slow",
                    "test timeout",
                    {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                    ToolPolicy("conversation", RiskLevel.CONVERSATION),
                    0.01,
                    slow,
                    lambda _args: ("local", "slow", {}),
                )
            )
            session = SessionManager().create(Actor("guest", "Guest", Role.GUEST), AuthMethod.NONE)
            started = time.perf_counter()
            result = registry.execute(
                ToolCall("slow"),
                ToolContext(session, CancellationToken(), True),
            )
            elapsed = time.perf_counter() - started
            self.assertEqual(result.error_code, "timeout")
            self.assertFalse(result.retryable)
            self.assertLess(elapsed, 0.15)

    def test_tool_result_requires_evidence_for_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(
                policy=PolicyEngine(),
                confirmations=ConfirmationStore(),
                audit=AuditLog(Path(directory) / "audit.jsonl"),
            )

            def no_evidence(arguments, context):
                from asher.types import ToolResult
                return ToolResult(
                    call_id=context.metadata["call_id"],
                    tool_name=context.metadata["tool_name"],
                    success=True,
                    status="verified",
                    message="claimed",
                )

            registry.register(
                ToolDefinition(
                    "claim",
                    "test",
                    {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                    ToolPolicy("conversation", RiskLevel.CONVERSATION),
                    1.0,
                    no_evidence,
                    lambda _args: ("local", "claim", {}),
                )
            )
            session = SessionManager().create(Actor("guest", "Guest", Role.GUEST), AuthMethod.NONE)
            result = registry.execute(ToolCall("claim"), ToolContext(session, CancellationToken(), True))
            self.assertFalse(result.success)
            self.assertEqual(result.error_code, "unverified")

    def test_whatsapp_refuses_before_send_when_recipient_is_not_observed(self) -> None:
        class UnverifiedAdapter:
            def __init__(self) -> None:
                self.send_called = False

            @staticmethod
            def prepare(_contact: str) -> bool:
                return True

            @staticmethod
            def verify_target(_contact: str) -> bool:
                return False

            def send(self, _contact: str, _message: str, _cancellation) -> bool:
                self.send_called = True
                return True

            @staticmethod
            def verify(_contact: str, _message: str) -> bool:
                return True

        adapter = UnverifiedAdapter()
        tools = WhatsAppTools(adapter=adapter)
        session = SessionManager().create(
            Actor("owner", "Owner", Role.OWNER),
            AuthMethod.LOCAL_UI,
        )
        context = ToolContext(
            session,
            CancellationToken(),
            False,
            {"call_id": "message-call", "tool_name": "whatsapp.send"},
        )
        result = tools.send(
            {"contact": "Fixture Contact", "message": "Hello"},
            context,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "recipient_unverified")
        self.assertFalse(adapter.send_called)


if __name__ == "__main__":
    unittest.main()
