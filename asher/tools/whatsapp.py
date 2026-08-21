"""Contact-aware WhatsApp preparation and explicitly gated sending adapter."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


class WhatsAppAdapter(Protocol):
    def prepare(self, contact: str) -> bool: ...
    def verify_target(self, contact: str) -> bool: ...
    def send(self, contact: str, message: str, cancellation: Any) -> bool: ...
    def verify(self, contact: str, message: str) -> bool: ...


class DryRunWhatsAppAdapter:
    def prepare(self, contact: str) -> bool:
        return bool(contact.strip())

    def verify_target(self, contact: str) -> bool:
        return bool(contact.strip())

    def send(self, contact: str, message: str, cancellation: Any) -> bool:
        cancellation.raise_if_cancelled()
        return True

    def verify(self, contact: str, message: str) -> bool:
        return bool(contact.strip() and message.strip())


class WindowsWhatsAppAdapter:
    """Best-effort UIA adapter; live sending is disabled unless explicitly enabled."""

    def prepare(self, contact: str) -> bool:
        try:
            from pywinauto import Desktop, keyboard
            from actions.app_launcher import open_app

            windows = [window for window in Desktop(backend="uia").windows() if "whatsapp" in window.window_text().casefold() and window.is_visible()]
            if not windows:
                open_app("WhatsApp")
                for _ in range(12):
                    time.sleep(1)
                    windows = [window for window in Desktop(backend="uia").windows() if "whatsapp" in window.window_text().casefold() and window.is_visible()]
                    if windows:
                        break
            if not windows:
                return False
            window = windows[0]
            window.set_focus()
            keyboard.send_keys("^f")
            time.sleep(0.4)
            keyboard.send_keys("^a{BACKSPACE}")
            keyboard.send_keys(contact, with_spaces=True, pause=0.04)
            return True
        except Exception:
            return False

    def verify_target(self, contact: str) -> bool:
        # Searching is not proof that the exact result was selected. Until a
        # UIA observer can read and compare the active chat header, live send
        # must fail before any message text is pasted or Enter is pressed.
        return False

    def send(self, contact: str, message: str, cancellation: Any) -> bool:
        if os.getenv("ASHER_ENABLE_LIVE_WHATSAPP", "false").casefold() not in {"1", "true", "yes", "on"}:
            return False
        try:
            import pyautogui
            import pyperclip

            cancellation.raise_if_cancelled()
            original_clipboard = pyperclip.paste()
            try:
                pyperclip.copy(message)
                pyautogui.hotkey("ctrl", "v")
                cancellation.raise_if_cancelled()
                pyautogui.press("enter")
            finally:
                pyperclip.copy(original_clipboard)
            return True
        except Exception:
            return False

    def verify(self, contact: str, message: str) -> bool:
        # A production deployment should replace this with a UIA message-row
        # verifier. Refuse to claim delivery when no observer is available.
        return False


class WhatsAppTools:
    def __init__(self, adapter: WhatsAppAdapter | None = None, contact_resolver: Any | None = None) -> None:
        self.adapter = adapter or WindowsWhatsAppAdapter()
        self.contact_resolver = contact_resolver

    def _resolve(self, raw: str) -> tuple[str | None, tuple[str, ...]]:
        if self.contact_resolver is None:
            return (raw.strip() or None, ())
        result = self.contact_resolver.resolve(raw)
        if getattr(result, "ambiguous", False):
            return None, tuple(getattr(result, "alternatives", ()))
        canonical = getattr(result, "canonical", None)
        if canonical:
            return canonical, ()
        return None, ()

    def _target_is_verified(self, contact: str) -> bool:
        verifier = getattr(self.adapter, "verify_target", None)
        if not callable(verifier):
            return False
        try:
            return bool(verifier(contact))
        except Exception:
            return False

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                name="whatsapp.prepare", description="Resolve a contact and prepare a WhatsApp chat without sending.",
                input_schema={"type":"object", "properties":{"contact":{"type":"string", "minLength":1, "maxLength":200}}, "required":["contact"], "additionalProperties":False},
                policy=ToolPolicy("contacts", RiskLevel.HARMLESS_LOCAL), timeout_seconds=20,
                handler=self.prepare, preview=lambda args: (args["contact"], f"Open and prepare the WhatsApp chat for {args['contact']}", {"contact": args["contact"]}),
            ),
            ToolDefinition(
                name="whatsapp.send", description="Send an already-previewed WhatsApp message after non-voice approval and verify delivery.",
                input_schema={"type":"object", "properties":{"contact":{"type":"string", "minLength":1, "maxLength":200}, "message":{"type":"string", "minLength":1, "maxLength":4000}}, "required":["contact","message"], "additionalProperties":False},
                policy=ToolPolicy("external_communication", RiskLevel.EXTERNAL_COMMUNICATION), timeout_seconds=30,
                handler=self.send, preview=lambda args: (args["contact"], f"Send a WhatsApp message to {args['contact']}", {"contact": args["contact"], "message": args["message"]}),
            ),
        )

    def prepare(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        contact, alternatives = self._resolve(arguments["contact"])
        if alternatives:
            return _failure(context, "ambiguous_contact", f"Which contact did you mean: {' or '.join(alternatives)}?")
        if not contact:
            return _failure(context, "contact_not_found", "I could not resolve that contact safely.")
        if context.dry_run:
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Dry run prepared the WhatsApp chat for {contact}.", (Evidence("dry_run_chat", "No WhatsApp window was changed", {"contact": contact}),), dry_run=True)
        if not self.adapter.prepare(contact):
            return _failure(context, "chat_unverified", f"Could not prepare the exact WhatsApp chat for {contact}.")
        if not self._target_is_verified(contact):
            return _failure(
                context,
                "recipient_unverified",
                "WhatsApp search completed, but the exact recipient could not be observed. Nothing was sent.",
            )
        return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Prepared the WhatsApp chat for {contact}.", (Evidence("chat_prepared", "The adapter reported the target chat ready", {"contact": contact}),))

    def send(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        contact, alternatives = self._resolve(arguments["contact"])
        if alternatives:
            return _failure(context, "ambiguous_contact", f"Which contact did you mean: {' or '.join(alternatives)}?")
        if not contact:
            return _failure(context, "contact_not_found", "I could not resolve that contact safely.")
        message = arguments["message"].strip()
        if context.dry_run:
            if not DryRunWhatsAppAdapter().verify(contact, message):
                return _failure(context, "dry_run_invalid", "The dry-run message was not valid.")
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Dry run verified a message to {contact}; nothing was sent.", (Evidence("dry_run_delivery", "Simulated approved delivery", {"contact": contact, "message_length": len(message), "simulated": True}),), dry_run=True)
        # This pre-send observation is a hard safety boundary. Post-send
        # verification alone is too late: a wrong active chat may already have
        # received the message even if the tool subsequently reports failure.
        if not self._target_is_verified(contact):
            return _failure(
                context,
                "recipient_unverified",
                "The exact WhatsApp recipient could not be verified before sending. Nothing was sent.",
            )
        if not self.adapter.send(contact, message, context.cancellation):
            return _failure(context, "send_failed", "The message was not sent or the adapter refused live sending.")
        if not self.adapter.verify(contact, message):
            return _failure(context, "delivery_unverified", "The send attempt completed, but delivery could not be observed.")
        return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Verified delivery to {contact}.", (Evidence("message_observed", "The adapter observed the sent message", {"contact": contact, "message_length": len(message)}),))


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"], success=False, status="failed", message=message, error_code=code, started_at=now, completed_at=now)
