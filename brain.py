"""Compatibility facade for the current authenticated ASHER controller.

The historical project had a module-level, stateful brain.py that wrote
conversation history and handled approvals by voice. The active entry point is
now asher.agent.controller; this facade keeps a few old function names without
recreating those unsafe side effects.
"""

from __future__ import annotations

from asher.agent.controller import CompanionController
from asher.config import AsherConfig
from asher.core.redaction import redact_text
from asher.types import AuthMethod

ASHER_BRAIN_VERSION = "1.0.0"

_default_controller: CompanionController | None = None


def get_controller() -> CompanionController:
    global _default_controller
    if _default_controller is None:
        _default_controller = CompanionController(AsherConfig.load())
    return _default_controller


def respond(user_input: str, *, controller: CompanionController | None = None) -> str:
    runtime = controller or get_controller()
    session = runtime.create_owner_session(AuthMethod.LOCAL_UI)
    return redact_text(runtime.handle_text(user_input, session).text)


def process_command(user_input: str, *, controller: CompanionController | None = None) -> str:
    return respond(user_input, controller=controller)


def execute_ai_plan(command: str, *, controller: CompanionController | None = None) -> str:
    # Provider selection and tool execution are owned by the typed controller;
    # this name remains for callers migrating from the legacy planner.
    return respond(command, controller=controller)


def greet() -> str:
    return "Hello. ASHER is ready."


def goodbye() -> str:
    return "Goodbye."


def get_display_name(key: str) -> str:
    return str(key).replace("_", " ").strip().title()


def clean_whatsapp_contact(contact: str) -> str:
    return str(contact).strip()


def should_use_ai_planner(user_input: str) -> bool:
    return bool(str(user_input).strip())


def update_context_memory(new_value: str) -> bool:
    # Memory writes require the explicit Memory UI/tool confirmation path; the
    # compatibility facade never silently stores an arbitrary value.
    return False


__all__ = [
    "ASHER_BRAIN_VERSION",
    "clean_whatsapp_contact",
    "execute_ai_plan",
    "get_controller",
    "get_display_name",
    "goodbye",
    "greet",
    "process_command",
    "respond",
    "should_use_ai_planner",
    "update_context_memory",
]

