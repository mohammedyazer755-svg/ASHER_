"""Shared types for authorization, planning, tools, and observable results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


class Role(str, Enum):
    OWNER = "owner"
    TRUSTED = "trusted"
    GUEST = "guest"


class RiskLevel(IntEnum):
    CONVERSATION = 0
    HARMLESS_LOCAL = 1
    EXTERNAL_COMMUNICATION = 2
    SENSITIVE = 3
    FINANCIAL_OR_SECURITY = 4


class AuthMethod(str, Enum):
    NONE = "none"
    VOICE = "voice"
    LOCAL_UI = "local_ui"
    DEVICE_CREDENTIAL = "device_credential"


@dataclass(frozen=True)
class Actor:
    user_id: str
    display_name: str
    role: Role
    permissions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    actor: Actor
    authenticated_at: datetime
    expires_at: datetime
    auth_method: AuthMethod
    suspicious: bool = False


@dataclass(frozen=True)
class Confirmation:
    confirmation_id: str
    tool_name: str
    target: str
    effect: str
    preview: dict[str, Any]
    risk: RiskLevel
    expires_at: datetime
    session_id: str = ""
    actor_id: str = ""
    argument_digest: str = ""
    approved: bool = False
    method: AuthMethod = AuthMethod.NONE


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid4().hex)
    confirmation: Confirmation | None = None


@dataclass(frozen=True)
class Evidence:
    kind: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    observed_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    status: str
    message: str
    evidence: tuple[Evidence, ...] = ()
    error_code: str | None = None
    retryable: bool = False
    dry_run: bool = False
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("started_at", "completed_at"):
            value[key] = value[key].isoformat()
        for item in value["evidence"]:
            item["observed_at"] = item["observed_at"].isoformat()
        return value


@dataclass(frozen=True)
class PlanStep:
    call: ToolCall
    description: str
    depends_on: tuple[str, ...] = ()
    step_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class ExecutionPlan:
    goal: str
    steps: tuple[PlanStep, ...]
    plan_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=utc_now)
