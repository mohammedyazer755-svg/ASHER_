"""Central per-tool role, permission, confirmation, and strong-auth policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from asher.types import AuthMethod, Confirmation, RiskLevel, Role, SessionContext


class DecisionKind(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_STRONG_AUTH = "require_strong_auth"


@dataclass(frozen=True)
class PolicyDecision:
    kind: DecisionKind
    reason: str

    @property
    def allowed(self) -> bool:
        return self.kind is DecisionKind.ALLOW


@dataclass(frozen=True)
class ToolPolicy:
    capability: str
    risk: RiskLevel


class PolicyEngine:
    def evaluate(
        self,
        policy: ToolPolicy,
        session: SessionContext | None,
        confirmation: Confirmation | None = None,
    ) -> PolicyDecision:
        if session is None:
            return PolicyDecision(DecisionKind.DENY, "An active session is required")
        if session.expires_at <= datetime.now(UTC):
            return PolicyDecision(DecisionKind.DENY, "The authenticated session has expired")

        actor = session.actor
        if (
            session.suspicious
            and policy.risk > RiskLevel.CONVERSATION
            and (
                confirmation is None
                or not confirmation.approved
                or confirmation.method is not AuthMethod.DEVICE_CREDENTIAL
            )
        ):
            return PolicyDecision(DecisionKind.REQUIRE_STRONG_AUTH, "Suspicious session requires device authentication")

        if policy.risk is RiskLevel.CONVERSATION:
            return PolicyDecision(DecisionKind.ALLOW, "Conversation is available to guests")

        if actor.role is Role.GUEST:
            return PolicyDecision(DecisionKind.DENY, "Guests cannot access private data or tools")

        if actor.role is Role.TRUSTED:
            aliases = {
                "private_memory": {"private_memory", "memory.read"},
                "private_memory_write": {"private_memory_write", "memory.write"},
                "memory.read": {"private_memory", "memory.read"},
                "memory.write": {"private_memory_write", "memory.write"},
            }
            allowed = aliases.get(policy.capability, {policy.capability})
            if not any(perm in actor.permissions for perm in allowed):
                return PolicyDecision(DecisionKind.DENY, "This capability was not granted to the trusted user")

        if policy.risk is RiskLevel.HARMLESS_LOCAL:
            if session.auth_method in {AuthMethod.VOICE, AuthMethod.LOCAL_UI, AuthMethod.DEVICE_CREDENTIAL}:
                return PolicyDecision(DecisionKind.ALLOW, "Authenticated harmless local action")
            return PolicyDecision(DecisionKind.DENY, "Authentication is required")

        if policy.risk is RiskLevel.FINANCIAL_OR_SECURITY and actor.role is not Role.OWNER:
            return PolicyDecision(DecisionKind.DENY, "Only the owner may request financial or security actions")

        if confirmation is None or not confirmation.approved:
            kind = (
                DecisionKind.REQUIRE_STRONG_AUTH
                if policy.risk >= RiskLevel.SENSITIVE
                else DecisionKind.REQUIRE_CONFIRMATION
            )
            return PolicyDecision(kind, "An exact local preview must be approved")

        if confirmation.actor_id != actor.user_id or confirmation.session_id != session.session_id:
            return PolicyDecision(DecisionKind.DENY, "Approval belongs to a different actor or session")

        if confirmation.method is AuthMethod.VOICE or confirmation.method is AuthMethod.NONE:
            return PolicyDecision(DecisionKind.DENY, "Voice-only approval is not accepted")

        if policy.risk >= RiskLevel.SENSITIVE and confirmation.method is not AuthMethod.DEVICE_CREDENTIAL:
            return PolicyDecision(DecisionKind.REQUIRE_STRONG_AUTH, "Device authentication is required")

        return PolicyDecision(DecisionKind.ALLOW, "Authorized by policy")
