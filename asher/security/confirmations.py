"""Expiring confirmations bound to the exact actor, session, target, and effect."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from asher.types import AuthMethod, Confirmation, RiskLevel, SessionContext


def canonical_argument_digest(arguments: Mapping[str, object] | None) -> str:
    """Return a stable digest for the exact structured tool payload."""

    encoded = json.dumps(
        dict(arguments or {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ConfirmationStore:
    def __init__(self, ttl_seconds: int = 90) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self._lock = threading.Lock()
        self._pending: dict[str, Confirmation] = {}
        # Approved confirmations remain server-side until the exact tool call
        # consumes them. This makes confirmation provenance and one-time use
        # enforceable at the registry boundary instead of trusting a caller-
        # constructed Confirmation dataclass.
        self._approved: dict[str, Confirmation] = {}

    def create(
        self,
        session: SessionContext,
        *,
        tool_name: str,
        target: str,
        effect: str,
        preview: dict,
        risk: RiskLevel,
        arguments: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> Confirmation:
        now = now or datetime.now(UTC)
        confirmation = Confirmation(
            confirmation_id=uuid4().hex,
            tool_name=tool_name,
            target=target,
            effect=effect,
            preview=copy.deepcopy(preview),
            risk=risk,
            expires_at=now + self.ttl,
            session_id=session.session_id,
            actor_id=session.actor.user_id,
            argument_digest=canonical_argument_digest(arguments),
        )
        with self._lock:
            self._pending[confirmation.confirmation_id] = confirmation
        return confirmation

    def get(self, confirmation_id: str, *, now: datetime | None = None) -> Confirmation | None:
        now = now or datetime.now(UTC)
        with self._lock:
            confirmation = self._pending.get(confirmation_id)
            if confirmation and confirmation.expires_at > now:
                return confirmation
            self._pending.pop(confirmation_id, None)
        return None

    def approve(
        self,
        confirmation_id: str,
        session: SessionContext,
        method: AuthMethod,
        *,
        now: datetime | None = None,
    ) -> Confirmation:
        pending = self.get(confirmation_id, now=now)
        if pending is None:
            raise ValueError("Confirmation is missing or expired")
        if pending.session_id != session.session_id or pending.actor_id != session.actor.user_id:
            raise PermissionError("Confirmation belongs to a different session")
        if method is AuthMethod.VOICE or method is AuthMethod.NONE:
            raise PermissionError("Consequential actions cannot be approved by voice alone")
        approved = Confirmation(
            confirmation_id=pending.confirmation_id,
            tool_name=pending.tool_name,
            target=pending.target,
            effect=pending.effect,
            preview=copy.deepcopy(pending.preview),
            risk=pending.risk,
            expires_at=pending.expires_at,
            session_id=pending.session_id,
            actor_id=pending.actor_id,
            argument_digest=pending.argument_digest,
            approved=True,
            method=method,
        )
        with self._lock:
            self._pending.pop(confirmation_id, None)
            self._approved[confirmation_id] = approved
        return approved

    def consume(
        self,
        supplied: Confirmation,
        session: SessionContext,
        *,
        tool_name: str,
        target: str,
        effect: str,
        preview: dict,
        risk: RiskLevel,
        arguments: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> Confirmation:
        """Consume one store-minted approval bound to the exact operation."""

        now = now or datetime.now(UTC)
        with self._lock:
            approved = self._approved.get(supplied.confirmation_id)
            if approved is None:
                raise PermissionError("Confirmation was not issued by the active confirmation store")
            expected = (
                approved.tool_name == tool_name
                and approved.target == target
                and approved.effect == effect
                and approved.preview == preview
                and approved.risk == risk
                and approved.session_id == session.session_id
                and approved.actor_id == session.actor.user_id
                and approved.argument_digest == canonical_argument_digest(arguments)
                and approved.approved
                and approved.method not in {AuthMethod.NONE, AuthMethod.VOICE}
                and approved.expires_at > now
            )
            # The call must carry the exact canonical value returned by
            # approve(), not merely a guessed ID with attacker-chosen fields.
            if not expected or supplied != approved:
                raise PermissionError("Confirmation does not match the approved operation")
            self._approved.pop(approved.confirmation_id, None)
            return approved

    def reject(self, confirmation_id: str, session: SessionContext) -> bool:
        pending = self.get(confirmation_id)
        if pending is None:
            return False
        if pending.session_id != session.session_id or pending.actor_id != session.actor.user_id:
            raise PermissionError("Confirmation belongs to a different session")
        with self._lock:
            self._pending.pop(confirmation_id, None)
        return True

    def cancel_all(self) -> int:
        with self._lock:
            count = len(self._pending) + len(self._approved)
            self._pending.clear()
            self._approved.clear()
        return count
