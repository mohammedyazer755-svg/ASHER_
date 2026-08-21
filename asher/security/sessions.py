"""Short-lived, actor-bound authentication sessions."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from asher.types import Actor, AuthMethod, Role, SessionContext


class SessionManager:
    def __init__(self, ttl_minutes: int = 10) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionContext] = {}

    def create(
        self,
        actor: Actor,
        method: AuthMethod,
        *,
        suspicious: bool = False,
        now: datetime | None = None,
    ) -> SessionContext:
        now = now or datetime.now(UTC)
        if actor.role is not Role.GUEST and method is AuthMethod.NONE:
            raise ValueError("Authenticated users require an authentication method")
        session = SessionContext(
            session_id=uuid4().hex,
            actor=actor,
            authenticated_at=now,
            expires_at=now + self.ttl,
            auth_method=method,
            suspicious=suspicious,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str, *, now: datetime | None = None) -> SessionContext | None:
        now = now or datetime.now(UTC)
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.expires_at > now:
                return session
            self._sessions.pop(session_id, None)
        return None

    def invalidate(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def invalidate_user(self, user_id: str) -> int:
        with self._lock:
            matching = [key for key, value in self._sessions.items() if value.actor.user_id == user_id]
            for key in matching:
                del self._sessions[key]
        return len(matching)

    def clear(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
        return count

