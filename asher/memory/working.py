"""Bounded in-process conversation/task memory."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

from asher.types import utc_now


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    text: str
    timestamp: datetime = field(default_factory=utc_now)


class WorkingMemory:
    def __init__(self, maximum_turns: int = 20) -> None:
        self.maximum_turns = maximum_turns
        self._turns: dict[str, deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=self.maximum_turns)
        )
        self._lock = threading.Lock()

    def append(self, session_id: str, role: str, text: str) -> ConversationTurn:
        turn = ConversationTurn(role=role, text=text, timestamp=utc_now())
        with self._lock:
            self._turns[session_id].append(turn)
        return turn

    def recent(self, session_id: str, limit: int = 8) -> list[ConversationTurn]:
        with self._lock:
            return list(self._turns.get(session_id, ())) [-max(1, limit):]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)
