"""Thread-safe assistant state and lightweight event publication."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from asher.types import utc_now


class AssistantState(str, Enum):
    STANDBY = "standby"
    LISTENING = "listening"
    UNDERSTANDING = "understanding"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    ACTING = "acting"
    COMPLETE = "complete"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class StateEvent:
    state: AssistantState
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)


class StateStore:
    def __init__(self) -> None:
        self._state = AssistantState.STANDBY
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[StateEvent], None]] = []

    @property
    def current(self) -> AssistantState:
        with self._lock:
            return self._state

    def subscribe(self, callback: Callable[[StateEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def transition(self, state: AssistantState, message: str = "", **details: Any) -> StateEvent:
        event = StateEvent(state=state, message=message, details=details)
        with self._lock:
            self._state = state
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                continue
        return event

