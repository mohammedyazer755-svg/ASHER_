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
    """Canonical assistant states exposed to desktop/voice presentation layers.

    The cinematic UI uses these names directly. Legacy coarse names remain
    available as aliases so existing integrations do not break while callers
    migrate to the more precise state model.
    """

    STANDBY = "standby"
    WAKE_DETECTED = "wake_detected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    OBSERVING = "observing"
    SPEAKING = "speaking"
    SUCCESS = "success"
    ERROR = "error"
    OFFLINE = "offline"
    STOPPED = "stopped"
    LOCKED = "locked"

    # Backward-compatible legacy states used by the isolated UI test/fallback
    # controller. The real CompanionController now emits the precise states
    # above, so old integrations keep their historical string values.
    UNDERSTANDING = "understanding"
    ACTING = "acting"
    COMPLETE = "complete"


@dataclass(frozen=True)
class StateEvent:
    state: AssistantState
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
    previous_state: AssistantState | None = None


class StateStore:
    """One observable state store shared by the agent, voice runtime and UI."""

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
        with self._lock:
            previous = self._state
            self._state = state
            subscribers = tuple(self._subscribers)
        event = StateEvent(
            state=state,
            message=message,
            details=details,
            previous_state=previous,
        )
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                continue
        return event
