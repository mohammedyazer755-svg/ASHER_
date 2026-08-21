"""Cooperative plan cancellation and application-wide emergency stop."""

from __future__ import annotations

import threading
from collections.abc import Callable


class CancelledError(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._callbacks: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "Cancelled by user") -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason.strip() or "Cancelled"
            self._event.set()
            callbacks = tuple(self._callbacks)

        for callback in callbacks:
            try:
                callback(self._reason)
            except Exception:
                # Cancellation must continue even if a provider cleanup hook fails.
                continue
        return True

    def add_callback(self, callback: Callable[[str], None]) -> Callable[[], None]:
        with self._lock:
            if self._event.is_set():
                reason = self._reason
            else:
                self._callbacks.append(callback)
                def unsubscribe() -> None:
                    with self._lock:
                        if callback in self._callbacks:
                            self._callbacks.remove(callback)

                return unsubscribe
        callback(reason)
        return lambda: None

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError(self.reason)

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


class EmergencyStop:
    """Owns every active plan token so one stop cancels the whole system."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: set[CancellationToken] = set()
        self._latched = False

    @property
    def latched(self) -> bool:
        with self._lock:
            return self._latched

    def register(self, token: CancellationToken) -> None:
        with self._lock:
            if self._latched:
                should_cancel = True
            else:
                self._tokens.add(token)
                should_cancel = False
        if should_cancel:
            token.cancel("Emergency stop is latched")

    def unregister(self, token: CancellationToken) -> None:
        with self._lock:
            self._tokens.discard(token)

    def trigger(self, reason: str = "Emergency stop activated") -> int:
        with self._lock:
            self._latched = True
            tokens = tuple(self._tokens)
        for token in tokens:
            token.cancel(reason)
        return len(tokens)

    def reset(self, *, local_ui_confirmed: bool) -> bool:
        if not local_ui_confirmed:
            return False
        with self._lock:
            self._latched = False
        return True
