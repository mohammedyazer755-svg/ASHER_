"""Contact-aware WhatsApp preparation and explicitly gated sending adapter."""

from __future__ import annotations

import hashlib
import os
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from asher.core.cancellation import CancelledError
from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


LIVE_WHATSAPP_ENV = "ASHER_ENABLE_LIVE_WHATSAPP"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def _live_whatsapp_enabled() -> bool:
    return os.getenv(LIVE_WHATSAPP_ENV, "false").strip().casefold() in _ENABLED_VALUES


def _contact_display(value: str) -> str:
    """Return a stable display/search surface without changing word content."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _normalized_contact(value: str) -> str:
    return _contact_display(value).casefold()


def _normalized_message(value: str) -> str:
    # Message comparison stays case- and whitespace-sensitive. Only Unicode
    # composition and platform newline spelling are normalized.
    normalized = unicodedata.normalize("NFKC", str(value))
    return normalized.replace("\r\n", "\n").replace("\r", "\n").strip()


def _message_digest(value: str) -> str:
    return hashlib.sha256(_normalized_message(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UIAObservation:
    """Privacy-safe result from a UI Automation observation.

    ``limitation`` must describe UI structure/capability only. Implementations
    must never place message text in it.
    """

    observed: bool
    limitation: str = ""


@dataclass(frozen=True)
class OutgoingMessageSnapshot:
    """Stable identities of matching outgoing rows present before a send."""

    row_ids: frozenset[str]


class WhatsAppUIAObserver(Protocol):
    """Injectable, fail-closed observation boundary for WhatsApp Desktop."""

    def prepare_exact_contact(self, contact: str) -> UIAObservation: ...

    def verify_active_chat_header(self, contact: str) -> UIAObservation: ...

    def focus_message_composer(self) -> UIAObservation: ...

    def snapshot_exact_outgoing_rows(
        self,
        message: str,
    ) -> tuple[UIAObservation, OutgoingMessageSnapshot | None]: ...

    def observe_new_exact_outgoing_row(
        self,
        message: str,
        previous: OutgoingMessageSnapshot,
    ) -> UIAObservation: ...


class WhatsAppInputBackend(Protocol):
    """Input-only boundary kept separate from UIA observation."""

    def stage_message(self, message: str, cancellation: Any) -> bool: ...

    def commit_message(self, cancellation: Any) -> bool: ...

    def discard_staged_message(self) -> None: ...


def _element_value(control: Any, attribute: str) -> Any:
    try:
        return getattr(getattr(control, "element_info"), attribute)
    except Exception:
        return None


def _control_name(control: Any) -> str:
    value = _element_value(control, "name")
    if isinstance(value, str) and value.strip():
        return value
    try:
        value = control.window_text()
    except Exception:
        return ""
    return value if isinstance(value, str) else ""


def _control_type(control: Any) -> str:
    value = _element_value(control, "control_type")
    return str(value or "").casefold()


def _control_marker(control: Any) -> str:
    values = (
        _element_value(control, "automation_id"),
        _element_value(control, "class_name"),
    )
    return " ".join(str(value) for value in values if value).casefold()


def _visible_and_enabled(control: Any) -> bool:
    try:
        return bool(control.is_visible() and control.is_enabled())
    except Exception:
        return False


def _rectangle(control: Any) -> tuple[int, int, int, int] | None:
    try:
        value = control.rectangle()
        rectangle = (
            int(value.left),
            int(value.top),
            int(value.right),
            int(value.bottom),
        )
    except Exception:
        return None
    if rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]:
        return None
    return rectangle


def _stable_row_id(control: Any) -> str | None:
    runtime_id = _element_value(control, "runtime_id")
    if isinstance(runtime_id, (tuple, list)) and runtime_id:
        return "runtime:" + ",".join(str(item) for item in runtime_id)
    automation_id = _element_value(control, "automation_id")
    if isinstance(automation_id, str) and automation_id.strip():
        return "automation:" + automation_id.strip()
    return None


class PywinautoWhatsAppObserver:
    """Conservative observer for the current WhatsApp Desktop UIA surface.

    WhatsApp does not publish a stable cross-version UIA schema. This observer
    therefore accepts only unique controls with exact normalized names in the
    expected pane, and only message rows explicitly marked as outgoing with a
    stable runtime/automation identifier. If a WhatsApp build exposes a
    different tree, it returns a precise limitation instead of guessing.
    """

    _SEARCH_NAMES = frozenset(
        {
            "search",
            "search or start new chat",
            "search or start a new chat",
        }
    )
    _COMPOSER_NAMES = frozenset({"message", "type a message"})
    _OUTGOING_MARKERS = ("outgoing", "message-out", "sent-message")
    _MESSAGE_SURFACE_MARKERS = ("conversation", "message-list", "messages", "chat-history")

    def __init__(
        self,
        *,
        desktop_factory: Any | None = None,
        launcher: Any | None = None,
        sleeper: Any = time.sleep,
        timeout_seconds: float = 6.0,
        poll_seconds: float = 0.2,
    ) -> None:
        self._desktop_factory = desktop_factory
        self._launcher = launcher
        self._sleeper = sleeper
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.poll_seconds = max(0.01, float(poll_seconds))
        self._window: Any | None = None

    def _load_dependencies(self) -> UIAObservation:
        try:
            if self._desktop_factory is None:
                from pywinauto import Desktop

                self._desktop_factory = Desktop
            if self._launcher is None:
                from actions.app_launcher import open_app

                self._launcher = open_app
        except Exception:
            return UIAObservation(
                False,
                "WhatsApp UI Automation requires pywinauto and the Windows app launcher.",
            )
        return UIAObservation(True)

    def _desktop_windows(self) -> tuple[Any, ...]:
        assert self._desktop_factory is not None
        desktop = self._desktop_factory(backend="uia")
        try:
            controls = desktop.windows()
        except Exception:
            return ()
        return tuple(
            control
            for control in controls
            if _visible_and_enabled(control)
            and "whatsapp" in _normalized_contact(_control_name(control))
        )

    def _find_window(self, *, launch: bool) -> tuple[Any | None, UIAObservation]:
        dependencies = self._load_dependencies()
        if not dependencies.observed:
            return None, dependencies
        windows = self._desktop_windows()
        if not windows and launch:
            try:
                assert self._launcher is not None
                self._launcher("WhatsApp")
            except Exception:
                return None, UIAObservation(False, "WhatsApp Desktop could not be launched.")
            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                self._sleeper(self.poll_seconds)
                windows = self._desktop_windows()
                if windows:
                    break
        if len(windows) != 1:
            limitation = (
                "WhatsApp UI Automation found no visible WhatsApp Desktop window."
                if not windows
                else "WhatsApp UI Automation found multiple visible WhatsApp windows and cannot choose safely."
            )
            return None, UIAObservation(False, limitation)
        self._window = windows[0]
        return self._window, UIAObservation(True)

    @staticmethod
    def _descendants(window: Any) -> tuple[Any, ...]:
        try:
            return tuple(window.descendants())
        except Exception:
            return ()

    @staticmethod
    def _right_pane(control: Any, window_rectangle: tuple[int, int, int, int]) -> bool:
        value = _rectangle(control)
        if value is None:
            return False
        midpoint = window_rectangle[0] + (window_rectangle[2] - window_rectangle[0]) * 0.42
        return (value[0] + value[2]) / 2 >= midpoint

    @staticmethod
    def _left_pane(control: Any, window_rectangle: tuple[int, int, int, int]) -> bool:
        value = _rectangle(control)
        if value is None:
            return False
        boundary = window_rectangle[0] + (window_rectangle[2] - window_rectangle[0]) * 0.48
        return (value[0] + value[2]) / 2 < boundary

    def prepare_exact_contact(self, contact: str) -> UIAObservation:
        display = _contact_display(contact)
        expected = _normalized_contact(display)
        if not expected:
            return UIAObservation(False, "The normalized WhatsApp contact is empty.")
        window, result = self._find_window(launch=True)
        if window is None:
            return result
        window_rectangle = _rectangle(window)
        if window_rectangle is None:
            return UIAObservation(False, "WhatsApp did not expose a usable window rectangle through UI Automation.")
        try:
            window.set_focus()
        except Exception:
            return UIAObservation(False, "WhatsApp could not be focused through UI Automation.")

        descendants = self._descendants(window)
        search_controls = tuple(
            control
            for control in descendants
            if _control_type(control) == "edit"
            and _visible_and_enabled(control)
            and self._left_pane(control, window_rectangle)
            and (
                _normalized_contact(_control_name(control)) in self._SEARCH_NAMES
                or "search" in _control_marker(control)
            )
        )
        if len(search_controls) != 1:
            return UIAObservation(
                False,
                "WhatsApp UI Automation did not expose exactly one recognizable chat-search field.",
            )
        search = search_controls[0]
        try:
            search.set_focus()
            search.set_edit_text(display)
        except Exception:
            return UIAObservation(False, "WhatsApp chat search could not be set through UI Automation.")

        deadline = time.monotonic() + self.timeout_seconds
        matches: tuple[Any, ...] = ()
        while True:
            matches = tuple(
                control
                for control in self._descendants(window)
                if _control_type(control) in {"listitem", "dataitem", "button"}
                and _visible_and_enabled(control)
                and self._left_pane(control, window_rectangle)
                and _normalized_contact(_control_name(control)) == expected
            )
            if matches or time.monotonic() >= deadline:
                break
            self._sleeper(self.poll_seconds)
        if len(matches) != 1:
            return UIAObservation(
                False,
                "WhatsApp UI Automation did not expose exactly one visible exact-match contact result; no result was selected.",
            )
        try:
            matches[0].click_input()
        except Exception:
            return UIAObservation(False, "The unique exact-match WhatsApp contact could not be selected through UI Automation.")

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            verified = self.verify_active_chat_header(display)
            if verified.observed or time.monotonic() >= deadline:
                return verified
            self._sleeper(self.poll_seconds)

    def verify_active_chat_header(self, contact: str) -> UIAObservation:
        expected = _normalized_contact(contact)
        window, result = self._find_window(launch=False)
        if window is None:
            return result
        window_rectangle = _rectangle(window)
        if window_rectangle is None:
            return UIAObservation(False, "WhatsApp did not expose a usable window rectangle through UI Automation.")
        header_bottom = window_rectangle[1] + (window_rectangle[3] - window_rectangle[1]) * 0.23
        matches = []
        for control in self._descendants(window):
            rectangle = _rectangle(control)
            if (
                _control_type(control) in {"text", "button", "group"}
                and _visible_and_enabled(control)
                and rectangle is not None
                and self._right_pane(control, window_rectangle)
                and (rectangle[1] + rectangle[3]) / 2 <= header_bottom
                and _normalized_contact(_control_name(control)) == expected
            ):
                matches.append(control)
        if len(matches) != 1:
            return UIAObservation(
                False,
                "WhatsApp UI Automation did not expose exactly one active-chat header matching the normalized contact.",
            )
        return UIAObservation(True)

    def focus_message_composer(self) -> UIAObservation:
        window, result = self._find_window(launch=False)
        if window is None:
            return result
        window_rectangle = _rectangle(window)
        if window_rectangle is None:
            return UIAObservation(False, "WhatsApp did not expose a usable window rectangle through UI Automation.")
        lower_boundary = window_rectangle[1] + (window_rectangle[3] - window_rectangle[1]) * 0.62
        matches = []
        for control in self._descendants(window):
            rectangle = _rectangle(control)
            marker = _control_marker(control)
            name = _normalized_contact(_control_name(control))
            if (
                _control_type(control) in {"edit", "document"}
                and _visible_and_enabled(control)
                and rectangle is not None
                and self._right_pane(control, window_rectangle)
                and (rectangle[1] + rectangle[3]) / 2 >= lower_boundary
                and (name in self._COMPOSER_NAMES or "composer" in marker or "message-input" in marker)
            ):
                matches.append(control)
        if len(matches) != 1:
            return UIAObservation(
                False,
                "WhatsApp UI Automation did not expose exactly one recognizable message composer.",
            )
        try:
            matches[0].click_input()
            matches[0].set_focus()
        except Exception:
            return UIAObservation(False, "The WhatsApp message composer could not be focused through UI Automation.")
        return UIAObservation(True)

    def _outgoing_row_ids(self, message: str) -> tuple[UIAObservation, frozenset[str] | None]:
        expected = _normalized_message(message)
        window, result = self._find_window(launch=False)
        if window is None:
            return result, None
        window_rectangle = _rectangle(window)
        if window_rectangle is None:
            return UIAObservation(False, "WhatsApp did not expose a usable window rectangle through UI Automation."), None
        descendants = self._descendants(window)
        surfaces = tuple(
            control
            for control in descendants
            if _control_type(control) in {"list", "pane", "group"}
            and _visible_and_enabled(control)
            and self._right_pane(control, window_rectangle)
            and any(marker in _control_marker(control) for marker in self._MESSAGE_SURFACE_MARKERS)
        )
        if not surfaces:
            return (
                UIAObservation(
                    False,
                    "This WhatsApp build does not expose a recognizable message-list surface through UI Automation; live sending is refused.",
                ),
                None,
            )

        identifiers: set[str] = set()
        unstable_match = False
        for control in descendants:
            marker = _control_marker(control)
            if (
                _control_type(control) not in {"listitem", "dataitem", "group"}
                or not _visible_and_enabled(control)
                or not self._right_pane(control, window_rectangle)
                or not any(value in marker for value in self._OUTGOING_MARKERS)
            ):
                continue
            texts = {_normalized_message(_control_name(control))}
            try:
                children = control.descendants()
            except Exception:
                children = ()
            texts.update(
                _normalized_message(_control_name(child))
                for child in children
                if _control_type(child) == "text" and _visible_and_enabled(child)
            )
            if expected not in texts:
                continue
            identifier = _stable_row_id(control)
            if identifier is None:
                unstable_match = True
            else:
                identifiers.add(identifier)
        if unstable_match:
            return (
                UIAObservation(
                    False,
                    "WhatsApp exposed a matching outgoing row without a stable UI Automation identity; it cannot be used as send evidence.",
                ),
                None,
            )
        return UIAObservation(True), frozenset(identifiers)

    def snapshot_exact_outgoing_rows(
        self,
        message: str,
    ) -> tuple[UIAObservation, OutgoingMessageSnapshot | None]:
        result, identifiers = self._outgoing_row_ids(message)
        if not result.observed or identifiers is None:
            return result, None
        return UIAObservation(True), OutgoingMessageSnapshot(identifiers)

    def observe_new_exact_outgoing_row(
        self,
        message: str,
        previous: OutgoingMessageSnapshot,
    ) -> UIAObservation:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            result, identifiers = self._outgoing_row_ids(message)
            if not result.observed or identifiers is None:
                return result
            new_rows = identifiers - previous.row_ids
            if len(new_rows) == 1:
                return UIAObservation(True)
            if len(new_rows) > 1:
                return UIAObservation(
                    False,
                    "WhatsApp exposed multiple new exact outgoing rows, so the send result is ambiguous.",
                )
            if time.monotonic() >= deadline:
                return UIAObservation(
                    False,
                    "WhatsApp did not expose one new exact outgoing message row before the observation timeout; recipient delivery is not claimed.",
                )
            self._sleeper(self.poll_seconds)


class ClipboardWhatsAppInputBackend:
    """Clipboard-backed input with a cancellable stage/commit boundary."""

    def __init__(self) -> None:
        self._staged = False

    def stage_message(self, message: str, cancellation: Any) -> bool:
        cancellation.raise_if_cancelled()
        try:
            import pyautogui
            import pyperclip

            original_clipboard = pyperclip.paste()
            try:
                pyperclip.copy(message)
                cancellation.raise_if_cancelled()
                pyautogui.hotkey("ctrl", "v")
            finally:
                pyperclip.copy(original_clipboard)
            self._staged = True
            return True
        except CancelledError:
            raise
        except Exception:
            return False

    def commit_message(self, cancellation: Any) -> bool:
        cancellation.raise_if_cancelled()
        if not self._staged:
            return False
        try:
            import pyautogui

            cancellation.raise_if_cancelled()
            pyautogui.press("enter")
            self._staged = False
            return True
        except CancelledError:
            raise
        except Exception:
            return False

    def discard_staged_message(self) -> None:
        if not self._staged:
            return
        try:
            import pyautogui

            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
        except Exception:
            pass
        finally:
            self._staged = False


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


@dataclass(frozen=True)
class _PendingObservation:
    contact: str
    message_digest: str
    snapshot: OutgoingMessageSnapshot


class WindowsWhatsAppAdapter:
    """Gated Windows adapter whose input is subordinate to exact UIA checks."""

    def __init__(
        self,
        *,
        observer: WhatsAppUIAObserver | None = None,
        input_backend: WhatsAppInputBackend | None = None,
    ) -> None:
        self.observer = observer or PywinautoWhatsAppObserver()
        self.input_backend = input_backend or ClipboardWhatsAppInputBackend()
        self.last_limitation = ""
        self._lock = threading.RLock()
        self._pending: _PendingObservation | None = None

    def _accept(self, result: UIAObservation) -> bool:
        self.last_limitation = "" if result.observed else result.limitation
        return result.observed

    def prepare(self, contact: str) -> bool:
        display = _contact_display(contact)
        if not display:
            self.last_limitation = "The normalized WhatsApp contact is empty."
            return False
        try:
            return self._accept(self.observer.prepare_exact_contact(display))
        except Exception:
            self.last_limitation = "WhatsApp UI Automation failed while preparing the exact contact."
            return False

    def verify_target(self, contact: str) -> bool:
        display = _contact_display(contact)
        if not display:
            self.last_limitation = "The normalized WhatsApp contact is empty."
            return False
        try:
            return self._accept(self.observer.verify_active_chat_header(display))
        except Exception:
            self.last_limitation = "WhatsApp UI Automation failed while verifying the active chat header."
            return False

    def send(self, contact: str, message: str, cancellation: Any) -> bool:
        if not _live_whatsapp_enabled():
            self.last_limitation = (
                f"Live WhatsApp sending is disabled; set {LIVE_WHATSAPP_ENV}=true explicitly to enable it."
            )
            return False
        display = _contact_display(contact)
        normalized_message = _normalized_message(message)
        if not display or not normalized_message:
            self.last_limitation = "The normalized WhatsApp recipient or message is empty."
            return False
        cancellation.raise_if_cancelled()
        staged = False
        committed = False
        with self._lock:
            if self._pending is not None:
                self.last_limitation = "A previous WhatsApp send is still awaiting UI Automation observation."
                return False
            try:
                if not self.verify_target(display):
                    return False
                if not self._accept(self.observer.focus_message_composer()):
                    return False
                # Focusing a control is not proof that the same chat remained
                # active. Re-observe before any message text enters the UI.
                if not self.verify_target(display):
                    return False
                snapshot_result, snapshot = self.observer.snapshot_exact_outgoing_rows(
                    normalized_message
                )
                if not self._accept(snapshot_result) or snapshot is None:
                    return False
                cancellation.raise_if_cancelled()
                if not self.input_backend.stage_message(normalized_message, cancellation):
                    self.last_limitation = "The WhatsApp message could not be staged for sending."
                    return False
                staged = True
                cancellation.raise_if_cancelled()
                # This is the final recipient check before the irreversible
                # Enter key. A changed or unreadable header clears the draft.
                if not self.verify_target(display):
                    return False
                cancellation.raise_if_cancelled()
                if not self.input_backend.commit_message(cancellation):
                    self.last_limitation = "The staged WhatsApp message could not be committed."
                    return False
                committed = True
                self._pending = _PendingObservation(
                    contact=_normalized_contact(display),
                    message_digest=_message_digest(normalized_message),
                    snapshot=snapshot,
                )
                self.last_limitation = ""
                return True
            except CancelledError:
                raise
            except Exception:
                self.last_limitation = "Live WhatsApp input failed safely before an outgoing row was verified."
                return False
            finally:
                if staged and not committed:
                    self.input_backend.discard_staged_message()

    def verify(self, contact: str, message: str) -> bool:
        display = _contact_display(contact)
        with self._lock:
            pending = self._pending
            if (
                pending is None
                or pending.contact != _normalized_contact(display)
                or pending.message_digest != _message_digest(message)
            ):
                self.last_limitation = "No matching committed WhatsApp send is awaiting observation."
                return False
            try:
                if not self.verify_target(display):
                    return False
                return self._accept(
                    self.observer.observe_new_exact_outgoing_row(
                        _normalized_message(message),
                        pending.snapshot,
                    )
                )
            except Exception:
                self.last_limitation = "WhatsApp UI Automation failed while observing the outgoing message row."
                return False
            finally:
                self._pending = None


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

    def _adapter_limitation(self) -> str:
        # Only expose the built-in adapter's controlled, privacy-safe text. An
        # arbitrary injected adapter could otherwise reflect message content.
        if isinstance(self.adapter, WindowsWhatsAppAdapter):
            value = self.adapter.last_limitation.strip()
            return f" {value}" if value else ""
        return ""

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
                name="whatsapp.send", description="Send an already-previewed WhatsApp message after non-voice approval and observe its exact outgoing row.",
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
            return _failure(
                context,
                "chat_unverified",
                f"Could not prepare the exact WhatsApp chat for {contact}.{self._adapter_limitation()}",
            )
        if not self._target_is_verified(contact):
            return _failure(
                context,
                "recipient_unverified",
                "WhatsApp search completed, but the exact recipient could not be observed. Nothing was sent."
                + self._adapter_limitation(),
            )
        return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Prepared the WhatsApp chat for {contact}.", (Evidence("chat_prepared", "The active chat header exactly matched the normalized contact", {"contact": contact}),))

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
        context.cancellation.raise_if_cancelled()
        # This pre-send observation is a hard safety boundary. Post-send
        # observation alone is too late: a wrong active chat may already have
        # received the message even if the tool subsequently reports failure.
        if not self._target_is_verified(contact):
            return _failure(
                context,
                "recipient_unverified",
                "The exact WhatsApp recipient could not be verified before sending. Nothing was sent."
                + self._adapter_limitation(),
            )
        if not _live_whatsapp_enabled():
            return _failure(
                context,
                "live_send_disabled",
                f"Live WhatsApp sending is disabled. Set {LIVE_WHATSAPP_ENV}=true explicitly to enable it.",
            )
        context.cancellation.raise_if_cancelled()
        if not self.adapter.send(contact, message, context.cancellation):
            return _failure(
                context,
                "send_failed",
                "The message was not sent or the adapter refused live sending."
                + self._adapter_limitation(),
            )
        if not self.adapter.verify(contact, message):
            return _failure(
                context,
                "delivery_unverified",
                "The send input completed, but one new exact outgoing WhatsApp message row could not be observed; recipient delivery is not claimed."
                + self._adapter_limitation(),
            )
        return successful_result(
            context.metadata["call_id"],
            context.metadata["tool_name"],
            f"Observed the exact outgoing WhatsApp message row for {contact}; recipient delivery was not independently confirmed.",
            (
                Evidence(
                    "outgoing_message_observed",
                    "One new exact outgoing WhatsApp message row was observed",
                    {"contact": contact, "message_length": len(message)},
                ),
            ),
        )


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"], success=False, status="failed", message=message, error_code=code, started_at=now, completed_at=now)
