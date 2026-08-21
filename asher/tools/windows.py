"""Verified Windows application discovery/open/close tools.

No configuration string is passed to a shell. Manual applications must be an
argument list whose first item is an executable path/name.
"""

from __future__ import annotations

import ctypes
import difflib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


@dataclass(frozen=True)
class AppEntry:
    name: str
    kind: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class AppResolution:
    entry: AppEntry | None
    alternatives: tuple[str, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return self.entry is None and len(self.alternatives) > 1


ALIASES = {
    "browser": "google chrome",
    "calc": "calculator",
    "chrome": "google chrome",
    "code": "visual studio code",
    "vs code": "visual studio code",
    "vscode": "visual studio code",
}


def _normalise(value: str) -> str:
    return " ".join("".join(character.casefold() if character.isalnum() else " " for character in value).split())


class AppCatalog:
    def __init__(self, manual_file: str | Path | None = None) -> None:
        self.manual_file = Path(manual_file) if manual_file else None
        self._entries: tuple[AppEntry, ...] | None = None

    def refresh(self) -> tuple[AppEntry, ...]:
        entries = self._load_start_apps() + self._load_manual()
        unique: dict[str, AppEntry] = {}
        for entry in entries:
            unique.setdefault(_normalise(entry.name), entry)
        self._entries = tuple(sorted(unique.values(), key=lambda item: item.name.casefold()))
        return self._entries

    def entries(self) -> tuple[AppEntry, ...]:
        return self._entries if self._entries is not None else self.refresh()

    def _load_start_apps(self) -> list[AppEntry]:
        if os.name != "nt":
            return []
        fixed_script = "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress"
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", fixed_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            data = json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else []
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return []
        if isinstance(data, dict):
            data = [data]
        return [
            AppEntry(str(item["Name"]).strip(), "start_app", (str(item["AppID"]).strip(),))
            for item in data
            if isinstance(item, dict) and item.get("Name") and item.get("AppID")
        ]

    def _load_manual(self) -> list[AppEntry]:
        if not self.manual_file or not self.manual_file.is_file():
            return []
        try:
            data = json.loads(self.manual_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        entries: list[AppEntry] = []
        if not isinstance(data, dict):
            return entries
        for name, command in data.items():
            if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
                continue
            expanded = tuple(os.path.expandvars(item) for item in command)
            entries.append(AppEntry(str(name).strip(), "command", expanded))
        return entries

    def resolve(self, query: str, *, threshold: float = 0.68, ambiguity_margin: float = 0.08) -> AppResolution:
        normalized = _normalise(query)
        normalized = ALIASES.get(normalized, normalized)
        if not normalized:
            return AppResolution(None)
        scores: list[tuple[float, AppEntry]] = []
        for entry in self.entries():
            candidate = _normalise(entry.name)
            if candidate == normalized:
                return AppResolution(entry)
            score = difflib.SequenceMatcher(None, normalized, candidate).ratio()
            if normalized in candidate or candidate in normalized:
                score = max(score, 0.90 - abs(len(normalized) - len(candidate)) * 0.005)
            scores.append((score, entry))
        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores or scores[0][0] < threshold:
            return AppResolution(None)
        if len(scores) > 1 and scores[0][0] - scores[1][0] < ambiguity_margin:
            return AppResolution(None, (scores[0][1].name, scores[1][1].name))
        return AppResolution(scores[0][1])


def visible_windows() -> list[tuple[int, str]]:
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    windows: list[tuple[int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def callback(handle: int, _: int) -> bool:
        if not user32.IsWindowVisible(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        if buffer.value.strip():
            windows.append((int(handle), buffer.value.strip()))
        return True

    user32.EnumWindows(callback, 0)
    return windows


class WindowsAppTools:
    def __init__(self, catalog: AppCatalog) -> None:
        self.catalog = catalog

    def definitions(self) -> tuple[ToolDefinition, ...]:
        base_schema = {
            "type": "object",
            "properties": {"app_name": {"type": "string", "minLength": 1, "maxLength": 120}},
            "required": ["app_name"],
            "additionalProperties": False,
        }
        return (
            ToolDefinition(
                name="app.open",
                description="Open one installed Windows application and observe a matching window.",
                input_schema=base_schema,
                policy=ToolPolicy("open_app", RiskLevel.HARMLESS_LOCAL),
                timeout_seconds=20,
                handler=self.open_app,
                preview=lambda args: (args["app_name"], f"Open {args['app_name']}", dict(args)),
            ),
            ToolDefinition(
                name="app.close",
                description="Gracefully request that matching Windows application windows close, then verify disappearance.",
                input_schema=base_schema,
                policy=ToolPolicy("close_app", RiskLevel.HARMLESS_LOCAL),
                timeout_seconds=15,
                handler=self.close_app,
                preview=lambda args: (args["app_name"], f"Close visible {args['app_name']} windows gracefully", dict(args)),
            ),
            ToolDefinition(
                name="app.discover",
                description="Refresh the local installed-application catalog.",
                input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                policy=ToolPolicy("open_app", RiskLevel.HARMLESS_LOCAL),
                timeout_seconds=20,
                handler=self.discover,
                preview=lambda args: ("installed applications", "Read the Windows application catalog", {}),
                idempotent=True,
            ),
        )

    def discover(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        context.cancellation.raise_if_cancelled()
        entries = self.catalog.refresh()
        return successful_result(
            context.metadata["call_id"],
            context.metadata["tool_name"],
            f"Found {len(entries)} installed applications.",
            (Evidence("catalog_count", "Application catalog refreshed", {"count": len(entries)}),),
            dry_run=context.dry_run,
        )

    def open_app(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments["app_name"].strip()
        resolution = self.catalog.resolve(query)
        if resolution.ambiguous:
            return _tool_failure(context, "ambiguous_target", f"Did you mean {' or '.join(resolution.alternatives)}?")
        if resolution.entry is None:
            return _tool_failure(context, "not_found", f"I could not find {query} in the installed application catalog.")
        entry = resolution.entry
        if context.dry_run:
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                f"Dry run verified the request to open {entry.name}.",
                (Evidence("dry_run", "No application was launched", {"resolved_app": entry.name}),),
                dry_run=True,
            )
        context.cancellation.raise_if_cancelled()
        before = {handle for handle, _ in visible_windows()}
        try:
            if entry.kind == "start_app":
                process = subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{entry.command[0]}"])
            else:
                executable = entry.command[0]
                if ("\\" in executable or "/" in executable) and not Path(executable).is_file():
                    return _tool_failure(context, "executable_missing", f"The configured executable for {entry.name} does not exist.")
                process = subprocess.Popen(list(entry.command), shell=False)
        except OSError as error:
            return _tool_failure(context, "launch_failed", f"Could not launch {entry.name}: {type(error).__name__}")

        deadline = time.monotonic() + 8
        observed: tuple[int, str] | None = None
        target = _normalise(entry.name)
        while time.monotonic() < deadline:
            context.cancellation.raise_if_cancelled()
            for handle, title in visible_windows():
                if handle not in before and (target in _normalise(title) or _normalise(title) in target):
                    observed = (handle, title)
                    break
            if observed:
                break
            time.sleep(0.2)
        if not observed:
            return _tool_failure(context, "unverified_launch", f"{entry.name} was launched but no matching window was observed.")
        return successful_result(
            context.metadata["call_id"], context.metadata["tool_name"], f"Opened {entry.name}.",
            (Evidence("window_observed", "Matching application window appeared", {"window_title": observed[1], "process_id": process.pid}),),
        )

    def close_app(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments["app_name"].strip()
        resolution = self.catalog.resolve(query)
        if resolution.ambiguous:
            return _tool_failure(context, "ambiguous_target", f"Did you mean {' or '.join(resolution.alternatives)}?")
        resolved_name = resolution.entry.name if resolution.entry else query
        normalized = _normalise(resolved_name)
        if context.dry_run:
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                f"Dry run verified the request to close {resolved_name}.",
                (Evidence("dry_run", "No window was closed", {"resolved_app": resolved_name}),),
                dry_run=True,
            )
        matches = [(handle, title) for handle, title in visible_windows() if normalized in _normalise(title)]
        if not matches:
            return _tool_failure(context, "not_open", f"No visible {resolved_name} window was found.")
        if os.name != "nt":
            return _tool_failure(context, "unsupported_platform", "Windows application control is only available on Windows.")
        failed_posts: list[str] = []
        for handle, title in matches:
            context.cancellation.raise_if_cancelled()
            if not ctypes.windll.user32.PostMessageW(handle, 0x0010, 0, 0):
                failed_posts.append(title)
        if failed_posts:
            return _tool_failure(context, "close_request_failed", "Windows rejected one or more graceful close requests.")
        handles = {handle for handle, _ in matches}
        deadline = time.monotonic() + 8
        remaining = handles
        while time.monotonic() < deadline:
            context.cancellation.raise_if_cancelled()
            remaining = handles & {handle for handle, _ in visible_windows()}
            if not remaining:
                break
            time.sleep(0.2)
        if remaining:
            return _tool_failure(context, "unverified_close", "The close request was sent, but one or more windows remain visible.")
        return successful_result(
            context.metadata["call_id"], context.metadata["tool_name"], f"Closed {resolved_name}.",
            (Evidence("window_disappeared", "All targeted windows disappeared", {"count": len(matches)}),),
        )


def _tool_failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = time.time()
    from datetime import UTC, datetime

    timestamp = datetime.fromtimestamp(now, UTC)
    return ToolResult(
        call_id=context.metadata["call_id"],
        tool_name=context.metadata["tool_name"],
        success=False,
        status="failed",
        message=message,
        error_code=code,
        started_at=timestamp,
        completed_at=timestamp,
    )
