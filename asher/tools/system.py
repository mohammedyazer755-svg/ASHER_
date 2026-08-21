"""System/media/screenshot tools with dry-run and evidence-first results."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


class SystemTools:
    def __init__(self, screenshot_dir: str | Path, *, retention_count: int = 20) -> None:
        self.screenshot_dir = Path(screenshot_dir)
        if retention_count < 1:
            raise ValueError("Screenshot retention_count must be positive")
        self.retention_count = retention_count

    def definitions(self) -> tuple[ToolDefinition, ...]:
        empty = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        return (
            ToolDefinition(
                name="system.volume_up", description="Increase Windows master volume.", input_schema=empty,
                policy=ToolPolicy("media", RiskLevel.HARMLESS_LOCAL), timeout_seconds=10,
                handler=lambda args, ctx: self._media_key("volumeup", ctx, "Volume up"),
                preview=lambda args: ("Windows master volume", "Increase master volume", {}),
            ),
            ToolDefinition(
                name="system.volume_down", description="Decrease Windows master volume.", input_schema=empty,
                policy=ToolPolicy("media", RiskLevel.HARMLESS_LOCAL), timeout_seconds=10,
                handler=lambda args, ctx: self._media_key("volumedown", ctx, "Volume down"),
                preview=lambda args: ("Windows master volume", "Decrease master volume", {}),
            ),
            ToolDefinition(
                name="system.toggle_mute", description="Toggle Windows master mute.", input_schema=empty,
                policy=ToolPolicy("media", RiskLevel.HARMLESS_LOCAL), timeout_seconds=10,
                handler=lambda args, ctx: self._media_key("volumemute", ctx, "Toggle mute"),
                preview=lambda args: ("Windows master volume", "Toggle master mute", {}),
            ),
            ToolDefinition(
                name="system.screenshot", description="Capture a screenshot and verify the image file.", input_schema=empty,
                # A screenshot can contain passwords, private messages, or
                # health/financial data.  Treat capture as sensitive even
                # though it remains local and is never sent automatically.
                policy=ToolPolicy("screenshot", RiskLevel.SENSITIVE), timeout_seconds=20,
                handler=self.screenshot,
                preview=lambda args: ("current screen", "Capture a local screenshot", {}),
            ),
            ToolDefinition(
                name="system.lock", description="Request a Windows lock after explicit confirmation.", input_schema=empty,
                policy=ToolPolicy("security", RiskLevel.SENSITIVE), timeout_seconds=10,
                handler=self.lock,
                preview=lambda args: ("Windows workstation", "Lock the workstation and require sign-in to resume", {}),
            ),
        )

    def _media_key(self, key: str, context: ToolContext, label: str) -> ToolResult:
        if context.dry_run:
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                f"Dry run verified: {label} would be requested.",
                (Evidence("dry_run", "No media key was sent", {"key": key}),), dry_run=True,
            )
        try:
            import pyautogui

            context.cancellation.raise_if_cancelled()
            pyautogui.press(key)
        except Exception as error:
            return _failure(context, "dispatch_failed", f"Could not request {label}: {type(error).__name__}")
        # Without a Core Audio reader installed, an emitted key is not proof of
        # state. Be explicit rather than reporting a false success.
        return _failure(context, "unverified_state", f"{label} was requested, but Windows audio state could not be observed safely.")

    def screenshot(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.dry_run:
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                "Dry run verified screenshot capture.",
                (Evidence("dry_run", "No screenshot was written", {}),), dry_run=True,
            )
        try:
            import pyautogui

            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.screenshot_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            image = pyautogui.screenshot()
            context.cancellation.raise_if_cancelled()
            image.save(path)
            if not path.is_file() or path.stat().st_size <= 0:
                return _failure(context, "unverified_file", "Screenshot capture did not produce a readable file.")
            width, height = image.size
            self._prune_screenshots()
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                f"Screenshot saved to {path.name}.",
                (Evidence("screenshot_file", "Screenshot file exists and has dimensions", {"path": str(path), "width": width, "height": height}),),
            )
        except Exception as error:
            return _failure(context, "capture_failed", f"Screenshot failed safely: {type(error).__name__}")

    def _prune_screenshots(self) -> None:
        """Keep a bounded local history so screen contents do not accumulate."""

        files = sorted(
            (item for item in self.screenshot_dir.glob("screenshot_*.png") if item.is_file()),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in files[self.retention_count :]:
            try:
                stale.unlink()
            except OSError:
                continue

    def lock(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.dry_run:
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                "Dry run verified workstation lock request.",
                (Evidence("dry_run", "The workstation was not locked", {}),), dry_run=True,
            )
        if os.name != "nt":
            return _failure(context, "unsupported_platform", "Workstation locking is available only on Windows.")
        try:
            import ctypes

            context.cancellation.raise_if_cancelled()
            if not ctypes.windll.user32.LockWorkStation():
                return _failure(context, "lock_rejected", "Windows rejected the lock request.")
            return successful_result(
                context.metadata["call_id"], context.metadata["tool_name"],
                "Windows accepted the workstation lock request.",
                (Evidence("lock_api_accepted", "LockWorkStation returned success", {}),),
            )
        except Exception as error:
            return _failure(context, "lock_failed", f"Could not lock Windows: {type(error).__name__}")


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(
        call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"],
        success=False, status="failed", message=message, error_code=code,
        started_at=now, completed_at=now,
    )
