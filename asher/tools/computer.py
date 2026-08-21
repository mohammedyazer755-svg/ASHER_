"""Explicit, bounded keyboard/mouse/OCR tools; no arbitrary shell execution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


class ComputerTools:
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                name="computer.keypress", description="Send a bounded named key or short hotkey sequence after approval.",
                input_schema={"type":"object", "properties":{"keys":{"type":"string","minLength":1,"maxLength":80}}, "required":["keys"], "additionalProperties":False},
                policy=ToolPolicy("computer_control", RiskLevel.SENSITIVE), timeout_seconds=10,
                handler=self.keypress, preview=lambda args: ("focused window", f"Send the key sequence {args['keys']}", {"keys": args["keys"]}),
            ),
            ToolDefinition(
                name="computer.click", description="Click explicit screen coordinates after approval.",
                input_schema={"type":"object", "properties":{"x":{"type":"integer"},"y":{"type":"integer"}}, "required":["x","y"], "additionalProperties":False},
                policy=ToolPolicy("computer_control", RiskLevel.SENSITIVE), timeout_seconds=10,
                handler=self.click, preview=lambda args: ("screen", f"Click screen coordinate ({args['x']}, {args['y']})", dict(args)),
            ),
            ToolDefinition(
                name="computer.ocr", description="Run optional OCR on a local screenshot without sending it to a model.",
                input_schema={"type":"object", "properties":{"image_path":{"type":"string","minLength":1,"maxLength":4096}}, "required":["image_path"], "additionalProperties":False},
                policy=ToolPolicy("screenshot", RiskLevel.HARMLESS_LOCAL), timeout_seconds=20,
                handler=self.ocr, preview=lambda args: (args["image_path"], f"Read text from the local screenshot {args['image_path']}", {"path": args["image_path"]}), idempotent=True,
            ),
        )

    def keypress(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.dry_run:
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Dry run verified the key sequence.", (Evidence("dry_run", "No key was sent", {"keys": arguments["keys"]}),), dry_run=True)
        try:
            import pyautogui

            context.cancellation.raise_if_cancelled()
            # Restrict to pyautogui's named key grammar; no shell or code eval.
            keys = arguments["keys"].strip().lower()
            allowed = set("abcdefghijklmnopqrstuvwxyz0123456789+-_ ") | {"ctrl", "alt", "shift", "enter", "esc", "tab", "backspace", "home", "end", "up", "down", "left", "right"}
            if not all(part in allowed for part in keys.replace("+", " ").split()):
                return _failure(context, "invalid_key_sequence", "The key sequence contains unsupported tokens.")
            pyautogui.hotkey(*[part for part in keys.split("+") if part]) if "+" in keys else pyautogui.press(keys)
        except Exception as error:
            return _failure(context, "keypress_failed", f"Keypress failed: {type(error).__name__}")
        return _failure(context, "unverified_state", "The keypress was dispatched, but its application state was not verified.")

    def click(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if not 0 <= arguments["x"] <= 10000 or not 0 <= arguments["y"] <= 10000:
            return _failure(context, "invalid_coordinates", "Coordinates are outside the safe screen bounds.")
        if context.dry_run:
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Dry run verified the click coordinates.", (Evidence("dry_run", "No click was sent", dict(arguments)),), dry_run=True)
        try:
            import pyautogui

            context.cancellation.raise_if_cancelled()
            pyautogui.click(arguments["x"], arguments["y"])
        except Exception as error:
            return _failure(context, "click_failed", f"Click failed: {type(error).__name__}")
        return _failure(context, "unverified_state", "The click was dispatched, but its effect was not verified.")

    def ocr(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        from pathlib import Path

        path = Path(arguments["image_path"]).expanduser().resolve()
        if not path.is_file():
            return _failure(context, "image_not_found", "The screenshot file does not exist.")
        if context.dry_run:
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Dry run verified the screenshot path.", (Evidence("dry_run", "OCR was not run", {"path": str(path)}),), dry_run=True)
        try:
            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(path))
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"OCR read {len(text)} characters.", (Evidence("ocr_completed", "Local OCR returned text", {"path": str(path), "characters": len(text)}),))
        except ImportError:
            return _failure(context, "ocr_unavailable", "pytesseract and Tesseract are not installed.")
        except Exception as error:
            return _failure(context, "ocr_failed", f"OCR failed: {type(error).__name__}")


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"], success=False, status="failed", message=message, error_code=code, started_at=now, completed_at=now)
