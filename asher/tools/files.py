"""Allow-listed file/folder tools with path traversal protection."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


class FileTools:
    def __init__(
        self,
        allowed_roots: list[str | Path],
        *,
        protected_paths: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> None:
        self.allowed_roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
        self.protected_paths = tuple(Path(path).expanduser().resolve() for path in protected_paths)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        path_schema = {"type": "string", "minLength": 1, "maxLength": 4096}
        return (
            ToolDefinition(
                name="files.list", description="List files and folders under an allow-listed root.",
                input_schema={"type": "object", "properties":{"path": path_schema}, "required":["path"], "additionalProperties":False},
                policy=ToolPolicy("private_files", RiskLevel.SENSITIVE), timeout_seconds=10,
                handler=self.list_files, preview=lambda args: (args["path"], f"List the contents of {args['path']}", {"path": args["path"]}), idempotent=True,
            ),
            ToolDefinition(
                name="files.read_text", description="Read a UTF-8 text file under an allow-listed root.",
                input_schema={"type":"object", "properties":{"path":path_schema, "max_chars":{"type":"integer"}}, "required":["path"], "additionalProperties":False},
                policy=ToolPolicy("private_files", RiskLevel.SENSITIVE), timeout_seconds=10,
                handler=self.read_text, preview=lambda args: (args["path"], f"Read the text in {args['path']}", {"path": args["path"]}), idempotent=True,
            ),
            ToolDefinition(
                name="files.write_text", description="Write UTF-8 text to an existing allow-listed location.",
                input_schema={"type":"object", "properties":{"path":path_schema, "content":{"type":"string", "maxLength":1_000_000}}, "required":["path","content"], "additionalProperties":False},
                policy=ToolPolicy("private_files", RiskLevel.SENSITIVE), timeout_seconds=15,
                handler=self.write_text, preview=lambda args: (args["path"], f"Replace the contents of {args['path']}", {"path": args["path"], "content": args["content"]}),
            ),
            ToolDefinition(
                name="files.delete", description="Move an allow-listed file to the recycle bin when supported.",
                input_schema={"type":"object", "properties":{"path":path_schema}, "required":["path"], "additionalProperties":False},
                policy=ToolPolicy("private_files", RiskLevel.SENSITIVE), timeout_seconds=15,
                handler=self.delete, preview=lambda args: (args["path"], f"Move {args['path']} to the recycle bin", {"path": args["path"]}),
            ),
        )

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw).expanduser().resolve()
        if not any(candidate == root or root in candidate.parents for root in self.allowed_roots):
            raise PermissionError("Path is outside ASHER's configured allow-list")
        return candidate

    def _is_protected(self, path: Path) -> bool:
        return any(path == protected for protected in self.protected_paths)

    def list_files(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            path = self._resolve(arguments["path"])
            if not path.is_dir():
                return _failure(context, "not_directory", "The requested path is not a directory")
            entries = sorted(path.iterdir(), key=lambda item: item.name.casefold())[:500]
            data = [{"name": item.name, "directory": item.is_dir()} for item in entries]
            if context.dry_run:
                data = [{"count": len(data)}]
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Found {len(entries)} entries.", (Evidence("directory_listing", "Allow-listed directory was inspected", {"path": str(path), "entries": data}),), dry_run=context.dry_run)
        except (OSError, PermissionError) as error:
            return _failure(context, "path_denied", str(error))

    def read_text(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            path = self._resolve(arguments["path"])
            if not path.is_file():
                return _failure(context, "not_file", "The requested path is not a file")
            max_chars = max(1, min(int(arguments.get("max_chars", 20_000)), 100_000))
            content = path.read_text(encoding="utf-8")[:max_chars]
            if context.dry_run:
                return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Dry run verified the file is readable.", (Evidence("dry_run", "File contents were not exposed", {"path": str(path), "characters": len(content)}),), dry_run=True)
            # Content is deliberately not placed in audit evidence; callers may
            # consume it through a separate trusted UI response channel.
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Read {len(content)} characters from {path.name}.", (Evidence("file_read", "Allow-listed text file was read", {"path": str(path), "characters": len(content)}),))
        except (OSError, UnicodeError, PermissionError) as error:
            return _failure(context, "read_failed", str(error))

    def write_text(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            path = self._resolve(arguments["path"])
            if self._is_protected(path):
                return _failure(context, "protected_target", "ASHER's runtime data cannot be overwritten through the file tool")
            if context.dry_run:
                return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Dry run verified the requested file write.", (Evidence("dry_run", "No file was modified", {"path": str(path), "characters": len(arguments["content"])}),), dry_run=True)
            if path.exists() and not path.is_file():
                return _failure(context, "not_file", "The requested path is not a regular file")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".asher-tmp")
            temporary.write_text(arguments["content"], encoding="utf-8")
            context.cancellation.raise_if_cancelled()
            temporary.replace(path)
            if not path.is_file() or path.read_text(encoding="utf-8") != arguments["content"]:
                return _failure(context, "write_unverified", "The file write could not be verified")
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Updated {path.name}.", (Evidence("file_hashless_match", "Written text was read back successfully", {"path": str(path), "characters": len(arguments["content"])}),))
        except (OSError, UnicodeError, PermissionError) as error:
            return _failure(context, "write_failed", str(error))

    def delete(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            path = self._resolve(arguments["path"])
            if path in self.allowed_roots:
                return _failure(context, "root_delete_forbidden", "Deleting an allow-listed root is not permitted")
            if path.is_dir():
                return _failure(context, "directory_delete_forbidden", "Only individual files may be moved to the recycle bin")
            if self._is_protected(path):
                return _failure(context, "protected_target", "ASHER's runtime data cannot be deleted through the file tool")
            if not path.exists():
                return _failure(context, "not_found", "The requested file does not exist")
            if context.dry_run:
                return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Dry run verified the requested deletion.", (Evidence("dry_run", "No file was removed", {"path": str(path)}),), dry_run=True)
            try:
                from send2trash import send2trash  # type: ignore[import-not-found]
            except ImportError:
                return _failure(context, "recycle_bin_unavailable", "Safe recycle-bin integration is unavailable; file was not removed")
            send2trash(str(path))
            context.cancellation.raise_if_cancelled()
            if path.exists():
                return _failure(context, "delete_unverified", "The file still exists after the recycle-bin request")
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Moved {path.name} to the recycle bin.", (Evidence("file_absent", "Target is no longer at the original path", {"path": str(path)}),))
        except (OSError, PermissionError) as error:
            return _failure(context, "delete_failed", str(error))


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"], success=False, status="failed", message=message, error_code=code, started_at=now, completed_at=now)
