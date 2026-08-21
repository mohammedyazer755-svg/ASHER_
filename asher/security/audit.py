"""Append-only, redacted local audit log separate from conversation history."""

from __future__ import annotations

import json
import hashlib
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asher.core.redaction import REDACTED, SENSITIVE_KEYS, redact_mapping, redact_text


OMITTED_CONTENT_KEYS = {
    "body", "content", "message", "raw_audio", "secret", "text", "value",
    "contact", "recipient", "query", "path", "url", "window_title",
}


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _safe_details(self, details: dict[str, Any] | None) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in (details or {}).items():
            normalized = key.lower().replace("-", "_")
            if normalized in OMITTED_CONTENT_KEYS:
                safe[key] = "[CONTENT OMITTED]"
            elif normalized in SENSITIVE_KEYS:
                safe[key] = REDACTED
            else:
                safe[key] = redact_mapping(value)
        return safe

    def append(
        self,
        event: str,
        *,
        actor_id: str = "system",
        session_id: str = "",
        tool_name: str = "",
        target: str = "",
        outcome: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        target_text = str(target).strip()
        target_digest = (
            "target:" + hashlib.sha256(target_text.encode("utf-8")).hexdigest()[:12]
            if target_text
            else ""
        )
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": redact_text(event),
            "actor_id": actor_id,
            "session_id": session_id,
            "tool_name": tool_name,
            # Keep only a correlation digest; contact names, window titles,
            # file paths, and URLs must not become plaintext audit data.
            "target": target_digest,
            "outcome": redact_text(outcome),
            "details": self._safe_details(details),
        }
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def read_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in lines[-max(1, limit):]:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError:
                continue
        return records
