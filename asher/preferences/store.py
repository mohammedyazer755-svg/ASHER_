"""Privacy-scoped local store for PreferenceCore feedback examples."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from asher.core.redaction import contains_prohibited_secret, redact_mapping, redact_text
from asher.preferences.schema import FEEDBACK_DIMENSIONS, FEEDBACK_KINDS, PreferenceEvent
from asher.storage import Database
from asher.types import Actor, Role


_MAX_TEXT_CHARS = 6000
_ALLOWED_CONTEXT_KEYS = {
    "provider",
    "offline",
    "confirmation_required",
    "tool_names",
    "user_chars",
    "assistant_chars",
}


class PreferenceStore:
    """Owner-only store for explicit, local preference-learning feedback."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def _setting(self, key: str, default: str) -> str:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT setting_value FROM app_settings WHERE setting_key=?",
                (key,),
            ).fetchone()
        return default if row is None else str(row["setting_value"])

    @property
    def enabled(self) -> bool:
        # Preference capture is opt-in.  We never silently turn normal chat into
        # a training dataset.
        return self._setting("preference_learning_enabled", "false").casefold() == "true"

    def configure(self, *, enabled: bool) -> None:
        now = datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_settings(setting_key, setting_value, updated_at) "
                "VALUES(?, ?, ?) ON CONFLICT(setting_key) DO UPDATE SET "
                "setting_value=excluded.setting_value, updated_at=excluded.updated_at",
                (
                    "preference_learning_enabled",
                    "true" if bool(enabled) else "false",
                    now,
                ),
            )

    @staticmethod
    def _require_owner(requester: Actor, owner_id: str) -> None:
        if requester.role is not Role.OWNER or requester.user_id != owner_id:
            raise PermissionError("PreferenceCore feedback belongs to the owner profile")

    @staticmethod
    def _clean_text(value: str, *, field: str, required: bool = True) -> str:
        text = str(value).strip()
        if required and not text:
            raise ValueError(f"{field} is required")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"{field} is too long for a preference example")
        if contains_prohibited_secret(text):
            raise ValueError("Credential-like content is never stored as PreferenceCore data")
        return redact_text(text)

    @staticmethod
    def _safe_context(context: dict[str, Any] | None) -> dict[str, Any]:
        source = context or {}
        clean: dict[str, Any] = {}
        for key in _ALLOWED_CONTEXT_KEYS:
            if key not in source:
                continue
            value = source[key]
            if key == "tool_names":
                if not isinstance(value, (list, tuple)):
                    continue
                clean[key] = [str(item)[:120] for item in value[:20]]
            elif key in {"offline", "confirmation_required"}:
                clean[key] = bool(value)
            elif key in {"user_chars", "assistant_chars"}:
                clean[key] = max(0, int(value))
            else:
                clean[key] = str(value)[:120]
        return redact_mapping(clean)

    @staticmethod
    def _from_row(row: Any) -> PreferenceEvent:
        dimensions = json.loads(str(row["dimensions_json"]))
        context = json.loads(str(row["context_json"]))
        return PreferenceEvent(
            event_id=str(row["event_id"]),
            owner_id=str(row["owner_id"]),
            session_id=str(row["session_id"]),
            source_hash=str(row["source_hash"]),
            user_text=str(row["user_text"]),
            assistant_text=str(row["assistant_text"]),
            preferred_text=(
                str(row["preferred_text"]) if row["preferred_text"] is not None else None
            ),
            feedback_kind=str(row["feedback_kind"]),
            dimensions=tuple(str(item) for item in dimensions),
            context=dict(context),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    def record(
        self,
        requester: Actor,
        *,
        owner_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        feedback_kind: str,
        preferred_text: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PreferenceEvent:
        self._require_owner(requester, owner_id)
        if not self.enabled:
            raise PermissionError("Preference learning is disabled; enable it locally first")
        kind = str(feedback_kind).strip().casefold()
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"Unsupported preference feedback: {kind}")
        clean_user = self._clean_text(user_text, field="User text")
        clean_assistant = self._clean_text(assistant_text, field="Assistant text")
        clean_preferred: str | None = None
        if kind == "preferred_reply":
            clean_preferred = self._clean_text(
                preferred_text or "", field="Preferred reply"
            )
        elif preferred_text is not None and str(preferred_text).strip():
            raise ValueError("Preferred text is only valid for preferred_reply feedback")
        clean_context = self._safe_context(context)
        dimensions = FEEDBACK_DIMENSIONS[kind]
        created_at = datetime.now(UTC)
        source_hash = hashlib.sha256(
            (clean_user + "\n---\n" + clean_assistant).encode("utf-8")
        ).hexdigest()
        event = PreferenceEvent(
            event_id=uuid4().hex,
            owner_id=owner_id,
            session_id=str(session_id),
            source_hash=source_hash,
            user_text=clean_user,
            assistant_text=clean_assistant,
            preferred_text=clean_preferred,
            feedback_kind=kind,
            dimensions=dimensions,
            context=clean_context,
            created_at=created_at,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO preference_events("
                "event_id, owner_id, session_id, source_hash, user_text, assistant_text, "
                "preferred_text, feedback_kind, dimensions_json, context_json, created_at, deleted_at"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    event.event_id,
                    event.owner_id,
                    event.session_id,
                    event.source_hash,
                    event.user_text,
                    event.assistant_text,
                    event.preferred_text,
                    event.feedback_kind,
                    json.dumps(list(event.dimensions), ensure_ascii=False, separators=(",", ":")),
                    json.dumps(event.context, ensure_ascii=False, separators=(",", ":")),
                    event.created_at.isoformat(),
                ),
            )
        return event

    def list_events(
        self,
        requester: Actor,
        *,
        owner_id: str,
        limit: int = 500,
    ) -> tuple[PreferenceEvent, ...]:
        self._require_owner(requester, owner_id)
        bounded = max(1, min(int(limit), 5000))
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM preference_events WHERE owner_id=? AND deleted_at IS NULL "
                "ORDER BY created_at ASC LIMIT ?",
                (owner_id, bounded),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def delete(
        self,
        requester: Actor,
        *,
        owner_id: str,
        event_id: str,
    ) -> bool:
        self._require_owner(requester, owner_id)
        now = datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE preference_events SET deleted_at=? "
                "WHERE event_id=? AND owner_id=? AND deleted_at IS NULL",
                (now, str(event_id), owner_id),
            )
        return cursor.rowcount > 0
