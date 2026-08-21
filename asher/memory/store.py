"""Consent-aware, editable SQLite long-term memory."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from asher.core.redaction import contains_prohibited_secret
from asher.storage import Database
from asher.types import Actor, Role


MEMORY_TYPES = {
    "episodic",
    "interaction_preference",
    "people_relationship",
    "preference_routine",
    "project_goal",
    "semantic",
    "task",
}


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    owner_id: str
    memory_type: str
    key: str
    value: str
    source: str
    confidence: float
    sensitivity: str
    consented: bool
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("created_at", "updated_at", "expires_at"):
            if value[field] is not None:
                value[field] = value[field].isoformat()
        return value


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _canonical_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _credential_field(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized in {
        "password",
        "passcode",
        "pin",
        "api_key",
        "api_token",
        "access_token",
        "refresh_token",
        "secret",
        "secret_token",
        "credential",
        "credentials",
    }


class MemoryStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _can_access(requester: Actor, owner_id: str, capability: str) -> bool:
        if requester.role is Role.OWNER and requester.user_id == owner_id:
            return True
        return requester.role is Role.TRUSTED and capability in requester.permissions

    @staticmethod
    def _from_row(row: Any) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            owner_id=row["owner_id"],
            memory_type=row["memory_type"],
            key=row["memory_key"],
            value=row["value"],
            source=row["source"],
            confidence=float(row["confidence"]),
            sensitivity=row["sensitivity"],
            consented=bool(row["consented"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=_parse_time(row["expires_at"]),
        )

    def put(
        self,
        requester: Actor,
        *,
        owner_id: str,
        memory_type: str,
        key: str,
        value: str,
        source: str,
        confidence: float = 1.0,
        sensitivity: str = "normal",
        consented: bool = False,
        confirmed: bool = False,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        if not self._can_access(requester, owner_id, "memory.write"):
            raise PermissionError("Private memory access denied")
        if not confirmed:
            raise PermissionError("Memory creation and updates require explicit confirmation")
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        key = _canonical_key(key)
        value = value.strip()
        source = source.strip() or "user"
        if not key or not value:
            raise ValueError("Memory key and value are required")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if sensitivity not in {"normal", "sensitive"}:
            raise ValueError("Sensitivity must be normal or sensitive")
        if sensitivity == "sensitive" and not consented:
            raise PermissionError("Sensitive memory requires explicit consent")
        if _credential_field(key) or _credential_field(value) or contains_prohibited_secret(value) or contains_prohibited_secret(key):
            raise ValueError("Passwords, PINs, API keys, tokens, and credentials cannot be stored")

        now = datetime.now(UTC)
        memory_id = uuid4().hex
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT memory_id, created_at FROM memory_records "
                "WHERE owner_id=? AND memory_type=? AND memory_key=?",
                (owner_id, memory_type, key),
            ).fetchone()
            if existing:
                memory_id = existing["memory_id"]
                created_at = existing["created_at"]
                connection.execute(
                    "UPDATE memory_records SET value=?, source=?, confidence=?, sensitivity=?, "
                    "consented=?, updated_at=?, expires_at=?, deleted_at=NULL WHERE memory_id=?",
                    (
                        value,
                        source,
                        confidence,
                        sensitivity,
                        int(consented),
                        now.isoformat(),
                        expires_at.isoformat() if expires_at else None,
                        memory_id,
                    ),
                )
            else:
                created_at = now.isoformat()
                connection.execute(
                    "INSERT INTO memory_records(memory_id, owner_id, memory_type, memory_key, value, "
                    "source, confidence, sensitivity, consented, created_at, updated_at, expires_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory_id,
                        owner_id,
                        memory_type,
                        key,
                        value,
                        source,
                        confidence,
                        sensitivity,
                        int(consented),
                        created_at,
                        now.isoformat(),
                        expires_at.isoformat() if expires_at else None,
                    ),
                )
        record = self.get(requester, memory_id)
        assert record is not None
        return record

    def update(
        self,
        requester: Actor,
        memory_id: str,
        *,
        memory_type: str,
        value: str,
        source: str,
        confidence: float = 1.0,
        sensitivity: str = "normal",
        consented: bool = False,
        confirmed: bool = False,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        """Update one record atomically while preserving its stable ID.

        The UI and tool layer use this method instead of delete-then-create,
        which avoids a transient missing-memory window and preserves audit and
        references held by callers.
        """

        current = self.get(requester, memory_id)
        if current is None:
            raise KeyError("Memory was not found")
        if not self._can_access(requester, current.owner_id, "memory.write"):
            raise PermissionError("Private memory access denied")
        if not confirmed:
            raise PermissionError("Memory updates require explicit confirmation")
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        value = value.strip()
        source = source.strip() or "user"
        if not value:
            raise ValueError("Memory value is required")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if sensitivity not in {"normal", "sensitive"}:
            raise ValueError("Sensitivity must be normal or sensitive")
        if sensitivity == "sensitive" and not consented:
            raise PermissionError("Sensitive memory requires explicit consent")
        if _credential_field(value) or contains_prohibited_secret(value):
            raise ValueError("Passwords, PINs, API keys, tokens, and credentials cannot be stored")

        now = datetime.now(UTC)
        try:
            with self.database.transaction() as connection:
                duplicate = connection.execute(
                    "SELECT memory_id FROM memory_records "
                    "WHERE owner_id=? AND memory_type=? AND memory_key=? "
                    "AND memory_id<>? AND deleted_at IS NULL",
                    (current.owner_id, memory_type, current.key, memory_id),
                ).fetchone()
                if duplicate:
                    raise ValueError("A memory with that type and key already exists")
                connection.execute(
                    "UPDATE memory_records SET memory_type=?, value=?, source=?, confidence=?, "
                    "sensitivity=?, consented=?, updated_at=?, expires_at=? WHERE memory_id=?",
                    (
                        memory_type,
                        value,
                        source,
                        confidence,
                        sensitivity,
                        int(consented),
                        now.isoformat(),
                        expires_at.isoformat() if expires_at else None,
                        memory_id,
                    ),
                )
        except Exception:
            raise
        record = self.get(requester, memory_id)
        if record is None:
            raise RuntimeError("Memory update could not be verified")
        return record

    def get(self, requester: Actor, memory_id: str) -> MemoryRecord | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_records WHERE memory_id=? AND deleted_at IS NULL",
                (memory_id,),
            ).fetchone()
        if row is None or not self._can_access(requester, row["owner_id"], "memory.read"):
            return None
        record = self._from_row(row)
        if record.expires_at and record.expires_at <= datetime.now(UTC):
            return None
        return record

    def list(
        self,
        requester: Actor,
        *,
        owner_id: str,
        memory_type: str | None = None,
        query: str = "",
        include_sensitive: bool = True,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        if not self._can_access(requester, owner_id, "memory.read"):
            return []
        clauses = ["owner_id=?", "deleted_at IS NULL", "(expires_at IS NULL OR expires_at>?)"]
        parameters: list[Any] = [owner_id, datetime.now(UTC).isoformat()]
        if memory_type:
            clauses.append("memory_type=?")
            parameters.append(memory_type)
        if query.strip():
            clauses.append("(memory_key LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\')")
            escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            parameters.extend([f"%{escaped}%", f"%{escaped}%"])
        if not include_sensitive:
            clauses.append("sensitivity='normal'")
        parameters.append(max(1, min(limit, 1000)))
        sql = "SELECT * FROM memory_records WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT ?"
        with self.database.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, requester: Actor, memory_id: str, *, confirmed: bool = False) -> bool:
        """Permanently remove one record after an explicit confirmation.

        A prior implementation only set ``deleted_at``.  That made a user
        deletion reversible by anyone with filesystem access to SQLite and
        left the plaintext in the main database/WAL indefinitely.  Explicit
        deletion now removes the row; callers that need expiry bookkeeping can
        still use :meth:`purge_expired` followed by :meth:`purge_deleted`.
        """

        record = self.get(requester, memory_id)
        if record is None:
            return False
        if not self._can_access(requester, record.owner_id, "memory.write"):
            raise PermissionError("Private memory access denied")
        if not confirmed:
            raise PermissionError("Memory deletion requires explicit confirmation")
        with self.database.transaction() as connection:
            result = connection.execute(
                "DELETE FROM memory_records WHERE memory_id=? AND deleted_at IS NULL",
                (memory_id,),
            )
        self._compact_after_delete()
        return result.rowcount == 1

    def purge_expired(self, requester: Actor, *, owner_id: str) -> int:
        if not self._can_access(requester, owner_id, "memory.write"):
            raise PermissionError("Private memory access denied")
        now = datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            result = connection.execute(
                "DELETE FROM memory_records "
                "WHERE owner_id=? AND deleted_at IS NULL AND expires_at IS NOT NULL AND expires_at<=?",
                (owner_id, now),
            )
        if result.rowcount:
            self._compact_after_delete()
        return result.rowcount

    def purge_deleted(self, requester: Actor, *, owner_id: str) -> int:
        """Permanently remove expired/tombstoned records for one owner."""

        if not self._can_access(requester, owner_id, "memory.write"):
            raise PermissionError("Private memory access denied")
        with self.database.transaction() as connection:
            result = connection.execute(
                "DELETE FROM memory_records WHERE owner_id=? AND deleted_at IS NOT NULL",
                (owner_id,),
            )
        self._compact_after_delete()
        return result.rowcount

    def _compact_after_delete(self) -> None:
        """Checkpoint SQLite's WAL so deleted plaintext is not retained there."""

        try:
            with self.database.connect() as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            # The row deletion is authoritative even if another process keeps
            # the WAL busy.  A later maintenance pass can checkpoint it.
            pass

    def export_json(self, requester: Actor, *, owner_id: str, destination: str | Path) -> Path:
        if requester.role is not Role.OWNER or requester.user_id != owner_id:
            raise PermissionError("Only the owner can export private memory")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = [item.to_dict() for item in self.list(requester, owner_id=owner_id, limit=1000)]
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps({"memories": records}, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
        return destination
