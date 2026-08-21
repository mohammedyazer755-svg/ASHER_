"""SQLite lifecycle and schema migrations for local ASHER state."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


SCHEMA_VERSION = 1


class _ManagedConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the handle.

    ``sqlite3.Connection`` commits/rolls back on ``with`` but intentionally
    does not close.  On Windows that leaves the database locked until garbage
    collection, which is especially painful for temporary test runtimes and
    clean shutdown.  Keeping the familiar connection API while closing at the
    context boundary makes every existing ``with database.connect()`` call
    deterministic.
    """

    def __enter__(self) -> "_ManagedConnection":
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._migration_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, factory=_ManagedConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        with self._migration_lock, self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('owner','trusted','guest')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_permissions (
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    capability TEXT NOT NULL,
                    PRIMARY KEY(user_id, capability)
                );

                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('normal','sensitive')),
                    consented INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    deleted_at TEXT,
                    UNIQUE(owner_id, memory_type, memory_key)
                );

                CREATE INDEX IF NOT EXISTS memory_owner_active
                ON memory_records(owner_id, deleted_at, memory_type);

                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    completed_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
