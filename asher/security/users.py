"""User and per-capability permission persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from asher.storage import Database
from asher.types import Actor, Role


DEFAULT_TRUSTED_CAPABILITIES = frozenset({"conversation", "public_web", "open_app", "media"})


class UserStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_owner(self, display_name: str) -> Actor:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT user_id FROM users WHERE role='owner' AND active=1 LIMIT 1"
            ).fetchone()
            now = datetime.now(UTC).isoformat()
            if row:
                user_id = row["user_id"]
                connection.execute(
                    "UPDATE users SET display_name=?, updated_at=? WHERE user_id=?",
                    (display_name, now, user_id),
                )
            else:
                user_id = uuid4().hex
                connection.execute(
                    "INSERT INTO users(user_id, display_name, role, active, created_at, updated_at) "
                    "VALUES(?, ?, 'owner', 1, ?, ?)",
                    (user_id, display_name, now, now),
                )
        actor = self.get(user_id)
        assert actor is not None
        return actor

    def create(self, display_name: str, role: Role, permissions: set[str] | None = None) -> Actor:
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Display name is required")
        if role is Role.OWNER and any(user.role is Role.OWNER for user in self.list_active()):
            raise ValueError("An active owner already exists")

        user_id = uuid4().hex
        now = datetime.now(UTC).isoformat()
        granted = permissions if permissions is not None else (
            set(DEFAULT_TRUSTED_CAPABILITIES) if role is Role.TRUSTED else set()
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO users(user_id, display_name, role, active, created_at, updated_at) "
                "VALUES(?, ?, ?, 1, ?, ?)",
                (user_id, display_name, role.value, now, now),
            )
            connection.executemany(
                "INSERT INTO user_permissions(user_id, capability) VALUES(?, ?)",
                [(user_id, capability) for capability in sorted(granted)],
            )
        actor = self.get(user_id)
        assert actor is not None
        return actor

    def get(self, user_id: str) -> Actor | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT user_id, display_name, role FROM users WHERE user_id=? AND active=1",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            permissions = frozenset(
                item["capability"]
                for item in connection.execute(
                    "SELECT capability FROM user_permissions WHERE user_id=?",
                    (user_id,),
                ).fetchall()
            )
        return Actor(
            user_id=row["user_id"],
            display_name=row["display_name"],
            role=Role(row["role"]),
            permissions=permissions,
        )

    def list_active(self) -> list[Actor]:
        with self.database.connect() as connection:
            ids = [
                row["user_id"]
                for row in connection.execute(
                    "SELECT user_id FROM users WHERE active=1 ORDER BY role, display_name"
                ).fetchall()
            ]
        return [actor for user_id in ids if (actor := self.get(user_id)) is not None]

    def set_permissions(self, user_id: str, permissions: set[str]) -> Actor:
        actor = self.get(user_id)
        if actor is None:
            raise KeyError(user_id)
        if actor.role is Role.GUEST and permissions:
            raise ValueError("Guest permissions cannot be elevated")
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM user_permissions WHERE user_id=?", (user_id,))
            connection.executemany(
                "INSERT INTO user_permissions(user_id, capability) VALUES(?, ?)",
                [(user_id, capability) for capability in sorted(permissions)],
            )
            connection.execute(
                "UPDATE users SET updated_at=? WHERE user_id=?",
                (datetime.now(UTC).isoformat(), user_id),
            )
        updated = self.get(user_id)
        assert updated is not None
        return updated

    def revoke(self, user_id: str) -> bool:
        actor = self.get(user_id)
        if actor is None:
            return False
        if actor.role is Role.OWNER:
            raise ValueError("The owner cannot be revoked; transfer ownership explicitly")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE users SET active=0, updated_at=? WHERE user_id=?",
                (datetime.now(UTC).isoformat(), user_id),
            )
        return True


def guest_actor() -> Actor:
    return Actor(user_id="guest", display_name="Guest", role=Role.GUEST)

