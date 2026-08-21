"""Backward-compatible legacy memory store with atomic, typed persistence.

New code uses :class:`asher.memory.store.MemoryStore`. This adapter remains for
older modules and never accepts credential-like values.
"""

from __future__ import annotations

import json
import os
import threading
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from asher.core.redaction import contains_prohibited_secret


BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = os.getenv("ASHER_LEGACY_MEMORY_FILE", str(BASE_DIR / "memory.json"))
_LOCK = threading.RLock()


def _credential_field(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return normalized in {
        "password", "passcode", "pin", "api_key", "api_token",
        "access_token", "refresh_token", "secret", "secret_token", "credential",
    }


def _empty() -> dict[str, list[dict[str, str]]]:
    return {"memory": []}


def load_memory() -> dict[str, list[dict[str, str]]]:
    path = Path(MEMORY_FILE)
    if not path.is_file():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty()
    if not isinstance(data, dict) or not isinstance(data.get("memory"), list):
        return _empty()
    records: list[dict[str, str]] = []
    for item in data["memory"]:
        if not isinstance(item, dict):
            continue
        key, value = item.get("key"), item.get("value")
        if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip():
            records.append({"key": key.strip(), "value": value.strip()})
    return {"memory": records}


def save_memory(data: dict) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("memory"), list):
        return False
    clean = _empty()
    for item in data["memory"]:
        if not isinstance(item, dict):
            continue
        key, value = item.get("key"), item.get("value")
        if not isinstance(key, str) or not isinstance(value, str) or not key.strip() or not value.strip():
            continue
        if _credential_field(key) or _credential_field(value) or contains_prohibited_secret(key) or contains_prohibited_secret(value):
            return False
        clean["memory"].append({"key": key.strip(), "value": value.strip()})
    path = Path(MEMORY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(clean, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            return True
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except UnboundLocalError:
                pass
            return False


def remember(key: str, value: str) -> bool:
    key, value = str(key).strip(), str(value).strip()
    if not key or not value or _credential_field(key) or _credential_field(value) or contains_prohibited_secret(key) or contains_prohibited_secret(value):
        return False
    data = load_memory()
    for item in data["memory"]:
        if item["key"].casefold() == key.casefold():
            item["key"] = key
            item["value"] = value
            return save_memory(data)
    data["memory"].append({"key": key, "value": value})
    return save_memory(data)


def get_memory(key: str):
    target = str(key).casefold().strip()
    for item in load_memory()["memory"]:
        if item["key"].casefold() == target:
            return item["value"]
    return None


def forget_memory(key: str) -> bool:
    target = str(key).casefold().strip()
    data = load_memory()
    before = len(data["memory"])
    data["memory"] = [item for item in data["memory"] if item["key"].casefold() != target]
    return before != len(data["memory"]) and save_memory(data)
