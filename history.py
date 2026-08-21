"""Disabled legacy plaintext history adapter.

ASHER's active runtime keeps typed, redacted audit metadata and bounded working
memory in its private runtime directory. The old history.json is not read,
printed, or appended by default, preventing credentials and private message
previews from leaking into a global plaintext transcript.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HISTORY_FILE = Path(__file__).resolve().parent / "history.json"
EMPTY_HISTORY = {"chat": []}


def load_history() -> dict[str, list[dict[str, str]]]:
    # Explicitly return an empty compatibility view; existing legacy data is
    # left untouched on disk and never surfaced through the new application.
    return {"chat": []}


def save_history(_data: Any) -> bool:
    return False


def add_chat(_sender: str, _message: str) -> bool:
    return False


def show_history() -> None:
    print("Legacy plaintext history is disabled; use the redacted Activity view.")


def clear_history() -> bool:
    # Do not delete a user's old file implicitly.
    return False

