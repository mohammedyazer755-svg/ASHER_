"""Typed records for ASHER PreferenceCore training data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


FEEDBACK_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "accept_response": ("overall_preference",),
    "reject_response": ("overall_preference",),
    "shorter": ("brevity",),
    "more_detailed": ("brevity", "failure_detail"),
    "more_direct": ("directness",),
    "ask_less": ("clarification_tendency",),
    "ask_more": ("clarification_tendency",),
    "suggest_more": ("proactivity",),
    "suggest_less": ("proactivity",),
    "preferred_reply": ("language_style", "overall_preference"),
}

FEEDBACK_KINDS = frozenset(FEEDBACK_DIMENSIONS)


@dataclass(frozen=True)
class PreferenceEvent:
    """One explicit owner-labelled preference example.

    Raw interaction text is kept local in SQLite only after the owner opts in
    and explicitly labels the immediately preceding ASHER response.  Security
    decisions are deliberately not represented as trainable labels.
    """

    event_id: str
    owner_id: str
    session_id: str
    source_hash: str
    user_text: str
    assistant_text: str
    preferred_text: str | None
    feedback_kind: str
    dimensions: tuple[str, ...]
    context: dict[str, Any]
    created_at: datetime
