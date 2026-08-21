"""Minimal-context local retrieval without transmitting the full memory store."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from asher.memory.store import MemoryRecord, MemoryStore
from asher.types import Actor


TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.casefold()))


class MemoryRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        requester: Actor,
        *,
        owner_id: str,
        query: str,
        limit: int = 5,
        include_sensitive: bool = False,
    ) -> list[RetrievedMemory]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        candidates = self.store.list(
            requester,
            owner_id=owner_id,
            include_sensitive=include_sensitive,
            limit=500,
        )
        now = datetime.now(UTC)
        scored: list[RetrievedMemory] = []
        for record in candidates:
            record_tokens = _tokens(f"{record.key} {record.value} {record.memory_type}")
            overlap = len(query_tokens & record_tokens) / max(1, len(query_tokens | record_tokens))
            age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400)
            recency = math.exp(-age_days / 180.0)
            score = 0.75 * overlap + 0.15 * recency + 0.10 * record.confidence
            if overlap > 0:
                scored.append(RetrievedMemory(record, score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, min(limit, 10))]

