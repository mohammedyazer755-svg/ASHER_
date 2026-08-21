"""Tool adapters for consent-aware memory CRUD and retrieval."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from asher.memory.retrieval import MemoryRetriever
from asher.memory.store import MemoryStore
from asher.security.policy import ToolPolicy
from asher.tools.registry import ToolContext, ToolDefinition, successful_result
from asher.types import Evidence, RiskLevel, ToolResult


class MemoryTools:
    def __init__(self, store: MemoryStore, retriever: MemoryRetriever) -> None:
        self.store = store
        self.retriever = retriever

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                name="memory.search", description="Retrieve a few relevant local memories for the authenticated owner.",
                input_schema={"type":"object", "properties":{"query":{"type":"string","minLength":1,"maxLength":500}}, "required":["query"], "additionalProperties":False},
                policy=ToolPolicy("private_memory", RiskLevel.HARMLESS_LOCAL), timeout_seconds=5,
                handler=self.search, preview=lambda args: ("private memory", f"Search local memory for {args['query']}", {"query": args["query"]}), idempotent=True,
            ),
            ToolDefinition(
                name="memory.put", description="Create or update an explicitly consented local memory.",
                input_schema={"type":"object", "properties":{"memory_type":{"type":"string","minLength":1,"maxLength":80},"key":{"type":"string","minLength":1,"maxLength":200},"value":{"type":"string","minLength":1,"maxLength":4000},"sensitivity":{"type":"string","enum":["normal","sensitive"]}}, "required":["memory_type","key","value","sensitivity"], "additionalProperties":False},
                policy=ToolPolicy("private_memory_write", RiskLevel.SENSITIVE), timeout_seconds=8,
                handler=self.put, preview=lambda args: (args["key"], f"Save or update the memory named {args['key']}", {"memory_type": args["memory_type"], "key": args["key"], "sensitivity": args["sensitivity"], "value": args["value"]}),
            ),
            ToolDefinition(
                name="memory.delete", description="Delete one local memory after an exact preview and strong authentication.",
                input_schema={"type":"object", "properties":{"memory_id":{"type":"string","minLength":1,"maxLength":100}}, "required":["memory_id"], "additionalProperties":False},
                policy=ToolPolicy("private_memory_write", RiskLevel.SENSITIVE), timeout_seconds=8,
                handler=self.delete, preview=lambda args: (args["memory_id"], f"Delete the selected local memory {args['memory_id']}", {"memory_id": args["memory_id"]}),
            ),
        )

    def search(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        owner_id = context.session.actor.user_id
        results = self.retriever.retrieve(context.session.actor, owner_id=owner_id, query=arguments["query"], include_sensitive=False)
        # Values remain in the trusted response channel, never in audit evidence.
        summaries = [{"memory_id": item.record.memory_id, "key": item.record.key, "type": item.record.memory_type, "score": round(item.score, 3)} for item in results]
        return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Found {len(results)} relevant memories.", (Evidence("memory_retrieval", "Relevant local records were selected", {"records": summaries}),), dry_run=context.dry_run)

    def put(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            record = self.store.put(
                context.session.actor,
                owner_id=context.session.actor.user_id,
                memory_type=arguments["memory_type"], key=arguments["key"], value=arguments["value"],
                source="user_confirmed", sensitivity=arguments["sensitivity"],
                consented=arguments["sensitivity"] == "normal", confirmed=True,
            )
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], f"Saved memory {record.key}.", (Evidence("memory_persisted", "The memory record was committed locally", {"memory_id": record.memory_id, "key": record.key}),))
        except (ValueError, PermissionError) as error:
            return _failure(context, "memory_rejected", str(error))

    def delete(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            deleted = self.store.delete(context.session.actor, arguments["memory_id"], confirmed=True)
            if not deleted:
                return _failure(context, "memory_not_found", "The memory no longer exists.")
            return successful_result(context.metadata["call_id"], context.metadata["tool_name"], "Memory deleted.", (Evidence("memory_deleted", "The record was permanently removed locally", {"memory_id": arguments["memory_id"]}),))
        except PermissionError as error:
            return _failure(context, "memory_denied", str(error))


def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(call_id=context.metadata["call_id"], tool_name=context.metadata["tool_name"], success=False, status="failed", message=message, error_code=code, started_at=now, completed_at=now)
