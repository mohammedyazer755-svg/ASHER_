"""Local/API planner providers with strict structured output and safe fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from asher.brain.plans import PLAN_JSON_SCHEMA, ProposedPlan, validate_plan_payload
from asher.core.redaction import redact_text
from asher.tools.registry import ToolRegistry


class ProviderError(RuntimeError):
    pass


class PlannerProvider(Protocol):
    name: str
    offline: bool

    def plan(self, command: str, *, context: dict[str, Any], tool_schemas: list[dict[str, Any]], registry: ToolRegistry) -> ProposedPlan: ...


SYSTEM_INSTRUCTIONS = """
You are the planning component of ASHER. Return only the supplied strict JSON
schema. Choose only tools listed in the schema. Never execute shell commands,
invent tools, expose secrets, or claim that a tool succeeded. Preserve contact
names and message text. Use the smallest ordered plan. Consequential tools
must remain separate steps so the local policy layer can preview and confirm.
Give every step a unique short id and let depends_on reference only earlier
step ids. For a request that only needs a conversational answer, put the
complete answer in response and set steps to an empty array. Never invent a
placeholder tool such as none, no_tool, null, or n/a. Use only the minimal
context supplied by the application.
""".strip()


class OpenAIResponsesProvider:
    name = "openai_responses"
    offline = False

    def __init__(self, model: str = "gpt-5.6-luna", *, reasoning_model: str = "gpt-5.6-terra", timeout: float = 25.0, client: Any | None = None) -> None:
        self.model = model
        self.reasoning_model = reasoning_model
        self.timeout = timeout
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]
            except ImportError as error:
                raise ProviderError("OpenAI Python SDK is not installed") from error
            if not os.getenv("OPENAI_API_KEY", "").strip():
                raise ProviderError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=self.timeout, max_retries=0)
        return self._client

    def plan(self, command: str, *, context: dict[str, Any], tool_schemas: list[dict[str, Any]], registry: ToolRegistry) -> ProposedPlan:
        complexity = len(command.split()) > 14 or any(word in command.casefold() for word in ("why", "emotion", "frustrated", "several", "plan"))
        model = self.reasoning_model if complexity else self.model
        prompt_context = {key: value for key, value in context.items() if key != "session"}
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": f"Minimal context: {json.dumps(prompt_context, ensure_ascii=False, default=str)}\nUser request: {redact_text(command)}",
            "text": {"format": {"type": "json_schema", "name": "asher_plan", "schema": PLAN_JSON_SCHEMA, "strict": True}},
            "tools": tool_schemas,
            "store": False,
        }
        # GPT-5.6 supports intentional effort levels; low keeps routine voice
        # requests responsive while Terra receives medium effort for complex goals.
        kwargs["reasoning"] = {"effort": "medium" if complexity else "low"}
        try:
            response = self.client.responses.create(**kwargs)
            raw = getattr(response, "output_text", "") or ""
            if not raw and hasattr(response, "model_dump"):
                raw = _extract_output_text(response.model_dump())
            payload = json.loads(raw)
            return validate_plan_payload(payload, registry, provider=self.name, offline=False)
        except Exception as error:
            if isinstance(error, ProviderError):
                raise
            raise ProviderError(f"OpenAI planner failed: {type(error).__name__}") from error



_NO_TOOL_SENTINELS = {"none", "no_tool", "no-tool", "null", "n/a"}


def _normalize_local_no_tool_response(payload: Any) -> Any:
    """Collapse one harmless local-model no-tool placeholder into no steps.

    The strict validator remains authoritative for every real tool call. This
    adapter only handles the common structured-output mistake where a local
    model emits a single `tool: none` placeholder for a pure conversation.
    """

    if not isinstance(payload, dict):
        return payload
    response = payload.get("response")
    raw_steps = payload.get("steps")
    if not isinstance(response, str) or not response.strip():
        return payload
    if not isinstance(raw_steps, list) or len(raw_steps) != 1:
        return payload
    step = raw_steps[0]
    if not isinstance(step, dict):
        return payload
    name = step.get("tool") or step.get("name")
    if not isinstance(name, str) or name.strip().casefold() not in _NO_TOOL_SENTINELS:
        return payload
    if step.get("arguments", {}) != {}:
        return payload
    if step.get("depends_on", []) != []:
        return payload

    normalized = dict(payload)
    normalized["steps"] = []
    return normalized


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:9b",
        timeout: float = 60.0,
        *,
        offline: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        host = (urlparse(self.base_url).hostname or "").casefold().strip("[]")
        self.offline = host in {"localhost", "127.0.0.1", "::1"} if offline is None else bool(offline)

    def plan(self, command: str, *, context: dict[str, Any], tool_schemas: list[dict[str, Any]], registry: ToolRegistry) -> ProposedPlan:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": f"Minimal context: {json.dumps({key: value for key, value in context.items() if key != 'session'}, ensure_ascii=False, default=str)}\nUser request: {redact_text(command)}"},
            ],
            "format": PLAN_JSON_SCHEMA,
            "stream": False,
            # Qwen3.5 supports a thinking channel. Planning here already uses a
            # strict local schema, so disable long hidden reasoning for the
            # routine planner path to keep voice/text latency predictable.
            "think": False,
            "options": {"temperature": 0, "num_predict": 700, "num_ctx": 8192},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload.get("message", {}).get("content", "")
            if not content:
                raise ProviderError("Ollama returned no message content")
            parsed = _normalize_local_no_tool_response(json.loads(content))
            return validate_plan_payload(parsed, registry, provider=self.name, offline=True)
        except ProviderError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ProviderError(f"Ollama planner unavailable: {type(error).__name__}") from error
        except Exception as error:
            raise ProviderError(f"Ollama planner failed: {type(error).__name__}") from error


@dataclass
class HybridPlanner:
    deterministic: Any
    registry: ToolRegistry
    openai: OpenAIResponsesProvider | None = None
    ollama: OllamaProvider | None = None
    online_enabled: bool = True
    last_provider: str = "deterministic"
    offline: bool = True

    def plan(self, command: str, *, context: dict[str, Any] | None = None) -> ProposedPlan:
        context = context or {}
        deterministic = self.deterministic.plan(
            command,
            last_app=str(context.get("last_app", "")),
            last_contact=str(context.get("last_contact", "")),
            last_search_query=str(context.get("last_search_query", "")),
        )
        if deterministic is not None:
            self.last_provider = "deterministic"
            self.offline = True
            return deterministic

        errors: list[str] = []
        session = context.get("session")
        schemas = self.registry.schemas_for(session) if session is not None else []
        providers: list[PlannerProvider] = []
        if self.openai is not None and self.online_enabled:
            providers.append(self.openai)
        if self.ollama is not None:
            providers.append(self.ollama)
        for provider in providers:
            try:
                result = provider.plan(command, context=context, tool_schemas=schemas, registry=self.registry)
                self.last_provider = provider.name
                self.offline = provider.offline
                return result
            except (ProviderError, TypeError) as error:
                errors.append(str(error))
        self.last_provider = "fallback"
        self.offline = True
        return ProposedPlan(
            goal=command,
            response="The configured planner is unavailable. I can still handle supported local commands, but I won’t guess at this request.",
            provider="fallback",
            offline=True,
        )


def _extract_output_text(payload: dict[str, Any]) -> str:
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return str(content.get("text", ""))
    return ""
