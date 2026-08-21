"""Strict, provider-neutral plan schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from asher.tools.registry import SchemaValidationError, ToolRegistry, validate_arguments
from asher.core.redaction import redact_text
from asher.types import ExecutionPlan, PlanStep, ToolCall


@dataclass(frozen=True)
class ProposedPlan:
    goal: str
    steps: tuple[PlanStep, ...] = ()
    response: str = ""
    provider: str = "deterministic"
    offline: bool = True


class PlanValidationError(ValueError):
    pass


def validate_plan_payload(payload: dict[str, Any], registry: ToolRegistry, *, provider: str, offline: bool) -> ProposedPlan:
    if not isinstance(payload, dict):
        raise PlanValidationError("Plan must be an object")
    goal = redact_text(str(payload.get("goal", "")).strip())
    response = redact_text(str(payload.get("response", "")).strip())
    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list) or len(raw_steps) > 8:
        raise PlanValidationError("A plan must contain between zero and eight steps")
    steps: list[PlanStep] = []
    known_step_ids: set[str] = set()
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise PlanValidationError(f"Step {index + 1} is not an object")
        name = raw.get("tool") or raw.get("name")
        arguments = raw.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise PlanValidationError(f"Step {index + 1} has no tool")
        if not isinstance(arguments, dict):
            raise PlanValidationError(f"Step {index + 1} arguments must be an object")
        step_id = raw.get("id")
        if not isinstance(step_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", step_id):
            raise PlanValidationError(f"Step {index + 1} has an invalid id")
        if step_id in known_step_ids:
            raise PlanValidationError(f"Duplicate step id: {step_id}")
        definition = registry.get(name.strip())
        if definition is None:
            raise PlanValidationError(f"Unknown tool: {name}")
        try:
            validate_arguments(definition.input_schema, arguments)
        except SchemaValidationError as error:
            raise PlanValidationError(f"Invalid arguments for {name}: {error}") from error
        dependencies = raw.get("depends_on", [])
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            raise PlanValidationError(f"Invalid dependencies for {name}")
        unknown_dependencies = set(dependencies) - known_step_ids
        if unknown_dependencies:
            raise PlanValidationError(
                f"Step {step_id} depends on unknown or future steps: "
                f"{', '.join(sorted(unknown_dependencies))}"
            )
        description = redact_text(str(raw.get("description", name)).strip())
        steps.append(
            PlanStep(
                ToolCall(name.strip(), arguments),
                description,
                tuple(dependencies),
                step_id,
            )
        )
        known_step_ids.add(step_id)
    return ProposedPlan(goal=goal, steps=tuple(steps), response=response, provider=provider, offline=offline)


def as_execution_plan(plan: ProposedPlan) -> ExecutionPlan:
    return ExecutionPlan(goal=plan.goal, steps=plan.steps)


PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "goal": {"type": "string"},
        "response": {"type": "string"},
        "steps": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                    "description": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "tool", "arguments", "description", "depends_on"],
            },
        },
    },
    "required": ["goal", "response", "steps"],
}
