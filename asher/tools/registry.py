"""Central typed tool registry and safe execution boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Callable

from asher.core.cancellation import CancelledError, CancellationToken
from asher.security.audit import AuditLog
from asher.security.confirmations import ConfirmationStore
from asher.security.policy import DecisionKind, PolicyEngine, ToolPolicy
from asher.types import Evidence, SessionContext, ToolCall, ToolResult


@dataclass(frozen=True)
class ToolContext:
    session: SessionContext
    cancellation: CancellationToken
    dry_run: bool
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any], ToolContext], ToolResult]
PreviewBuilder = Callable[[dict[str, Any]], tuple[str, str, dict[str, Any]]]
Verifier = Callable[[dict[str, Any], ToolContext, ToolResult], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    policy: ToolPolicy
    timeout_seconds: float
    handler: ToolHandler
    preview: PreviewBuilder
    verifier: Verifier | None = None
    # Automatic retry is safe only when repeating the operation cannot create
    # another side effect (for example, a local read/list operation).
    idempotent: bool = False

    def function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "strict": True,
        }


class SchemaValidationError(ValueError):
    pass


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }.get(expected, False)


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise SchemaValidationError("Tool schema must describe an object")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = required - arguments.keys()
    if missing:
        raise SchemaValidationError(f"Missing required fields: {', '.join(sorted(missing))}")
    if schema.get("additionalProperties") is False:
        unexpected = arguments.keys() - properties.keys()
        if unexpected:
            raise SchemaValidationError(f"Unexpected fields: {', '.join(sorted(unexpected))}")

    for key, value in arguments.items():
        rule = properties.get(key)
        if rule is None:
            continue
        expected = rule.get("type")
        if expected and not _type_matches(value, expected):
            raise SchemaValidationError(f"{key} must be {expected}")
        if "enum" in rule and value not in rule["enum"]:
            raise SchemaValidationError(f"{key} is not an allowed value")
        if isinstance(value, str):
            if len(value) < rule.get("minLength", 0):
                raise SchemaValidationError(f"{key} is too short")
            if len(value) > rule.get("maxLength", 1_000_000):
                raise SchemaValidationError(f"{key} is too long")
        if isinstance(value, list):
            if len(value) > rule.get("maxItems", 10_000):
                raise SchemaValidationError(f"{key} has too many items")


class ToolRegistry:
    def __init__(
        self,
        *,
        policy: PolicyEngine,
        confirmations: ConfirmationStore,
        audit: AuditLog,
    ) -> None:
        self.policy = policy
        self.confirmations = confirmations
        self.audit = audit
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        if definition.timeout_seconds <= 0:
            raise ValueError("Tool timeout must be positive")
        validate_arguments(definition.input_schema, {
            key: _sample_value(rule)
            for key, rule in definition.input_schema.get("properties", {}).items()
            if key in definition.input_schema.get("required", [])
        })
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def schemas_for(self, session: SessionContext) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for definition in self._tools.values():
            decision = self.policy.evaluate(definition.policy, session)
            if decision.kind is not DecisionKind.DENY:
                schemas.append(definition.function_schema())
        return schemas

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        started = datetime.now(UTC)
        definition = self._tools.get(call.tool_name)
        if definition is None:
            return _failure(call, "unknown_tool", "The requested tool is not registered", started)

        try:
            validate_arguments(definition.input_schema, call.arguments)
        except SchemaValidationError as error:
            return _failure(call, "invalid_arguments", str(error), started)

        context.cancellation.raise_if_cancelled()

        # Never trust a Confirmation carried by a ToolCall on its own. It must
        # be the exact, unexpired, one-time approval minted by this registry's
        # ConfirmationStore for this actor/session and operation preview.
        if call.confirmation is not None:
            target, effect, preview = definition.preview(call.arguments)
            try:
                canonical = self.confirmations.consume(
                    call.confirmation,
                    context.session,
                    tool_name=definition.name,
                    target=target,
                    effect=effect,
                    preview=preview,
                    risk=definition.policy.risk,
                    arguments=call.arguments,
                )
            except (PermissionError, ValueError) as error:
                self.audit.append(
                    "confirmation_denied",
                    actor_id=context.session.actor.user_id,
                    session_id=context.session.session_id,
                    tool_name=definition.name,
                    target=target,
                    outcome="invalid_confirmation",
                )
                return _failure(call, "permission_denied", str(error), started)
            call = ToolCall(
                call.tool_name,
                dict(call.arguments),
                call_id=call.call_id,
                confirmation=canonical,
            )

        decision = self.policy.evaluate(definition.policy, context.session, call.confirmation)

        if decision.kind in {DecisionKind.REQUIRE_CONFIRMATION, DecisionKind.REQUIRE_STRONG_AUTH}:
            target, effect, preview = definition.preview(call.arguments)
            pending = self.confirmations.create(
                context.session,
                tool_name=definition.name,
                target=target,
                effect=effect,
                preview=preview,
                risk=definition.policy.risk,
                arguments=call.arguments,
            )
            self.audit.append(
                "confirmation_requested",
                actor_id=context.session.actor.user_id,
                session_id=context.session.session_id,
                tool_name=definition.name,
                target=target,
                outcome=decision.kind.value,
                details={"confirmation_id": pending.confirmation_id, "risk": int(definition.policy.risk)},
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=definition.name,
                success=False,
                status="awaiting_strong_auth" if decision.kind is DecisionKind.REQUIRE_STRONG_AUTH else "awaiting_confirmation",
                message=decision.reason,
                evidence=(Evidence("confirmation_preview", effect, {"confirmation_id": pending.confirmation_id, "target": target, "preview": preview}),),
                error_code=decision.kind.value,
                started_at=started,
                completed_at=datetime.now(UTC),
            )

        if decision.kind is DecisionKind.DENY:
            self.audit.append(
                "tool_denied",
                actor_id=context.session.actor.user_id,
                session_id=context.session.session_id,
                tool_name=definition.name,
                outcome=decision.reason,
            )
            return _failure(call, "permission_denied", decision.reason, started)

        operation_token = CancellationToken()
        unlink_cancellation = context.cancellation.add_callback(operation_token.cancel)
        execution_context = replace(
            context,
            cancellation=operation_token,
            metadata={**context.metadata, "call_id": call.call_id, "tool_name": definition.name},
        )
        executor: ThreadPoolExecutor | None = None
        timed_out = False
        try:
            # Do not use the executor as a context manager here: its
            # ``__exit__`` waits for a non-cooperative handler, defeating the
            # advertised timeout and making emergency stop feel frozen.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"asher-{definition.name}")
            future = executor.submit(definition.handler, call.arguments, execution_context)
            try:
                result = future.result(timeout=definition.timeout_seconds)
            except FutureTimeout:
                timed_out = True
                operation_token.cancel(f"Tool timed out: {definition.name}")
                future.cancel()
                message = (
                    f"{definition.name} timed out and may be retried safely"
                    if definition.idempotent
                    else f"{definition.name} timed out; completion is unknown, so it will not be retried"
                )
                result = _failure(
                    call,
                    "timeout",
                    message,
                    started,
                    retryable=definition.idempotent,
                )
        except CancelledError as error:
            result = _failure(call, "cancelled", str(error), started)
        except Exception as error:
            result = _failure(call, "tool_error", f"{definition.name} failed safely: {type(error).__name__}", started)
        finally:
            unlink_cancellation()
            if executor is not None:
                executor.shutdown(wait=not timed_out, cancel_futures=True)

        if result.success and definition.verifier is not None:
            try:
                result = definition.verifier(call.arguments, execution_context, result)
            except Exception as error:
                result = _failure(call, "verification_error", f"Verification failed: {type(error).__name__}", started)

        if result.success and not result.evidence:
            result = _failure(call, "unverified", "The action was attempted but produced no verification evidence", started)

        target, _, _ = definition.preview(call.arguments)
        self.audit.append(
            "tool_completed" if result.success else "tool_failed",
            actor_id=context.session.actor.user_id,
            session_id=context.session.session_id,
            tool_name=definition.name,
            target=target,
            outcome=result.status,
            details={"error_code": result.error_code, "dry_run": result.dry_run},
        )
        return result


def _sample_value(rule: dict[str, Any]) -> Any:
    if "enum" in rule:
        return rule["enum"][0]
    return {
        "string": "x" * max(1, rule.get("minLength", 1)),
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "object": {},
        "array": [],
    }.get(rule.get("type"), None)


def _failure(
    call: ToolCall,
    code: str,
    message: str,
    started: datetime,
    *,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        success=False,
        status="cancelled" if code == "cancelled" else "failed",
        message=message,
        error_code=code,
        retryable=retryable,
        started_at=started,
        completed_at=datetime.now(UTC),
    )


def successful_result(
    call_id: str,
    tool_name: str,
    message: str,
    evidence: tuple[Evidence, ...],
    *,
    status: str = "verified",
    dry_run: bool = False,
) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        success=True,
        status=status,
        message=message,
        evidence=evidence,
        dry_run=dry_run,
        started_at=now,
        completed_at=now,
    )
