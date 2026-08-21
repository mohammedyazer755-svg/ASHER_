"""Cancellable, resumable plan execution with confirmation checkpoints."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from asher.core.cancellation import CancelledError, CancellationToken, EmergencyStop
from asher.core.state import AssistantState, StateStore
from asher.security.audit import AuditLog
from asher.security.confirmations import ConfirmationStore
from asher.tools.registry import ToolContext, ToolRegistry
from asher.types import AuthMethod, ExecutionPlan, SessionContext, ToolCall, ToolResult


@dataclass(frozen=True)
class ExecutionUpdate:
    status: str
    message: str
    plan_id: str | None = None
    current_step: int = 0
    total_steps: int = 0
    result: ToolResult | None = None
    confirmation_id: str | None = None


@dataclass
class ActiveExecution:
    plan: ExecutionPlan
    session: SessionContext
    token: CancellationToken
    dry_run: bool
    next_index: int = 0
    results: list[ToolResult] = field(default_factory=list)
    attempts: dict[int, int] = field(default_factory=dict)
    waiting_confirmation_id: str | None = None


class AgentLoop:
    def __init__(
        self,
        registry: ToolRegistry,
        confirmations: ConfirmationStore,
        audit: AuditLog,
        *,
        emergency_stop: EmergencyStop | None = None,
        states: StateStore | None = None,
        max_retries: int = 1,
    ) -> None:
        self.registry = registry
        self.confirmations = confirmations
        self.audit = audit
        self.emergency_stop = emergency_stop or EmergencyStop()
        self.states = states or StateStore()
        self.max_retries = max(0, max_retries)
        self._lock = threading.RLock()
        self._active: ActiveExecution | None = None

    @property
    def active(self) -> ActiveExecution | None:
        with self._lock:
            return self._active

    def start(self, plan: ExecutionPlan, session: SessionContext, *, dry_run: bool = True) -> list[ExecutionUpdate]:
        if not plan.steps:
            return [ExecutionUpdate("complete", "No tools were needed.", plan.plan_id)]
        with self._lock:
            if self._active is not None:
                return [ExecutionUpdate("busy", "Another plan is already running.", plan.plan_id)]
            token = CancellationToken()
            self.emergency_stop.register(token)
            self._active = ActiveExecution(plan=plan, session=session, token=token, dry_run=dry_run)
        self.audit.append("plan_started", actor_id=session.actor.user_id, session_id=session.session_id, details={"plan_id": plan.plan_id, "steps": len(plan.steps)})
        self.states.transition(AssistantState.EXECUTING, "Executing the approved plan", plan_id=plan.plan_id)
        return self._run_until_pause()

    def _run_until_pause(self) -> list[ExecutionUpdate]:
        updates: list[ExecutionUpdate] = []
        while True:
            with self._lock:
                active = self._active
                if active is None:
                    return updates
                if active.token.cancelled:
                    updates.append(self._finish_cancelled(active))
                    return updates
                if active.next_index >= len(active.plan.steps):
                    updates.append(self._finish_success(active))
                    return updates
                index = active.next_index
                step = active.plan.steps[index]

            if step.depends_on:
                completed_ids = {
                    item.step_id
                    for item in active.plan.steps[: active.next_index]
                }
                if not set(step.depends_on).issubset(completed_ids):
                    updates.append(
                        self._finish_failure(
                            active,
                            "A plan dependency was not completed.",
                        )
                    )
                    return updates

            attempt = active.attempts.get(index, 0)
            active.attempts[index] = attempt + 1
            context = ToolContext(
                session=active.session,
                cancellation=active.token,
                dry_run=active.dry_run,
                metadata={"plan_id": active.plan.plan_id, "step_index": index},
            )
            try:
                self.states.transition(
                    AssistantState.EXECUTING,
                    step.description or f"Executing {step.call.tool_name}",
                    plan_id=active.plan.plan_id,
                    step_id=step.step_id,
                    step_index=index,
                )
                result = self.registry.execute(step.call, context)
            except CancelledError as error:
                updates.append(self._finish_cancelled(active, str(error)))
                return updates
            if active.token.cancelled or self.emergency_stop.latched:
                updates.append(self._finish_cancelled(active, active.token.reason or "Emergency stop activated"))
                return updates
            active.results.append(result)
            self.states.transition(
                AssistantState.OBSERVING,
                "Checking the tool result",
                plan_id=active.plan.plan_id,
                step_id=step.step_id,
                step_index=index,
                success=result.success,
                evidence_count=len(result.evidence),
            )
            updates.append(ExecutionUpdate("step", result.message, active.plan.plan_id, index + 1, len(active.plan.steps), result))

            if result.status in {"awaiting_confirmation", "awaiting_strong_auth"}:
                confirmation_id = _confirmation_id(result)
                active.waiting_confirmation_id = confirmation_id
                self.states.transition(AssistantState.AWAITING_CONFIRMATION, result.message, confirmation_id=confirmation_id, plan_id=active.plan.plan_id)
                updates.append(ExecutionUpdate(result.status, result.message, active.plan.plan_id, index + 1, len(active.plan.steps), result, confirmation_id))
                return updates

            if not result.success:
                if result.retryable and attempt < self.max_retries:
                    updates.append(ExecutionUpdate("retrying", "The step failed transiently; retrying safely.", active.plan.plan_id, index + 1, len(active.plan.steps), result))
                    self.states.transition(
                        AssistantState.EXECUTING,
                        "Retrying the failed step safely",
                        plan_id=active.plan.plan_id,
                        step_id=step.step_id,
                        step_index=index,
                    )
                    continue
                updates.append(self._finish_failure(active, result.message))
                return updates
            active.next_index += 1

    def approve_and_resume(
        self,
        confirmation_id: str,
        session: SessionContext,
        method: AuthMethod,
    ) -> list[ExecutionUpdate]:
        with self._lock:
            active = self._active
            if active is None:
                return [ExecutionUpdate("error", "There is no active plan to approve.")]
            if active.waiting_confirmation_id != confirmation_id:
                return [ExecutionUpdate("error", "That confirmation does not belong to the active plan.", active.plan.plan_id)]
            if (
                active.session.session_id != session.session_id
                or active.session.actor.user_id != session.actor.user_id
            ):
                return [
                    ExecutionUpdate(
                        "denied",
                        "That confirmation belongs to a different authenticated session.",
                        active.plan.plan_id,
                    )
                ]
            index = active.next_index
            step = active.plan.steps[index]
        try:
            approval = self.confirmations.approve(confirmation_id, session, method)
        except (ValueError, PermissionError) as error:
            return [ExecutionUpdate("denied", str(error), active.plan.plan_id)]
        with self._lock:
            # Clear the checkpoint only after the actor/session-bound approval
            # was accepted. A failed/expired approval must remain resumable or
            # rejectable rather than leaving an orphaned active plan.
            if self._active is not active or active.waiting_confirmation_id != confirmation_id:
                return [ExecutionUpdate("error", "The active confirmation changed; no action was taken.", active.plan.plan_id)]
            active.waiting_confirmation_id = None
        approved_call = ToolCall(step.call.tool_name, dict(step.call.arguments), call_id=step.call.call_id, confirmation=approval)
        active.plan = ExecutionPlan(
            goal=active.plan.goal,
            steps=active.plan.steps[:index]
            + (
                type(step)(
                    approved_call,
                    step.description,
                    step.depends_on,
                    step.step_id,
                ),
            )
            + active.plan.steps[index + 1:],
            plan_id=active.plan.plan_id,
            created_at=active.plan.created_at,
        )
        self.states.transition(
            AssistantState.EXECUTING,
            "Confirmation approved; resuming the plan",
            plan_id=active.plan.plan_id,
        )
        return self._run_until_pause()

    def reject(self, confirmation_id: str, session: SessionContext) -> list[ExecutionUpdate]:
        with self._lock:
            active = self._active
        if active is None:
            return [ExecutionUpdate("error", "There is no active plan to reject.")]
        if (
            active.session.session_id != session.session_id
            or active.session.actor.user_id != session.actor.user_id
        ):
            return [
                ExecutionUpdate(
                    "denied",
                    "That confirmation belongs to a different authenticated session.",
                    active.plan.plan_id,
                )
            ]
        try:
            self.confirmations.reject(confirmation_id, session)
        except (ValueError, PermissionError) as error:
            return [ExecutionUpdate("denied", str(error), active.plan.plan_id)]
        return [self._finish_failure(active, "The requested action was rejected.")]

    def cancel(self, reason: str = "Plan cancelled by user") -> list[ExecutionUpdate]:
        with self._lock:
            active = self._active
        if active is None:
            self.confirmations.cancel_all()
            return [ExecutionUpdate("cancelled", "No active plan remained.")]
        active.token.cancel(reason)
        self.confirmations.cancel_all()
        return [self._finish_cancelled(active, reason)]

    def trigger_emergency_stop(self) -> list[ExecutionUpdate]:
        self.emergency_stop.trigger()
        self.confirmations.cancel_all()
        with self._lock:
            active = self._active
        if active is None:
            self.states.transition(AssistantState.STOPPED, "Emergency stop activated")
            return [ExecutionUpdate("emergency_stopped", "Emergency stop activated.")]
        return [self._finish_cancelled(active, "Emergency stop activated")]

    def _finish_success(self, active: ActiveExecution) -> ExecutionUpdate:
        self._clear_active(active)
        self.states.transition(AssistantState.SUCCESS, "Plan completed with verification", plan_id=active.plan.plan_id)
        self.audit.append("plan_completed", actor_id=active.session.actor.user_id, session_id=active.session.session_id, details={"plan_id": active.plan.plan_id})
        return ExecutionUpdate("complete", "Plan completed with verified results.", active.plan.plan_id, len(active.plan.steps), len(active.plan.steps))

    def _finish_failure(self, active: ActiveExecution, message: str) -> ExecutionUpdate:
        self._clear_active(active)
        self.states.transition(AssistantState.ERROR, message, plan_id=active.plan.plan_id)
        # The user-facing message can contain a contact name, path, or provider
        # detail.  Keep that text in the in-memory/UI state, but record only a
        # stable status in the durable audit stream.
        self.audit.append(
            "plan_failed",
            actor_id=active.session.actor.user_id,
            session_id=active.session.session_id,
            outcome="failed",
            details={"plan_id": active.plan.plan_id},
        )
        return ExecutionUpdate("failed", message, active.plan.plan_id, active.next_index + 1, len(active.plan.steps))

    def _finish_cancelled(self, active: ActiveExecution, reason: str = "Plan cancelled") -> ExecutionUpdate:
        self._clear_active(active)
        self.states.transition(AssistantState.STOPPED, reason, plan_id=active.plan.plan_id)
        self.audit.append(
            "plan_cancelled",
            actor_id=active.session.actor.user_id,
            session_id=active.session.session_id,
            outcome="cancelled",
            details={"plan_id": active.plan.plan_id},
        )
        return ExecutionUpdate("cancelled", reason, active.plan.plan_id, active.next_index, len(active.plan.steps))

    def _clear_active(self, active: ActiveExecution) -> None:
        with self._lock:
            if self._active is active:
                self._active = None
                self.emergency_stop.unregister(active.token)


def _confirmation_id(result: ToolResult) -> str | None:
    for evidence in result.evidence:
        if evidence.kind == "confirmation_preview":
            return str(evidence.data.get("confirmation_id", "")) or None
    return None
