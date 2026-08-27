"""Plan executor that traverses the DAG using Phase 5.4 infrastructure."""

from __future__ import annotations

from app.agent.communicator import AgentCommunicator
from app.agent.contracts import AgentExecutionError, AgentRequest, AgentResult, BaseAgent
from app.agent.execution_context import AgentExecutionContext
from app.agent.execution_policy import ExecutionPolicy
from app.agent.plan import Plan, PlanStatus, PlanStep, PlanStepStatus, StepResult
from app.agent.registry import AgentRegistry
from app.agent.retry import RetryBoundary, RetryPolicy
from app.agent.routing import AgentRouter, CapabilityRouter
from app.agent.collaboration import CollaborationSession
from app.exceptions import PlanExecutionError, AgentCancellationError, AgentTimeoutError
from app.logger import get_logger
from app.observability.tracer import extract_span_info
from app.observability import events as obs_events

logger = get_logger(__name__)


def _emit_safe(event: obs_events.BaseEvent) -> None:
    """Emit an observability event, swallowing any failure."""
    try:
        event.emit()
    except Exception:
        pass


class PlanExecutor:
    """Executes an approved Plan by traversing its dependency graph.

    Uses the existing Phase 5.4 infrastructure:
    - CapabilityRouter for agent selection
    - AgentRegistry for agent discovery
    - RetryBoundary / AgentCommunicator for execution

    Does NOT contain replanning logic.
    """

    def __init__(
        self,
        router: AgentRouter,
        registry: AgentRegistry,
        communicator: AgentCommunicator,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._router = router
        self._registry = registry
        self._communicator = communicator
        self._retry_policy = retry_policy or RetryPolicy()

    def execute(
        self,
        plan: Plan,
        context: AgentExecutionContext,
        policy: ExecutionPolicy,
    ) -> Plan:
        """Execute the plan step by step, respecting dependencies.

        Args:
            plan: A validated Plan.
            context: The shared execution context.
            policy: Execution bounds.

        Returns:
            The plan with updated step statuses and results.
        """
        plan.status = PlanStatus.EXECUTING
        self._initialize_ready_steps(plan)

        # Phase 5.8: Emit execution-started event
        span = extract_span_info(context)
        if span:
            _emit_safe(obs_events.AgentExecutionStartedEvent(
                trace_id=span.trace_id, run_id=span.run_id,
                span_id=span.span_id, parent_span_id=span.parent_span_id,
                agent_name="PlanExecutor", input_text=plan.goal,
            ))

        steps_executed = 0
        failures = 0

        while True:
            if context.is_cancelled:
                logger.info("execution_cancelled", extra={"plan_id": plan.plan_id})
                plan.status = PlanStatus.FAILED
                if span:
                    _emit_safe(obs_events.TimeoutCancellationEvent(
                        trace_id=span.trace_id, run_id=span.run_id,
                        span_id=span.span_id, parent_span_id=span.parent_span_id,
                        reason="cancelled",
                    ))
                raise AgentCancellationError("Execution cancelled via context.", request=None)
            if context.is_timed_out:
                logger.info("execution_timed_out", extra={"plan_id": plan.plan_id})
                plan.status = PlanStatus.FAILED
                if span:
                    _emit_safe(obs_events.TimeoutCancellationEvent(
                        trace_id=span.trace_id, run_id=span.run_id,
                        span_id=span.span_id, parent_span_id=span.parent_span_id,
                        reason="timed_out",
                    ))
                raise AgentTimeoutError("Execution timed out via context.", request=None)

            ready_steps = [
                step for step in plan.steps.values()
                if step.status == PlanStepStatus.READY
            ]

            if not ready_steps:
                all_done = all(
                    s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.FAILED)
                    for s in plan.steps.values()
                )
                if all_done:
                    break
                break

            for step in ready_steps:
                if steps_executed >= policy.max_steps:
                    logger.warning(
                        "Max steps reached",
                        extra={
                            "plan_id": plan.plan_id,
                            "steps_executed": steps_executed,
                            "max_steps": policy.max_steps,
                        },
                    )
                    plan.status = PlanStatus.FAILED
                    if span:
                        _emit_safe(obs_events.AgentExecutionCompletedEvent(
                            trace_id=span.trace_id, run_id=span.run_id,
                            span_id=span.span_id, parent_span_id=span.parent_span_id,
                            agent_name="PlanExecutor", success=False, output="Max steps reached",
                        ))
                    return plan

                if failures >= policy.max_failures:
                    logger.warning(
                        "Max failures reached",
                        extra={
                            "plan_id": plan.plan_id,
                            "failures": failures,
                            "max_failures": policy.max_failures,
                        },
                    )
                    plan.status = PlanStatus.FAILED
                    if span:
                        _emit_safe(obs_events.AgentExecutionCompletedEvent(
                            trace_id=span.trace_id, run_id=span.run_id,
                            span_id=span.span_id, parent_span_id=span.parent_span_id,
                            agent_name="PlanExecutor", success=False, output="Max failures reached",
                        ))
                    return plan

                step_result = self._execute_step(step, plan, context)
                steps_executed += 1

                if step_result.success:
                    step.status = PlanStepStatus.COMPLETED
                    step.result = step_result
                    self._unlock_dependents(step.step_id, plan)
                    # Phase 5.8: step completed event
                    if span:
                        _emit_safe(obs_events.PlanStepCompletedEvent(
                            trace_id=span.trace_id, run_id=span.run_id,
                            span_id=span.span_id, parent_span_id=span.parent_span_id,
                            plan_id=plan.plan_id, step_id=step.step_id, status="completed",
                        ))
                else:
                    step.status = PlanStepStatus.FAILED
                    step.result = step_result
                    failures += 1
                    # Phase 5.8: step failed event
                    if span:
                        _emit_safe(obs_events.PlanStepCompletedEvent(
                            trace_id=span.trace_id, run_id=span.run_id,
                            span_id=span.span_id, parent_span_id=span.parent_span_id,
                            plan_id=plan.plan_id, step_id=step.step_id, status="failed",
                        ))

        all_completed = all(
            s.status == PlanStepStatus.COMPLETED
            for s in plan.steps.values()
        )
        
        if all_completed:
            plan.status = PlanStatus.COMPLETED
            if span:
                _emit_safe(obs_events.AgentExecutionCompletedEvent(
                    trace_id=span.trace_id, run_id=span.run_id,
                    span_id=span.span_id, parent_span_id=span.parent_span_id,
                    agent_name="PlanExecutor", success=True, output=None,
                ))
        else:
            required_failed = any(
                s.is_required and s.status == PlanStepStatus.FAILED
                for s in plan.steps.values()
            )
            if required_failed:
                plan.status = PlanStatus.FAILED
                logger.info("execution_failed", extra={"plan_id": plan.plan_id})
                if span:
                    _emit_safe(obs_events.AgentExecutionCompletedEvent(
                        trace_id=span.trace_id, run_id=span.run_id,
                        span_id=span.span_id, parent_span_id=span.parent_span_id,
                        agent_name="PlanExecutor", success=False, output="required_step_failed",
                    ))
            else:
                plan.status = PlanStatus.PARTIAL_SUCCESS
                logger.info("partial_success", extra={"plan_id": plan.plan_id})
                if span:
                    _emit_safe(obs_events.AgentExecutionCompletedEvent(
                        trace_id=span.trace_id, run_id=span.run_id,
                        span_id=span.span_id, parent_span_id=span.parent_span_id,
                        agent_name="PlanExecutor", success=True, output="partial_success",
                    ))

        return plan

    def _initialize_ready_steps(self, plan: Plan) -> None:
        """Mark steps with no dependencies as READY."""
        for step in plan.steps.values():
            if step.status == PlanStepStatus.PENDING and not step.dependencies:
                step.status = PlanStepStatus.READY

    def _unlock_dependents(self, completed_step_id: str, plan: Plan) -> None:
        """Unlock steps whose dependencies are now all COMPLETED."""
        for step in plan.steps.values():
            if step.status != PlanStepStatus.PENDING:
                continue
            if completed_step_id not in step.dependencies:
                continue

            all_deps_done = all(
                plan.steps[dep_id].status == PlanStepStatus.COMPLETED
                for dep_id in step.dependencies
            )
            if all_deps_done:
                step.status = PlanStepStatus.READY

    def _execute_step(
        self,
        step: PlanStep,
        plan: Plan,
        context: AgentExecutionContext,
    ) -> StepResult:
        """Execute a single step through the Phase 5.4 communication layer."""
        step.status = PlanStepStatus.RUNNING

        logger.info(
            "Executing plan step",
            extra={
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "task_type": step.task_type,
            },
        )

        metadata: dict[str, object] = {}
        if step.task_type:
            metadata["required_task_type"] = step.task_type
        if step.required_capabilities:
            metadata["required_capabilities"] = list(step.required_capabilities)

        request = AgentRequest(
            input_text=step.description,
            metadata=metadata,
            context=context,
            correlation_id=context.correlation_id,
        )

        try:
            # Phase 5.6: Step execution delegates to CollaborationSession
            # which handles router, retry, and handoffs
            import time
            start_time = time.perf_counter()
            session = CollaborationSession(
                step_id=step.step_id,
                initial_request=request,
                router=self._router,
                registry=self._registry,
                communicator=self._communicator,
                retry_policy=self._retry_policy,
            )
            
            agent_result = session.execute()
            latency_s = round(time.perf_counter() - start_time, 4)

            from app.observability.events import AgentStepEvent
            _emit_safe(AgentStepEvent(
                trace_id=getattr(context, "trace_id", None) if context else None,
                run_id=getattr(context, "run_id", None) if context else None,
                span_id=getattr(context, "span_id", None) if context else None,
                parent_span_id=getattr(context, "parent_span_id", None) if context else None,
                agent_name="executor",
                step_name=step.step_id,
                success=agent_result.success,
                latency_s=latency_s,
                error_info=None if agent_result.success else "Agent returned unsuccessful result."
            ))

            return StepResult(
                step_id=step.step_id,
                success=agent_result.success,
                agent_result=agent_result,
                error=None if agent_result.success else "Agent returned unsuccessful result.",
            )

        except Exception as error:
            logger.warning(
                "Step execution failed",
                extra={
                    "plan_id": plan.plan_id,
                    "step_id": step.step_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            from app.observability.events import AgentStepEvent
            _emit_safe(AgentStepEvent(
                trace_id=getattr(context, "trace_id", None) if context else None,
                run_id=getattr(context, "run_id", None) if context else None,
                span_id=getattr(context, "span_id", None) if context else None,
                parent_span_id=getattr(context, "parent_span_id", None) if context else None,
                agent_name="executor",
                step_name=step.step_id,
                success=False,
                latency_s=None,
                error_info=str(error)
            ))
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=str(error),
            )
