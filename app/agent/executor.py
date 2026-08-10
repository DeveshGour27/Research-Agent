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
from app.exceptions import PlanExecutionError
from app.logger import get_logger

logger = get_logger(__name__)


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

        steps_executed = 0
        failures = 0

        while True:
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
                    return plan

                step_result = self._execute_step(step, plan, context)
                steps_executed += 1

                if step_result.success:
                    step.status = PlanStepStatus.COMPLETED
                    step.result = step_result
                    self._unlock_dependents(step.step_id, plan)
                else:
                    step.status = PlanStepStatus.FAILED
                    step.result = step_result
                    failures += 1

        all_completed = all(
            s.status == PlanStepStatus.COMPLETED
            for s in plan.steps.values()
        )
        plan.status = PlanStatus.COMPLETED if all_completed else PlanStatus.FAILED

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

        agents = self._registry.get_all()
        selected_agent = self._router.select_agent(request, agents)

        if selected_agent is None:
            logger.warning(
                "No agent found for step",
                extra={
                    "plan_id": plan.plan_id,
                    "step_id": step.step_id,
                    "task_type": step.task_type,
                },
            )
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=f"No suitable agent found for step '{step.step_id}'.",
            )

        request.metadata["selected_agent"] = selected_agent.identity.name

        try:
            retry_boundary = RetryBoundary(self._retry_policy, self._communicator)
            agent_result = retry_boundary.send(request)

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
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=str(error),
            )
