"""Supervisor/orchestrator for multi-agent execution with optional planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.communicator import AgentCommunicator
from app.exceptions import AgentCancellationError, AgentTimeoutError
from app.agent.contracts import (
    AgentCapabilities,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.exceptions import AgentExecutionError
from app.agent.delegation import AgentDelegation
from app.agent.execution_context import AgentExecutionContext
from app.agent.execution_policy import ExecutionPolicy
from app.agent.executor import PlanExecutor
from app.agent.plan import Plan, PlanStatus, StepResult
from app.agent.planner import Planner
from app.agent.registry import AgentRegistry
from app.agent.replanning import ReplanningPolicy
from app.agent.retry import RetryBoundary, RetryPolicy
from app.agent.routing import AgentRouter, CapabilityRouter
from app.agent.state import AgentState
from app.agent.validator import PlanValidator
from app.logger import get_logger

logger = get_logger(__name__)


class Supervisor(BaseAgent):
    """
    Coordinate child agents through routing and structured delegation.

    Supports two execution paths:
    - Legacy (Phase 5.3/5.4): Sequential routing and delegation.
    - Planning (Phase 5.5): Planner -> Validator -> Executor pipeline.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent] | None = None,
        *,
        router: AgentRouter | None = None,
        max_attempts: int | None = None,
        registry: AgentRegistry | None = None,
        communicator: AgentCommunicator | None = None,
        retry_policy: RetryPolicy | None = None,
        planner: Planner | None = None,
        plan_validator: PlanValidator | None = None,
        plan_executor: PlanExecutor | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        if agents is None and registry is None:
            raise ValueError("Either agents or registry must be provided.")
        
        if registry:
            self._agents = registry.get_all()
        else:
            if not agents:
                raise ValueError("At least one child agent is required.")
            self._agents = list(agents)

        self._router = router or CapabilityRouter()
        self._registry = registry
        self._communicator = communicator
        self._retry_policy = retry_policy or RetryPolicy()

        if max_attempts is None:
            self._max_attempts = len(self._agents)
        else:
            if max_attempts <= 0:
                raise ValueError("max_attempts must be greater than zero.")

            self._max_attempts = min(
                max_attempts,
                len(self._agents),
            )

        self._identity = AgentIdentity(
            name="production-supervisor-agent",
            version="1.0.0",
            description="Production AI Research Supervisor Agent",
        )

        self._capabilities = AgentCapabilities(
            tool_use=False,
            memory=True,
            multi_turn=True,
            retrieval=False,
        )

        # Phase 5.5: Planning stack (all optional for backward compat)
        self._planner = planner
        self._plan_validator = plan_validator or (PlanValidator() if planner else None)
        self._plan_executor = plan_executor
        self._execution_policy = execution_policy or ExecutionPolicy()

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def _create_delegation(
        self,
        *,
        request: AgentRequest,
        context: AgentExecutionContext,
        child: BaseAgent,
    ) -> AgentDelegation:
        delegation = AgentDelegation(
            requesting_agent_id=self.identity.name,
            target_agent_id=child.identity.name,
            task=request.input_text,
            execution_id=context.execution_id,
            metadata={
                "request_id": request.request_id,
            },
        )

        delegation.mark_running()
        context.add_delegation(delegation)

        return delegation

    def execute(self, request: AgentRequest) -> AgentResult:
        normalized_input = request.input_text.strip()

        if not normalized_input:
            raise AgentExecutionError(
                "Agent request input_text must not be empty.",
                request=request,
                details={
                    "request_id": request.request_id,
                },
            )

        context = request.context

        if context is None:
            context = AgentExecutionContext(
                task=normalized_input,
            )

        context.mark_running()

        # Phase 5.5: Use planning path if planner and executor are provided
        if self._planner and self._plan_executor:
            return self._execute_with_planning(
                request=request,
                normalized_input=normalized_input,
                context=context,
            )

        # Legacy Phase 5.3/5.4 sequential delegation path
        return self._execute_legacy(
            request=request,
            normalized_input=normalized_input,
            context=context,
        )

    def _execute_with_planning(
        self,
        *,
        request: AgentRequest,
        normalized_input: str,
        context: AgentExecutionContext,
    ) -> AgentResult:
        """Execute via the Planner -> Validator -> Executor pipeline."""
        assert self._planner is not None
        assert self._plan_executor is not None

        replanning_policy = ReplanningPolicy(self._execution_policy)
        previous_plan: Plan | None = None
        failure_context: list[StepResult] | None = None

        while True:
            # 1. Generate plan
            plan = self._planner.generate_plan(
                goal=normalized_input,
                context=context,
                previous_plan=previous_plan,
                failure_context=failure_context,
            )

            logger.info(
                "Plan generated",
                extra={
                    "plan_id": plan.plan_id,
                    "step_count": len(plan.steps),
                    "is_replan": previous_plan is not None,
                },
            )

            # 2. Validate plan
            if self._plan_validator:
                self._plan_validator.validate(plan)

            # 3. Execute plan
            executed_plan = self._plan_executor.execute(
                plan=plan,
                context=context,
                policy=self._execution_policy,
            )

            # 4. Check result
            if executed_plan.status in (PlanStatus.COMPLETED, PlanStatus.PARTIAL_SUCCESS):
                # Collect successful outputs
                final_output = self._collect_plan_output(executed_plan)

                state = AgentState()
                state.finished = True
                state.final_answer = final_output
                
                is_partial = executed_plan.status == PlanStatus.PARTIAL_SUCCESS

                if is_partial:
                    context.mark_partial_success()
                    logger.info("partial_success", extra={"plan_id": executed_plan.plan_id})
                else:
                    context.mark_completed()

                return AgentResult(
                    request=request,
                    state=state,
                    output=final_output,
                    success=True,
                    context=context,
                    metadata={
                        "plan_id": executed_plan.plan_id,
                        "steps_completed": sum(
                            1 for s in executed_plan.steps.values()
                            if s.result and s.result.success
                        ),
                        "partial_success": is_partial,
                    },
                )

            # 5. Plan failed — check replanning policy
            if replanning_policy.should_replan(executed_plan):
                replanning_policy.record_replan()
                previous_plan = executed_plan
                failure_context = [
                    s.result for s in executed_plan.steps.values()
                    if s.result and not s.result.success
                ]
                logger.info(
                    "Replanning after failure",
                    extra={
                        "plan_id": executed_plan.plan_id,
                        "replan_count": replanning_policy.replan_count,
                    },
                )
                continue

            # 6. No more replanning — terminal failure
            context.mark_failed()
            logger.info("execution_failed", extra={"plan_id": executed_plan.plan_id})

            raise AgentExecutionError(
                "Plan execution failed and replanning exhausted.",
                request=request,
                details={
                    "plan_id": executed_plan.plan_id,
                    "plan_status": executed_plan.status.value,
                },
            )

    @staticmethod
    def _collect_plan_output(plan: Plan) -> str:
        """Collect output from completed plan steps."""
        outputs: list[str] = []
        for step in plan.steps.values():
            if (
                step.result
                and step.result.success
                and step.result.agent_result
                and step.result.agent_result.output
            ):
                outputs.append(step.result.agent_result.output)
        return "\n\n".join(outputs) if outputs else "Plan completed successfully."

    def _execute_legacy(
        self,
        *,
        request: AgentRequest,
        normalized_input: str,
        context: AgentExecutionContext,
    ) -> AgentResult:
        """Legacy Phase 5.3/5.4 sequential delegation path."""
        state = AgentState()
        state.messages.append(
            {
                "role": "user",
                "content": normalized_input,
            }
        )
        state.finished = False
        state.final_answer = None

        last_error: AgentExecutionError | None = None
        attempted_agents: set[str] = set()

        attempts = min(
            self._max_attempts,
            len(self._agents),
        )

        for attempt in range(attempts):
            if context.is_cancelled:
                logger.info("agent_cancelled", extra={"request_id": request.request_id})
                raise AgentCancellationError("Supervisor execution cancelled via context.", request=request)
            if context.is_timed_out:
                logger.info("agent_timeout", extra={"request_id": request.request_id})
                raise AgentTimeoutError("Supervisor execution timed out via context.", request=request)

            remaining_agents = [
                agent
                for agent in self._agents
                if agent.identity.name not in attempted_agents
            ]

            if not remaining_agents:
                break

            child_request = AgentRequest(
                input_text=normalized_input,
                metadata={
                    **request.metadata,
                    "supervisor_attempt": attempt + 1,
                },
                request_id=request.request_id,
                context=context,
                correlation_id=context.correlation_id,
                sender_id=self.identity.name,
            )

            child = self._router.select_agent(
                child_request,
                remaining_agents,
            )

            if child is None:
                last_error = AgentExecutionError(
                    "No suitable child agent was found.",
                    request=request,
                    details={
                        "request_id": request.request_id,
                        "attempt": attempt + 1,
                    },
                )
                break

            attempted_agents.add(child.identity.name)

            child_request.metadata["selected_agent"] = (
                child.identity.name
            )

            delegation = self._create_delegation(
                request=child_request,
                context=context,
                child=child,
            )

            context.mark_executing()

            try:
                if self._communicator:
                    retry_boundary = RetryBoundary(self._retry_policy, self._communicator)
                    child_result = retry_boundary.send(child_request)
                else:
                    child_result = child.execute(child_request)

            except Exception as error:
                logger.info("agent_failed", extra={"agent_id": child.identity.name, "error_type": type(error).__name__})
                delegation.mark_failed(str(error))

                context.publish_agent_output(
                    agent_id=child.identity.name,
                    output=None,
                    success=False,
                    metadata={
                        "selected_agent": child.identity.name,
                        "delegation_id": delegation.delegation_id,
                        "error": str(error),
                    },
                )

                if isinstance(error, AgentExecutionError):
                    last_error = error
                else:
                    last_error = AgentExecutionError(
                        f"Child agent execution failed: {error}",
                        request=child_request,
                        details={"original_error": type(error).__name__}
                    )
                continue

            if child_result.success and child_result.output is not None:
                logger.info("execution_recovered", extra={"agent_id": child.identity.name, "attempt": attempt + 1}) if attempt > 0 else None
                delegation.mark_completed(child_result.output)

                state.iteration = attempt + 1
                state.finished = True
                state.final_answer = child_result.output

                state.messages.append(
                    {
                        "role": "assistant",
                        "content": child_result.output,
                    }
                )

                result_metadata: dict[str, Any] = {
                    "selected_agent": child.identity.name,
                    "attempts": attempt + 1,
                    "delegation_id": delegation.delegation_id,
                }

                context.publish_agent_output(
                    agent_id=self.identity.name,
                    output=child_result.output,
                    success=True,
                    metadata=result_metadata,
                )

                context.mark_completed()

                return AgentResult(
                    request=request,
                    state=state,
                    output=child_result.output,
                    success=True,
                    context=context,
                    metadata=result_metadata,
                )

            delegation.mark_failed(
                "Child agent completed without a usable result."
            )

            context.publish_agent_output(
                agent_id=child.identity.name,
                output=child_result.output,
                success=False,
                metadata={
                    "selected_agent": child.identity.name,
                    "delegation_id": delegation.delegation_id,
                    "reason": "unusable_result",
                },
            )

            last_error = AgentExecutionError(
                "Child agent completed without a usable result.",
                request=request,
                details={
                    "request_id": request.request_id,
                    "selected_agent": child.identity.name,
                    "delegation_id": delegation.delegation_id,
                },
            )

        state.iteration = len(attempted_agents)
        state.finished = False
        state.final_answer = None

        context.mark_failed()

        if last_error is not None:
            raise last_error

        raise AgentExecutionError(
            "Supervisor execution failed without child-agent results.",
            request=request,
            details={
                "request_id": request.request_id,
            },
        )