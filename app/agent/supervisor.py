"""Supervisor/orchestrator for sequential multi-agent execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.contracts import (
    AgentCapabilities,
    AgentExecutionError,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.delegation import AgentDelegation
from app.agent.execution_context import AgentExecutionContext
from app.agent.routing import AgentRouter, CapabilityRouter
from app.agent.state import AgentState


class Supervisor(BaseAgent):
    """
    Coordinate child agents through routing and structured delegation.

    Execution remains sequential in Phase 5.3.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        *,
        router: AgentRouter | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if not agents:
            raise ValueError("At least one child agent is required.")

        self._agents = list(agents)
        self._router = router or CapabilityRouter()

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

            try:
                child_result = child.execute(child_request)

            except AgentExecutionError as error:
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

                last_error = error
                continue

            if child_result.success and child_result.output is not None:
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