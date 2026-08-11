"""Collaboration limits and orchestration constructs for Phase 5.6."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agent.contracts import AgentRequest, AgentResult, AgentExecutionError
from app.agent.routing import CapabilityRouter
from app.agent.registry import AgentRegistry
from app.agent.retry import RetryBoundary, RetryPolicy
from app.agent.communicator import AgentCommunicator
from app.exceptions import AgentTimeoutError, AgentCancellationError, FatalError, RecoverableError


@dataclass(frozen=True, slots=True)
class Artifact:
    """A strictly immutable shared output from an agent session."""

    artifact_id: str
    artifact_type: str
    producer_id: str
    version: int
    correlation_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class Handoff:
    """A coordination event signaling a shift in active agent."""
    
    target_task_type: str
    message: str
    correlation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CollaborationPolicy:
    """Enforces boundaries on peer-to-peer interactions."""

    max_collaboration_depth: int = 2
    max_messages_per_session: int = 10
    max_participating_agents: int = 3
    max_collaboration_steps: int = 20


class CollaborationSession:
    """
    Orchestrates a bounded multi-agent interaction synchronously.
    
    Acts as the strict boundary between the PlanExecutor (which owns the step)
    and the AgentCommunicator (which owns transport).
    """

    def __init__(
        self,
        step_id: str,
        initial_request: AgentRequest,
        router: CapabilityRouter,
        registry: AgentRegistry,
        communicator: AgentCommunicator,
        retry_policy: RetryPolicy,
        policy: CollaborationPolicy | None = None,
    ) -> None:
        self.step_id = step_id
        self.initial_request = initial_request
        self._router = router
        self._registry = registry
        self._communicator = communicator
        self._retry_policy = retry_policy
        self._policy = policy or CollaborationPolicy()
        
        self.messages_exchanged = 0
        self.participating_agents: set[str] = set()
        self.steps_taken = 0
        self._retry_boundary = RetryBoundary(policy=self._retry_policy, communicator=self._communicator)

    def execute(self) -> AgentResult:
        """Run the collaboration loop until completion or handoff resolution."""
        current_request = self.initial_request
        last_result = None
        context = current_request.context

        while True:
            if context:
                if context.is_cancelled:
                    raise AgentCancellationError("Collaboration session cancelled via context.", request=current_request)
                if context.is_timed_out:
                    raise AgentTimeoutError("Collaboration session timed out via context.", request=current_request)

            if self.steps_taken >= self._policy.max_collaboration_steps:
                raise FatalError(
                    "Max collaboration steps exceeded",
                    request=current_request,
                )
                
            if self.messages_exchanged >= self._policy.max_messages_per_session:
                raise FatalError(
                    "Max messages per session exceeded",
                    request=current_request,
                )

            # 1. Routing & Selection
            agents = self._registry.get_all()
            selected_agent = self._router.select_agent(current_request, agents)

            if not selected_agent:
                raise AgentExecutionError(
                    "No suitable agent found for collaboration",
                    request=current_request,
                )
                
            agent_name = selected_agent.identity.name
            self.participating_agents.add(agent_name)
            
            if len(self.participating_agents) > self._policy.max_participating_agents:
                raise AgentExecutionError(
                    "Max participating agents exceeded",
                    request=current_request,
                )
                
            current_request.metadata["selected_agent"] = agent_name

            # 2. Execution across RetryBoundary & Communicator
            try:
                result = self._retry_boundary.send(current_request)
            except AgentTimeoutError as error:
                # Target agent timed out
                if context:
                    context.publish_agent_output(agent_name, None, False, {"error": "timeout"})
                raise RecoverableError(f"Target agent {agent_name} timed out.", request=current_request) from error
            except AgentExecutionError as error:
                # Target agent failed (could be handoff rejected or invalid result)
                raise RecoverableError(f"Target agent {agent_name} execution failed.", request=current_request) from error
            
            last_result = result
            self.steps_taken += 1

            # 3. Check for CollaborationRequest (Handoff / Messaging)
            if result.collaboration_request:
                self.messages_exchanged += 1
                
                # Advance request to target agent
                current_request = AgentRequest(
                    input_text=result.collaboration_request.message,
                    metadata={
                        "required_task_type": result.collaboration_request.requested_task_type,
                        **result.collaboration_request.metadata,
                    },
                    context=current_request.context,
                    correlation_id=current_request.correlation_id,
                    sender_id=agent_name,
                )
                continue
                
            # If no collaboration request, this agent has finished the step
            break
            
        assert last_result is not None
        return last_result
