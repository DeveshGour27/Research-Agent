"""Communication abstraction for agent execution and messaging."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.agent.contracts import AgentRequest, AgentResult
from app.agent.registry import AgentRegistry
from app.exceptions import ContractValidationError


class AgentCommunicator(ABC):
    """
    Provider-neutral communication layer for agent requests.

    Abstracts whether the target agent is running in-process,
    via HTTP, or on a message queue.
    """

    @abstractmethod
    def send(self, request: AgentRequest) -> AgentResult:
        """
        Send a request to a target agent and await the result.

        Args:
            request: The validated agent request. Must contain a target
                     agent ID in its metadata or the communicator must
                     know how to route it.

        Returns:
            The structured AgentResult from the target.

        Raises:
            AgentNotFoundError: If the target agent cannot be located.
            ContractValidationError: If the request is invalid.
            CommunicationError: If transport fails.
            AgentExecutionError: If the agent executes but fails.
        """


class InProcessCommunicator(AgentCommunicator):
    """
    Synchronous, in-process communicator.

    Uses an AgentRegistry to locate the target agent and calls
    its execute() method directly.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def send(self, request: AgentRequest) -> AgentResult:
        """
        Send a request to an in-process agent.

        Args:
            request: The request. Expected to have 'selected_agent' in metadata.

        Raises:
            ContractValidationError: If 'selected_agent' is missing.
            AgentNotFoundError: If the registry cannot find the agent.
            AgentExecutionError: Let agent execution errors propagate transparently.
        """
        target_agent_id = request.metadata.get("selected_agent")
        if not target_agent_id:
            raise ContractValidationError(
                "Request metadata must contain 'selected_agent' for in-process routing.",
                details={"request_id": request.request_id},
            )

        # registry.get() raises AgentNotFoundError if missing
        agent = self._registry.get(target_agent_id)

        # We do NOT wrap AgentExecutionError or other domain exceptions
        # in CommunicationError. In-process execution lets them propagate naturally.
        return agent.execute(request)
