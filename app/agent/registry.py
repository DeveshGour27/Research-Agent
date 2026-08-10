"""Registry for discovering and looking up agents."""

from __future__ import annotations

from app.agent.contracts import BaseAgent
from app.exceptions import AgentNotFoundError


class AgentRegistry:
    """
    Central registry for coordinating agents.

    Responsible only for agent discovery and lookup.
    Does NOT execute agents or manage execution state.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent.

        Args:
            agent: The agent instance to register.

        Raises:
            ValueError: If an agent with the same identity name is already registered.
        """
        name = agent.identity.name
        if not name:
            raise ValueError("Agent identity name must not be empty.")

        if name in self._agents:
            raise ValueError(f"Agent with name '{name}' is already registered.")

        self._agents[name] = agent

    def get(self, agent_id: str) -> BaseAgent:
        """
        Retrieve a registered agent by ID.

        Args:
            agent_id: The stable identity name of the agent.

        Returns:
            The requested agent instance.

        Raises:
            AgentNotFoundError: If the agent is not found.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            raise AgentNotFoundError(
                f"Agent '{agent_id}' not found in registry.",
                details={"agent_id": agent_id},
            )
        return agent

    def get_all(self) -> list[BaseAgent]:
        """Return a list of all registered agents."""
        return list(self._agents.values())

    def has(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self._agents

    def agent_ids(self) -> list[str]:
        """Return a list of all registered agent IDs."""
        return list(self._agents.keys())
