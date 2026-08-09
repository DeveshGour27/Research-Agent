"""Deterministic agent routing for multi-agent execution."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.agent.contracts import AgentRequest, BaseAgent


class AgentRouter(ABC):
    """Provider-neutral interface for selecting an agent."""

    @abstractmethod
    def select_agent(
        self,
        request: AgentRequest,
        agents: list[BaseAgent],
    ) -> BaseAgent | None:
        """Select the best available agent for a request."""


class CapabilityRouter(AgentRouter):
    """
    Deterministic capability-aware router.

    Routing uses:
    1. Explicit required capabilities in request metadata.
    2. Token overlap between the request and agent identity metadata.
    3. Registration order as the deterministic final tie-breaker.
    """

    _CAPABILITY_FIELDS = (
        "tool_use",
        "memory",
        "multi_turn",
        "retrieval",
    )

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_]+", value.lower())
            if len(token) > 1
        }

    def _capability_score(
        self,
        request: AgentRequest,
        agent: BaseAgent,
    ) -> int:
        required = request.metadata.get("required_capabilities", [])

        if not required:
            return 0

        if not isinstance(required, (list, tuple, set)):
            return 0

        score = 0

        for capability in required:
            if capability not in self._CAPABILITY_FIELDS:
                continue

            if getattr(agent.capabilities, capability, False):
                score += 10
            else:
                score -= 100

        return score

    def _text_score(
        self,
        request: AgentRequest,
        agent: BaseAgent,
    ) -> int:
        request_tokens = self._tokens(request.input_text)

        if not request_tokens:
            return 0

        identity_tokens = (
            self._tokens(agent.identity.name)
            | self._tokens(agent.identity.description)
            | self._tokens(agent.description)
        )

        return len(request_tokens & identity_tokens)

    def select_agent(
        self,
        request: AgentRequest,
        agents: list[BaseAgent],
    ) -> BaseAgent | None:
        if not agents:
            return None

        scored_agents: list[tuple[int, int, BaseAgent]] = []

        for index, agent in enumerate(agents):
            capability_score = self._capability_score(request, agent)

            if capability_score < 0:
                continue

            text_score = self._text_score(request, agent)

            total_score = capability_score + text_score
            scored_agents.append((total_score, -index, agent))

        if not scored_agents:
            return None

        scored_agents.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return scored_agents[0][2]