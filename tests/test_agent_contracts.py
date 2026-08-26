from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.agent import (
    Agent,
    AgentExecutionError,
    AgentRequest,
)
from app.llm.base import ChatMessage, ChatResponse, LLMProvider, LLMResponse
from app.tools.registry import ToolRegistry


class ScriptedProvider(LLMProvider):
    @property
    def provider_id(self) -> str: return "scripted"

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    
    def generate(self, request, model_id="test") -> LLMResponse:
        messages = request.messages
        tools = request.tools
        if self.calls >= len(self._responses):
            raise AssertionError("No scripted response left for this provider call.")
        response = self._responses[self.calls]
        self.calls += 1
        return response


def test_agent_implements_base_agent_contract() -> None:
    provider = ScriptedProvider([LLMResponse(model="fake-model", content="Contract response")])
    agent = Agent(provider, ToolRegistry())

    assert agent.identity.name == "production-research-agent"
    assert agent.description == "Production AI Research Agent"
    assert agent.capabilities.tool_use is True
    assert agent.capabilities.multi_turn is True

    result = agent.execute(
        AgentRequest(
            input_text="Return a contract result",
            request_id="req-1",
            metadata={"source": "tests"},
        )
    )

    assert result.success is True
    assert result.output == "Contract response"
    assert result.request.request_id == "req-1"
    assert result.metadata["tool_calls"] == 0
    assert result.state.finished is True


def test_execute_rejects_blank_requests() -> None:
    provider = ScriptedProvider([LLMResponse(model="fake-model", content="Unused")])
    agent = Agent(provider, ToolRegistry())

    with pytest.raises(AgentExecutionError):
        agent.execute(AgentRequest(input_text="   "))
