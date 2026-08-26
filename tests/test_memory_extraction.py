"""Unit tests for automatic long-term memory extraction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.agent import Agent
from app.llm.base import ChatMessage, ChatResponse, LLMProvider, LLMResponse
from app.memory import JsonFileMemoryStore
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class DualScriptedProvider(LLMProvider):
    @property
    def provider_id(self) -> str: return "scripted"
    
    def __init__(self, agent_responses: list[LLMResponse], extractor_payloads: list[str]) -> None:
        self._agent_responses = agent_responses
        self._extractor_payloads = extractor_payloads
        self._agent_index = 0
        self._extractor_index = 0

    def generate(self, request, model_id="test") -> LLMResponse:
        from app.llm.models import TaskType, ModelResponse
        if getattr(request, "task_type", None) == TaskType.REFLECTION:
            if self._extractor_index >= len(self._extractor_payloads):
                raise AssertionError("No extractor payload left for this provider call.")
            payload = self._extractor_payloads[self._extractor_index]
            self._extractor_index += 1
            return ModelResponse(content=payload, model="extractor-model")
        else:
            if self._agent_index >= len(self._agent_responses):
                raise AssertionError("No agent response left for this provider call.")
            response = self._agent_responses[self._agent_index]
            self._agent_index += 1
            return response


def _registry_with_calculator() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def _store(path: Path) -> JsonFileMemoryStore:
    return JsonFileMemoryStore(storage_path=path)


def test_useful_fact_is_stored(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="fact@example.com")
    provider = DualScriptedProvider(
        agent_responses=[LLMResponse(model="agent-model", content="Got it.")],
        extractor_payloads=['{"should_store": true, "memory": "User prefers concise answers."}'],
    )
    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="chat-a",
        memory_store=store,
    )

    agent.run("Please remember I prefer concise answers.")

    memories = store.get_memories(user.user_id)
    assert [memory.content for memory in memories] == ["User prefers concise answers."]


def test_irrelevant_query_is_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="irrelevant@example.com")
    provider = DualScriptedProvider(
        agent_responses=[LLMResponse(model="agent-model", content="The weather is sunny.")],
        extractor_payloads=['{"should_store": false, "memory": ""}'],
    )
    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="chat-b",
        memory_store=store,
    )

    agent.run("What's the weather?")

    assert store.get_memories(user.user_id) == []


def test_related_fact_updates_existing_memory(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="update@example.com")
    provider = DualScriptedProvider(
        agent_responses=[
            LLMResponse(model="agent-model", content="Noted."),
            LLMResponse(model="agent-model", content="Updated."),
        ],
        extractor_payloads=[
            '{"should_store": true, "memory": "User prefers dark mode."}',
            '{"should_store": true, "memory": "User prefers dark mode and compact layout."}',
        ],
    )
    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="chat-c",
        memory_store=store,
    )

    agent.run("I prefer dark mode.")
    agent.run("Also I like compact layout.")

    memories = store.get_memories(user.user_id)
    assert len(memories) == 1
    assert memories[0].content == "User prefers dark mode and compact layout."


def test_memory_extraction_respects_user_isolation(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user_a = store.create_user(email="a@example.com")
    user_b = store.create_user(email="b@example.com")

    provider_a = DualScriptedProvider(
        agent_responses=[LLMResponse(model="agent-model", content="Done A.")],
        extractor_payloads=['{"should_store": true, "memory": "User A likes tea."}'],
    )
    provider_b = DualScriptedProvider(
        agent_responses=[LLMResponse(model="agent-model", content="Done B.")],
        extractor_payloads=['{"should_store": true, "memory": "User B likes coffee."}'],
    )

    agent_a = Agent(
        provider_a,
        _registry_with_calculator(),
        user_id=user_a.user_id,
        chat_id="chat-a",
        memory_store=store,
    )
    agent_b = Agent(
        provider_b,
        _registry_with_calculator(),
        user_id=user_b.user_id,
        chat_id="chat-b",
        memory_store=store,
    )

    agent_a.run("Remember my drink preference.")
    agent_b.run("Remember my drink preference.")

    memories_a = [memory.content for memory in store.get_memories(user_a.user_id)]
    memories_b = [memory.content for memory in store.get_memories(user_b.user_id)]
    assert memories_a == ["User A likes tea."]
    assert memories_b == ["User B likes coffee."]
