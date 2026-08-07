"""Unit tests for the Phase 2 tool-using agent engine."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

from app.agent import Agent
from app.config import Settings
from app.exceptions import ToolExecutionError
from app.llm.base import ChatMessage, ChatResponse, LLMProvider, LLMResponse, ToolCall
from app.llm.groq_provider import GroqProvider
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class ScriptedProvider(LLMProvider):
    """Deterministic provider for agent-loop unit tests."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.call_messages: list[list[dict[str, object]]] = []

    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:  # pragma: no cover
        raise AssertionError("ScriptedProvider.generate should not be used in these tests.")

    def generate_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LLMResponse:
        self.call_messages.append([dict(message) for message in messages])
        if self.calls >= len(self._responses):
            raise AssertionError("No scripted response left for this provider call.")
        response = self._responses[self.calls]
        self.calls += 1
        return response


class FailingTool(BaseTool):
    name = "always_fail"
    description = "Always fails for testing error handling."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: object) -> str:
        raise ToolExecutionError("Tool exploded.", tool_name=self.name)


class GroqSDKError(Exception):
    """Minimal SDK-like error carrying an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _make_groq_response(*, content: str | None, finish_reason: str = "stop") -> MagicMock:
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = []
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 4
    return response


def _registry_with_calculator() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def test_direct_llm_response() -> None:
    provider = ScriptedProvider([LLMResponse(model="fake-model", content="Direct answer")])
    state = Agent(provider, _registry_with_calculator()).run("Hello")

    assert state.finished is True
    assert state.final_answer == "Direct answer"
    assert state.tool_calls == []


def test_calculator_tool_call() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-1", name="calculate", arguments={"expression": "2 + 3"}),
            ),
            LLMResponse(model="fake-model", content="The answer is 5."),
        ]
    )

    state = Agent(provider, _registry_with_calculator()).run("What is 2 + 3?")

    assert state.finished is True
    assert state.final_answer == "The answer is 5."
    assert [call.name for call in state.tool_calls] == ["calculate"]
    assert state.observations[0].content == "5"


def test_multi_step_tool_loop() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-1", name="calculate", arguments={"expression": "10 + 5"}),
            ),
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-2", name="calculate", arguments={"expression": "15 * 2"}),
            ),
            LLMResponse(model="fake-model", content="Final answer: 30."),
        ]
    )

    state = Agent(provider, _registry_with_calculator()).run("Compute (10 + 5) * 2.")

    assert state.finished is True
    assert state.final_answer == "Final answer: 30."
    assert [call.id for call in state.tool_calls] == ["call-1", "call-2"]
    assert [observation.content for observation in state.observations] == ["15", "30"]


def test_tool_failure_is_handled() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-err", name="always_fail", arguments={}),
            ),
            LLMResponse(model="fake-model", content="Recovered after tool error."),
        ]
    )

    state = Agent(provider, registry).run("Try the failing tool.")

    assert state.finished is True
    assert state.final_answer == "Recovered after tool error."
    assert state.observations[0].is_error is True
    assert "Error:" in state.observations[0].content


@patch("groq.Groq")
def test_model_fallback(mock_groq_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = [
        GroqSDKError(429),
        _make_groq_response(content="Fallback answer."),
    ]

    provider = GroqProvider(Settings(groq_api_key="fake-key"))
    state = Agent(provider, _registry_with_calculator()).run("Say hello.")

    assert state.finished is True
    assert state.final_answer == "Fallback answer."
    assert [
        call.kwargs["model"] for call in mock_client.chat.completions.create.call_args_list[:2]
    ] == ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"]


def test_max_iteration_termination() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-1", name="calculate", arguments={"expression": "1 + 1"}),
            ),
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-2", name="calculate", arguments={"expression": "2 + 2"}),
            ),
            LLMResponse(model="fake-model", content="This should never be reached."),
        ]
    )

    state = Agent(provider, _registry_with_calculator(), max_iterations=2).run("Keep going.")

    assert state.finished is False
    assert state.final_answer is None
    assert state.iteration == 2


def test_first_message_is_stored() -> None:
    provider = ScriptedProvider([LLMResponse(model="fake-model", content="Hi there.")])
    state = Agent(provider, _registry_with_calculator()).run("Hello")

    assert state.messages[0]["role"] == "system"
    assert state.messages[1] == {"role": "user", "content": "Hello"}
    assert state.messages[2] == {"role": "assistant", "content": "Hi there."}


def test_second_message_can_use_the_first_message() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(model="fake-model", content="Paris."),
            LLMResponse(model="fake-model", content="You asked about France's capital."),
        ]
    )
    agent = Agent(provider, _registry_with_calculator())

    agent.run("What is the capital of France?")
    agent.run("What did I ask earlier?")

    second_call_messages = provider.call_messages[1]
    assert {"role": "user", "content": "What is the capital of France?"} in second_call_messages
    assert {"role": "assistant", "content": "Paris."} in second_call_messages
    assert {"role": "user", "content": "What did I ask earlier?"} in second_call_messages


def test_assistant_responses_are_preserved_across_turns() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(model="fake-model", content="First answer."),
            LLMResponse(model="fake-model", content="Second answer."),
        ]
    )
    agent = Agent(provider, _registry_with_calculator())

    first_state = agent.run("First question")
    second_state = agent.run("Second question")

    assert {"role": "assistant", "content": "First answer."} in first_state.messages
    assert {"role": "assistant", "content": "First answer."} in second_state.messages
    assert {"role": "assistant", "content": "Second answer."} in second_state.messages


def test_tool_calls_and_results_remain_in_history() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                tool_call=ToolCall(id="call-1", name="calculate", arguments={"expression": "7 + 8"}),
            ),
            LLMResponse(model="fake-model", content="15"),
            LLMResponse(model="fake-model", content="Using previous tool result."),
        ]
    )
    agent = Agent(provider, _registry_with_calculator())

    first_state = agent.run("Calculate 7 + 8")
    agent.run("Use prior context")

    assert any(message.get("role") == "tool" and message.get("content") == "15" for message in first_state.messages)
    second_call_messages = provider.call_messages[2]
    assert any(message.get("role") == "tool" and message.get("content") == "15" for message in second_call_messages)
    assert any(
        message.get("role") == "assistant" and message.get("tool_calls")
        for message in second_call_messages
    )


def test_multiple_turns_work() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(model="fake-model", content="A1"),
            LLMResponse(model="fake-model", content="A2"),
            LLMResponse(model="fake-model", content="A3"),
        ]
    )
    agent = Agent(provider, _registry_with_calculator())

    third_state = None
    agent.run("Q1")
    agent.run("Q2")
    third_state = agent.run("Q3")

    assert provider.calls == 3
    assert third_state.final_answer == "A3"
    assert third_state.messages[-1] == {"role": "assistant", "content": "A3"}


def test_reset_clears_history() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(model="fake-model", content="Before reset."),
            LLMResponse(model="fake-model", content="After reset."),
        ]
    )
    agent = Agent(provider, _registry_with_calculator())

    agent.run("First")
    agent.reset()
    state_after_reset = agent.run("Second")

    second_call_messages = provider.call_messages[1]
    assert second_call_messages == [
        {"role": "system", "content": second_call_messages[0]["content"]},
        {"role": "user", "content": "Second"},
    ]
    assert {"role": "assistant", "content": "Before reset."} not in state_after_reset.messages


def test_retrieved_memory_reaches_llm_context() -> None:
    provider = ScriptedProvider([LLMResponse(model="fake-model", content="Answer with context.")])
    agent = Agent(provider, _registry_with_calculator())
    agent.save_user_memory("User prefers Python examples.")
    agent.save_user_memory("User enjoys gardening.")

    agent.run("Show me Python tips")

    call_messages = provider.call_messages[0]
    memory_context_messages = [
        message
        for message in call_messages
        if message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and str(message.get("content")).startswith("Relevant user memories:")
    ]

    assert len(memory_context_messages) == 1
    memory_context = str(memory_context_messages[0]["content"])
    assert "User prefers Python examples." in memory_context
    assert "User enjoys gardening." not in memory_context


def test_memory_context_deduplicates_duplicate_memories() -> None:
    provider = ScriptedProvider([LLMResponse(model="fake-model", content="Done.")])
    agent = Agent(provider, _registry_with_calculator())
    agent.save_user_memory("User prefers concise responses.")
    agent.save_user_memory("User prefers concise responses.")

    agent.run("Any concise suggestion?")

    call_messages = provider.call_messages[0]
    memory_context = next(
        str(message["content"])
        for message in call_messages
        if message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and str(message.get("content")).startswith("Relevant user memories:")
    )
    assert memory_context.count("User prefers concise responses.") == 1
