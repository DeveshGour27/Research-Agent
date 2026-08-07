"""Provider-neutral contracts for language-model chat."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One text message in a conversation supplied to a chat provider."""

    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A normalized text response returned from a chat provider."""

    content: str
    model: str


@dataclass
class ToolCall:
    """A request from the model to invoke a named tool with arguments."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A normalized response from a tool-aware LLM call.

    Exactly one of ``content`` or ``tool_call`` will be non-``None`` in
    normal operation.  ``prompt_tokens`` and ``completion_tokens`` carry
    usage data when the provider makes it available.
    """

    model: str
    content: str | None = None
    tool_call: ToolCall | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    """Abstract interface implemented by each supported LLM provider."""

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Generate one assistant response for an ordered conversation."""

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Generate a response, optionally invoking a tool.

        The default implementation ignores *tools* and wraps
        :meth:`generate`, so providers without native function-calling
        support remain compatible with the agent loop.

        Providers with native tool support (e.g. Groq, OpenAI) should
        override this to return a :class:`ToolCall` when the model
        elects to use one.

        Args:
            messages: OpenAI-format message dicts
                      ``{"role": "user"|"assistant"|"tool", "content": ...}``.
            tools:    OpenAI-compatible tool-schema list.

        Returns:
            :class:`LLMResponse` with either ``content`` or ``tool_call`` set.
        """
        # Map only text-bearing user/assistant turns to ChatMessage.
        chat_messages: list[ChatMessage] = [
            ChatMessage(role=m["role"], content=m.get("content") or "")  # type: ignore[arg-type]
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        response = self.generate(chat_messages)
        return LLMResponse(model=response.model, content=response.content)
