"""Unit tests for provider-independent chat orchestration."""

from __future__ import annotations

import pytest

from app.exceptions import InputValidationError
from app.llm.base import ChatMessage, ChatResponse, LLMProvider
from app.llm.service import ChatService


class FakeProvider(LLMProvider):
    """Minimal provider used to test dependency injection."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def generate(self, request, model_id="test") -> ChatResponse:
        messages = request.messages
        self.messages = list(messages)
        return ChatResponse(content="Hello", model="fake-model")


def test_chat_appends_user_input_to_history_sent_to_provider() -> None:
    """The service sends existing history and the newly supplied message."""
    provider = FakeProvider()
    service = ChatService(provider)

    result = service.chat([{"role": "user", "content": "Earlier"}], "Current")

    assert result.content == "Hello"
    assert provider.messages == [{"role": "user", "content": "Earlier"}, {"role": "user", "content": "Current"}]


@pytest.mark.parametrize("user_input", ["", "   "])
def test_chat_rejects_empty_user_input(user_input: str) -> None:
    """Invalid requests do not reach a provider."""
    with pytest.raises(InputValidationError):
        ChatService(FakeProvider()).chat([], user_input)
