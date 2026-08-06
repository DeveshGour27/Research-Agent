"""OpenAI implementation of the provider-neutral chat interface."""

from __future__ import annotations

from typing import Any, Sequence

from app.config import Settings
from app.exceptions import ConfigurationError, LLMAPIError, LLMResponseParseError

from app.llm.base import ChatMessage, ChatResponse, LLMProvider


class OpenAIProvider(LLMProvider):
    """Generate text through OpenAI's Responses API.

    The SDK is imported only when this provider is constructed, which prevents
    other providers and unit tests from depending on the OpenAI package.
    """

    def __init__(self, configuration: Settings) -> None:
        if not configuration.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY must be configured to use the OpenAI provider.",
                details={"provider": "openai"},
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise ConfigurationError(
                "The OpenAI SDK is not installed. Install project dependencies first.",
                details={"provider": "openai"},
            ) from error

        self._configuration = configuration
        self._client: Any = OpenAI(
            api_key=configuration.openai_api_key,
            timeout=configuration.llm_request_timeout_seconds,
        )

    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Send normalized messages to OpenAI and normalize its text output."""
        try:
            response = self._client.responses.create(
                model=self._configuration.llm_model,
                input=[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                temperature=self._configuration.llm_temperature,
                max_output_tokens=self._configuration.llm_max_output_tokens,
            )
        except Exception as error:
            raise LLMAPIError(
                "The OpenAI request failed.",
                status_code=getattr(error, "status_code", None),
                details={"provider": "openai", "error_type": type(error).__name__},
            ) from error

        content = getattr(response, "output_text", "").strip()
        if not content:
            raise LLMResponseParseError(
                "The OpenAI response did not contain generated text.",
                details={"provider": "openai", "model": self._configuration.llm_model},
            )
        return ChatResponse(content=content, model=self._configuration.llm_model)
