"""Groq implementation of the provider-neutral chat interface."""

from __future__ import annotations

import json
from typing import Any, Sequence

from app.config import Settings
from app.exceptions import ConfigurationError, LLMAPIError, LLMResponseParseError
from app.llm.base import ChatMessage, ChatResponse, LLMProvider, LLMResponse, ToolCall
from app.logger import get_logger


logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    """Generate text through Groq's Chat Completions API.

    The SDK is imported only when this provider is constructed, which prevents
    other providers and unit tests from depending on the Groq package.
    """

    def __init__(self, configuration: Settings) -> None:
        if not configuration.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY must be configured to use the Groq provider.",
                details={"provider": "groq"},
            )

        try:
            from groq import Groq
        except ImportError as error:
            raise ConfigurationError(
                "The Groq SDK is not installed. Install project dependencies first.",
                details={"provider": "groq"},
            ) from error

        self._configuration = configuration
        self._client: Any = Groq(
            api_key=configuration.groq_api_key,
            timeout=configuration.llm_request_timeout_seconds,
        )

    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Generate through the configured Groq models, with transient fallback."""
        for model in self._iter_models_with_fallback():
            response = self._generate_for_model(messages, model)
            if response is not None:
                return response

        raise AssertionError("Configured Groq model sequence must not be empty.")

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Generate through Groq function calling with fallback on transient errors."""
        for model in self._iter_models_with_fallback():
            response = self._generate_with_tools_for_model(messages, tools, model)
            if response is not None:
                return response

        raise AssertionError("Configured Groq model sequence must not be empty.")

    def _iter_models_with_fallback(self) -> tuple[str, str]:
        return (self._configuration.llm_model, self._configuration.llm_fallback_model)

    def _generate_for_model(self, messages: Sequence[ChatMessage], model: str) -> ChatResponse | None:
        """Send one normalized request to a specified Groq model."""
        payload = [{"role": message.role, "content": message.content} for message in messages]

        response: Any
        try:
            response = self._request_chat_completion(model=model, messages=payload, tools=None)
        except LLMAPIError as error:
            if self._should_try_fallback(model=model, error=error):
                return None
            raise

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as error:
            raise LLMResponseParseError(
                "The Groq response structure was unexpected or choice was empty.",
                details={"provider": "groq", "model": model},
            ) from error

        if content is None:
            raise LLMResponseParseError(
                "The Groq response did not contain generated text.",
                details={"provider": "groq", "model": model},
            )

        content = content.strip()
        if not content:
            raise LLMResponseParseError(
                "The Groq response did not contain generated text.",
                details={"provider": "groq", "model": model},
            )

        return ChatResponse(content=content, model=model)

    def _generate_with_tools_for_model(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> LLMResponse | None:
        response: Any
        try:
            response = self._request_chat_completion(model=model, messages=messages, tools=tools)
        except LLMAPIError as error:
            if self._should_try_fallback(model=model, error=error):
                return None
            raise

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        try:
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason
        except (AttributeError, IndexError) as error:
            raise LLMResponseParseError(
                "The Groq response structure was unexpected or choice was empty.",
                details={"provider": "groq", "model": model},
            ) from error

        if finish_reason == "tool_calls":
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                raise LLMResponseParseError(
                    "The Groq response requested a tool call but none were provided.",
                    details={"provider": "groq", "model": model},
                )

            first_tool_call = tool_calls[0]
            function = getattr(first_tool_call, "function", None)
            tool_name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", "{}")
            if not tool_name:
                raise LLMResponseParseError(
                    "The Groq tool call did not include a function name.",
                    details={"provider": "groq", "model": model},
                )

            parsed_arguments: dict[str, Any]
            try:
                candidate = json.loads(raw_arguments or "{}")
                parsed_arguments = candidate if isinstance(candidate, dict) else {}
            except json.JSONDecodeError:
                parsed_arguments = {
                    "__parse_error__": "Invalid JSON arguments",
                    "__raw_arguments__": raw_arguments,
                }

            return LLMResponse(
                model=model,
                tool_call=ToolCall(
                    id=str(getattr(first_tool_call, "id", "")),
                    name=str(tool_name),
                    arguments=parsed_arguments,
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        content = getattr(message, "content", None)
        normalized_content = content.strip() if isinstance(content, str) else ""
        if not normalized_content:
            raise LLMResponseParseError(
                "The Groq response did not contain generated text.",
                details={"provider": "groq", "model": model},
            )

        return LLMResponse(
            model=model,
            content=normalized_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _request_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        try:
            request_payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": self._configuration.llm_temperature,
                "max_tokens": self._configuration.llm_max_output_tokens,
            }
            if tools:
                request_payload["tools"] = tools
            return self._client.chat.completions.create(**request_payload)
        except Exception as error:
            raise LLMAPIError(
                "The Groq request failed.",
                status_code=getattr(error, "status_code", None),
                details={
                    "provider": "groq",
                    "model": model,
                    "error_type": type(error).__name__,
                    "transient": self._is_transient_sdk_error(error),
                },
            ) from error

    def _should_try_fallback(self, model: str, error: LLMAPIError) -> bool:
        is_primary_model = model == self._configuration.llm_model
        if is_primary_model and self._is_transient_error(error):
            logger.warning(
                "Groq primary model failed transiently; using fallback model",
                extra={
                    "provider": "groq",
                    "primary_model": model,
                    "fallback_model": self._configuration.llm_fallback_model,
                    "status_code": error.status_code,
                    "error_type": error.details.get("error_type"),
                },
            )
            return True
        return False

    @staticmethod
    def _is_transient_error(error: LLMAPIError) -> bool:
        """Return whether a normalized Groq API error permits model fallback."""
        return bool(error.details.get("transient"))

    @staticmethod
    def _is_transient_sdk_error(error: Exception) -> bool:
        """Classify only availability failures that are safe to retry on another model."""
        status_code = getattr(error, "status_code", None)
        if status_code == 408 or status_code == 429:
            return True
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return True

        return isinstance(error, (ConnectionError, TimeoutError)) or type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
        }
