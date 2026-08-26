from __future__ import annotations
import json
from typing import Any
from app.config import Settings
from app.exceptions import ConfigurationError, LLMAPIError, LLMResponseParseError
from app.llm.models import ModelRequest, ModelResponse, ToolCall
from app.llm.provider import ModelProvider
from app.logger import get_logger

logger = get_logger(__name__)

class OpenAIProvider(ModelProvider):
    @property
    def provider_id(self) -> str:
        return "openai"

    def __init__(self, configuration: Settings) -> None:
        if not getattr(configuration, "openai_api_key", None):
            raise ConfigurationError("openai_api_key must be configured.", details={"provider": "openai"})
        try:
            import openai
            self._client = openai.OpenAI(api_key=configuration.openai_api_key, max_retries=0)
        except ImportError as e:
            raise ConfigurationError("The 'openai' package is not installed.") from e

    def generate(self, request: ModelRequest, model_id: str) -> ModelResponse:
        messages = list(request.messages)
        if request.system_instructions:
            messages.insert(0, {"role": "system", "content": request.system_instructions})

        payload = {
            "model": model_id,
            "messages": messages,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.response_format:
            payload["response_format"] = request.response_format

        try:
            raw_response = self._client.chat.completions.create(**payload)
        except Exception as error:
            transient = self._is_transient_sdk_error(error)
            raise LLMAPIError(
                "The OpenAI request failed.",
                status_code=getattr(error, "status_code", None),
                details={
                    "provider": "openai",
                    "model": model_id,
                    "error_type": type(error).__name__,
                    "transient": transient,
                },
            ) from error

        return self._parse_response(raw_response, model_id)

    def _parse_response(self, raw_response: Any, model_id: str) -> ModelResponse:
        usage = getattr(raw_response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        try:
            choice = raw_response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason
        except (AttributeError, IndexError) as error:
            raise LLMResponseParseError(
                "The OpenAI response structure was unexpected or choice was empty.",
                details={"provider": "openai", "model": model_id},
            ) from error

        content = getattr(message, "content", None)
        normalized_content = content.strip() if isinstance(content, str) else None

        tool_calls = []
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                function = getattr(tc, "function", None)
                if not function:
                    continue
                name = getattr(function, "name", "")
                raw_args = getattr(function, "arguments", "{}")
                try:
                    args = json.loads(raw_args or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {"__parse_error__": "Invalid JSON", "__raw_arguments__": raw_args}
                
                tool_calls.append(ToolCall(id=getattr(tc, "id", ""), name=name, arguments=args))

        return ModelResponse(
            content=normalized_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model_id,
            provider="openai",
            raw_response=None,
        )

    @staticmethod
    def _is_transient_sdk_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code in (408, 429): return True
        if isinstance(status_code, int) and 500 <= status_code <= 599: return True
        return isinstance(error, (ConnectionError, TimeoutError)) or type(error).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
