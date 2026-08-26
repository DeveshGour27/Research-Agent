from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
from app.llm.models import ModelResponse, ToolCall, ModelRequest, TaskType
from app.llm.provider import ModelProvider

ChatRole = Literal["user", "assistant", "system"]

@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str

# Use the new ones
ChatResponse = ModelResponse
LLMResponse = ModelResponse

class LLMProvider(ModelProvider):
    """Legacy adapter for ModelProvider"""
    @property
    def provider_id(self) -> str: return "legacy"

    def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        req = ModelRequest(messages=messages, tools=tools, task_type=TaskType.TOOL_CALLING if tools else TaskType.GENERAL)
        return self.generate(req, "legacy_model")

    def generate(self, request_or_messages: Any, model_id: str = "legacy_model") -> Any:
        if isinstance(request_or_messages, ModelRequest):
            return self._generate_legacy(request_or_messages, model_id)
        # Sequence[ChatMessage]
        req = ModelRequest(messages=[{"role": m.role, "content": m.content} for m in request_or_messages], task_type=TaskType.GENERAL)
        return self._generate_legacy(req, model_id)

    def _generate_legacy(self, request: ModelRequest, model_id: str) -> ModelResponse:
        # Default mock behavior or abstract? Tests might mock generate or generate_with_tools
        pass
