from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

class ModelCapability(Enum):
    TEXT_GENERATION = auto()
    REASONING = auto()
    TOOL_CALLING = auto()
    STRUCTURED_OUTPUT = auto()
    LONG_CONTEXT = auto()

class TaskType(Enum):
    PLANNING = auto()
    TOOL_CALLING = auto()
    RAG_SYNTHESIS = auto()
    REFLECTION = auto()
    GENERAL = auto()

@dataclass
class ToolCall:
    """Normalized tool call request from a model."""
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelRequest:
    messages: list[dict[str, Any]]
    system_instructions: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict | None = None
    response_format: Any | None = None
    required_capabilities: set[ModelCapability] = field(default_factory=set)
    task_type: TaskType = TaskType.GENERAL
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    raw_response: dict[str, Any] | None = None
    estimated_cost: float | None = None
    currency: str | None = None

    @property
    def tool_call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

@dataclass
class ModelProfile:
    provider: str
    model_id: str
    capabilities: set[ModelCapability]
    context_limit: int = 8192
    default_temperature: float = 0.7
    max_output_tokens: int = 2048
    priority: int = 100
    enabled: bool = True
    input_cost_per_m: float = 0.0
    output_cost_per_m: float = 0.0

