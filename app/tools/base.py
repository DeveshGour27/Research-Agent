"""Abstract base for all agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """The outcome of executing a tool.

    Attributes:
        tool_call_id: Echoes the LLM's :attr:`~app.llm.base.ToolCall.id`
                      so that the conversation history stays coherent.
        content:      String result (or error description) fed back to the LLM.
        is_error:     ``True`` when *content* describes a failure rather than
                      a successful result.
    """

    tool_call_id: str
    content: str
    is_error: bool = False


class BaseTool(ABC):
    """Contract that every agent tool must implement.

    Concrete subclasses must define the three class attributes below.

    Attributes:
        name:         Unique snake_case identifier used in tool schemas and
                      referenced by the LLM when it elects to call this tool.
        description:  One-sentence description shown to the LLM.
        input_schema: JSON Schema ``object`` describing the tool's parameters
                      (passed directly to the provider's ``tools`` payload).
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: object) -> str:
        """Run the tool with keyword arguments from the parsed tool call.

        Args:
            **kwargs: Tool-specific arguments validated against *input_schema*.

        Returns:
            A plain-string result fed back to the LLM as the tool observation.

        Raises:
            :class:`~app.exceptions.ToolExecutionError`: When the tool cannot
                complete its task for any reason.
        """

    def to_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
