"""State tracked across one agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.llm.base import ToolCall
from app.tools.base import ToolResult


@dataclass
class AgentState:
    """Mutable record of everything that occurred during one agent run.

    Attributes:
        messages:          Full conversation history in OpenAI-format dicts,
                           including system, user, assistant, and tool messages.
        iteration:         Number of LLM calls made so far.
        tool_calls:        Ordered list of tool-call requests the LLM issued.
        observations:      Ordered list of tool results fed back to the LLM.
        prompt_tokens:     Cumulative prompt tokens across all LLM calls.
        completion_tokens: Cumulative completion tokens across all LLM calls.
        finished:          ``True`` when the loop produced a final text answer.
        final_answer:      The agent's last text response.  ``None`` when the
                           loop was terminated by the max-iteration guard
                           without producing an answer.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[ToolResult] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finished: bool = False
    final_answer: str | None = None
