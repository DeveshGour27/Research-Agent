"""ReAct-style agent loop: user → LLM → tool? → observation → LLM → answer."""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.llm.router import LLMRouter
from app.logger import get_logger
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from app.agent.state import AgentState
from app.agent.execution_context import AgentExecutionContext
from app.hitl.service import HITLService
from app.agent.execution_context import AgentExecutionContext
from app.hitl.service import HITLService

logger = get_logger(__name__)

_SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant. You have access to tools that you can use "
    "to answer questions accurately and efficiently. Use a tool whenever it is "
    "the most efficient way to answer. Once you have all the information you "
    "need, respond with a clear, direct final answer — do not call another tool."
)


class AgentLoop:
    """Execute the tool-use loop for a single user request.

    The loop runs at most *max_iterations* times.  On each iteration it:

    1. Calls the LLM with the current message history and tool schemas.
    2. If the LLM returns a **tool call** — executes the tool, appends the
       observation to the history, and continues.
    3. If the LLM returns a **text answer** — stores it as the final answer
       and stops.

    If *max_iterations* is exhausted before a text answer is produced,
    :attr:`~app.agent.state.AgentState.finished` is left ``False``.

    Args:
        router:         Routes each LLM call to the configured provider.
        registry:       Holds registered tools and dispatches tool calls.
        max_iterations: Maximum number of LLM round-trips (tool steps count
                        as individual iterations).
    """

    def __init__(
        self,
        router: LLMRouter,
        registry: ToolRegistry,
        max_iterations: int | None = None,
        hitl_service: HITLService | None = None,
    ) -> None:
        self._router = router
        self._registry = registry
        self._max_iterations = max_iterations if max_iterations is not None else settings.max_retries * 3
        self._hitl_service = hitl_service

    @staticmethod
    def initial_messages() -> list[dict[str, Any]]:
        """Return the initial system message list for a new conversation."""
        return [{"role": "system", "content": _SYSTEM_PROMPT}]

    def run(self, messages: list[dict[str, Any]], context: AgentExecutionContext | None = None) -> AgentState:
        """Run the loop for the supplied *messages* and return final state."""
        state = AgentState(messages=[*messages])
        tool_schemas = self._registry.schemas()

        # RECOVERY: Check if the last message was an unresolved tool call request from the assistant.
        # This occurs if the job was paused for human approval and is now resuming.
        from app.llm.base import ToolCall
        from app.exceptions import AgentHITLPauseException

        if state.messages and state.messages[-1].get("role") == "assistant" and state.messages[-1].get("tool_calls"):
            logger.info("Resuming execution with unresolved tool calls from previous run.")
            last_msg = state.messages[-1]
            for tc in last_msg["tool_calls"]:
                tool_call = ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"])
                )
                try:
                    observation = self._registry.execute(tool_call, context=context, hitl_service=self._hitl_service)
                    state.observations.append(observation)
                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": observation.tool_call_id,
                            "content": observation.content,
                        }
                    )
                except AgentHITLPauseException as e:
                    e.partial_state = state
                    raise

        for _ in range(self._max_iterations):
            state.iteration += 1
            logger.debug(
                "Agent loop iteration",
                extra={"iteration": state.iteration, "messages": len(state.messages)},
            )

            llm_response = self._router.generate(state.messages, tool_schemas)
            state.prompt_tokens += llm_response.prompt_tokens
            state.completion_tokens += llm_response.completion_tokens

            # ── Final text answer ──────────────────────────────────────────
            if llm_response.tool_call is None:
                state.final_answer = llm_response.content
                state.finished = True
                state.messages.append(
                    {"role": "assistant", "content": llm_response.content}
                )
                logger.info(
                    "Agent finished with answer",
                    extra={
                        "iterations": state.iteration,
                        "tool_calls": len(state.tool_calls),
                        "prompt_tokens": state.prompt_tokens,
                        "completion_tokens": state.completion_tokens,
                    },
                )
                break

            # ── Tool call ──────────────────────────────────────────────────
            tool_call = llm_response.tool_call
            state.tool_calls.append(tool_call)

            logger.info(
                "Agent invoking tool",
                extra={"tool_name": tool_call.name, "iteration": state.iteration},
            )

            # Append the assistant message that requests the tool call.
            state.messages.append(self._build_tool_call_message(tool_call))

            # Malformed tool JSON arguments are surfaced as tool observations.
            if "__parse_error__" in tool_call.arguments:
                observation = ToolResult(
                    tool_call_id=tool_call.id,
                    content=(
                        "Error: malformed tool arguments. "
                        f"{tool_call.arguments.get('__parse_error__')}"
                    ),
                    is_error=True,
                )
            else:
                # Execute the tool; errors are wrapped in an error ToolResult.
                try:
                    observation = self._registry.execute(tool_call, context=context, hitl_service=self._hitl_service)
                except AgentHITLPauseException as e:
                    e.partial_state = state
                    raise
            state.observations.append(observation)

            # Append the tool result so the LLM sees the observation next turn.
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": observation.tool_call_id,
                    "content": observation.content,
                }
            )

        else:
            # The for-loop exhausted max_iterations without a final answer.
            logger.warning(
                "Agent loop hit max iterations without producing a final answer",
                extra={"max_iterations": self._max_iterations},
            )

        return state

    @staticmethod
    def _build_tool_call_message(tool_call: Any) -> dict[str, Any]:
        """Build the OpenAI-format assistant message for a tool-call request."""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
            ],
        }
