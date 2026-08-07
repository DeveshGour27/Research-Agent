"""Central registry for discovering and executing agent tools."""

from __future__ import annotations

from app.exceptions import ToolExecutionError, ToolNotFoundError
from app.llm.base import ToolCall
from app.logger import get_logger
from app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    """Register tools, expose their schemas, and dispatch tool calls.

    Usage::

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        # Pass schemas to the LLM
        schemas = registry.schemas()

        # Execute a call returned by the LLM
        result = registry.execute(tool_call)
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, tool: BaseTool) -> None:
        """Register *tool* under its :attr:`~app.tools.base.BaseTool.name`."""
        self._tools[tool.name] = tool
        logger.debug("Tool registered", extra={"tool_name": tool.name})

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> BaseTool:
        """Return the tool registered under *name*.

        Raises:
            :class:`~app.exceptions.ToolNotFoundError`: When no tool matches.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered.",
                details={"tool_name": name, "registered": sorted(self._tools)},
            )

    def schemas(self) -> list[dict]:  # type: ignore[type-arg]
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch *tool_call* to the matching tool.

        Errors — unknown tool, execution failure, or unexpected exception —
        are caught and surfaced as an error :class:`~app.tools.base.ToolResult`
        so the agent loop can feed the failure back to the LLM rather than
        crash.
        """
        try:
            tool = self.get(tool_call.name)
        except ToolNotFoundError:
            logger.warning(
                "Tool not found during execution",
                extra={"tool_name": tool_call.name},
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: tool '{tool_call.name}' is not registered.",
                is_error=True,
            )

        logger.debug(
            "Executing tool",
            extra={"tool_name": tool_call.name, "arguments": tool_call.arguments},
        )
        try:
            content = tool.execute(**tool_call.arguments)
            logger.debug("Tool succeeded", extra={"tool_name": tool_call.name})
            return ToolResult(tool_call_id=tool_call.id, content=content, is_error=False)

        except ToolExecutionError as error:
            logger.warning(
                "Tool execution error",
                extra={"tool_name": tool_call.name, "error": error.message},
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {error.message}",
                is_error=True,
            )
        except Exception as error:
            logger.exception(
                "Unexpected tool error",
                extra={"tool_name": tool_call.name, "error_type": type(error).__name__},
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: unexpected failure in '{tool_call.name}': {error}",
                is_error=True,
            )
