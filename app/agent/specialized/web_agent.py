"""WebResearchAgent implementation."""

from __future__ import annotations

from app.agent.contracts import (
    AgentCapabilities,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.state import AgentState
from app.exceptions import AgentExecutionError, ToolExecutionError
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class WebResearchAgent(BaseAgent):
    """Specialized agent for web research."""

    def __init__(self) -> None:
        self._identity = AgentIdentity(
            name="web_research_agent",
            version="1.0.0",
            description="Specialized agent for performing external web research.",
        )
        self._capabilities = AgentCapabilities(
            tool_use=True,
            memory=False,
            multi_turn=False,
            retrieval=False,
            task_types=frozenset({"web_search"}),
        )
        # Strictly scoped registry
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(WebSearchTool())

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        normalized_input = request.input_text.strip()
        if not normalized_input:
            raise AgentExecutionError(
                "Agent request input_text must not be empty.",
                request=request,
            )

        tool = self._tool_registry.get("web_search")

        try:
            # Deterministic execution
            output = tool.execute(query=normalized_input)
            success = True
            error = None
        except ToolExecutionError as e:
            output = None
            success = False
            error = AgentExecutionError(
                "Web search tool execution failed.",
                request=request,
                details={"error_message": str(e)},
            )
        except Exception as e:
            output = None
            success = False
            error = AgentExecutionError(
                "Unexpected error during web search.",
                request=request,
                details={"error_type": type(e).__name__, "message": str(e)},
            )

        state = AgentState(
            finished=success,
            final_answer=output,
        )

        return AgentResult(
            request=request,
            state=state,
            output=output,
            success=success,
            context=request.context,
            error=error,
        )
