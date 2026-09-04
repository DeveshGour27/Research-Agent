"""WebResearchAgent implementation."""

from __future__ import annotations

import json
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
from app.llm.models import ModelRequest, TaskType
from app.logger import get_logger

logger = get_logger(__name__)

class WebResearchAgent(BaseAgent):
    """Specialized agent for web research."""

    def __init__(self, gateway=None) -> None:
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
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(WebSearchTool())
        self._gateway = gateway

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
            # Execute tool to get raw JSON results
            raw_output = tool.execute(query=normalized_input)
            
            # Format output using LLM if available, otherwise fallback to raw json
            if self._gateway and raw_output != "No results found.":
                prompt_instruction = (
                    "You are the AI Research Assistant. Given the user's query and the following raw web search results, "
                    "synthesize a natural, conversational, and informative answer. Cite your sources where appropriate. "
                    "Always format your responses using clean, standard Markdown. Use standard hyphens and spaces instead of non-breaking or obscure unicode characters. Avoid returning raw JSON."
                )
                
                llm_request = ModelRequest(
                    messages=[
                        {"role": "system", "content": prompt_instruction},
                        {"role": "user", "content": f"Query: {normalized_input}\n\nSearch Results: {raw_output}"},
                    ],
                    task_type=TaskType.GENERAL,
                )
                response = self._gateway.generate(llm_request)
                output = response.content or raw_output
            else:
                output = raw_output
                
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
