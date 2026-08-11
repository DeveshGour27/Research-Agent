"""Tests for the WebResearchAgent."""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext
from app.agent.specialized.web_agent import WebResearchAgent
from app.exceptions import AgentExecutionError, ToolExecutionError, ToolNotFoundError
from app.tools.web_search import WebSearchTool


def test_web_research_agent_identity_and_capabilities() -> None:
    agent = WebResearchAgent()
    assert agent.identity.name == "web_research_agent"
    assert "web_search" in agent.capabilities.task_types
    assert "rag_search" not in agent.capabilities.task_types
    assert agent.capabilities.tool_use is True

def test_web_research_agent_tool_scoping() -> None:
    agent = WebResearchAgent()
    # It should have WebSearchTool registered, but no other arbitrary tools.
    assert agent._tool_registry.get("web_search") is not None
    with pytest.raises(ToolNotFoundError):
        agent._tool_registry.get("calculator")

@patch.object(WebSearchTool, "execute")
def test_web_research_agent_successful_execution(mock_execute: MagicMock) -> None:
    agent = WebResearchAgent()
    mock_execute.return_value = "Mocked web results"

    context = AgentExecutionContext(task="Search something")
    request = AgentRequest(input_text="AI news", context=context)
    
    result = agent.execute(request)

    mock_execute.assert_called_once_with(query="AI news")
    assert result.success is True
    assert result.output == "Mocked web results"
    assert result.request is request
    assert result.context is context

@patch.object(WebSearchTool, "execute")
def test_web_research_agent_tool_failure(mock_execute: MagicMock) -> None:
    agent = WebResearchAgent()
    mock_execute.side_effect = ToolExecutionError("Timeout", tool_name="web_search")

    request = AgentRequest(input_text="AI news")
    
    result = agent.execute(request)
    
    assert result.success is False
    assert result.output is None
    assert isinstance(result.error, AgentExecutionError)
    assert "Web search tool execution failed" in result.error.message

def test_web_research_agent_empty_input() -> None:
    agent = WebResearchAgent()
    request = AgentRequest(input_text="   ")
    with pytest.raises(AgentExecutionError, match="must not be empty"):
        agent.execute(request)
