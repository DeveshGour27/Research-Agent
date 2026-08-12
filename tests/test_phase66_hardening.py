import json
from unittest.mock import Mock, patch
import urllib.error

import pytest

from app.agent.contracts import AgentRequest, AgentExecutionContext, AgentExecutionError
from app.agent.llm_planner import LLMPlanner
from app.agent.specialized.web_agent import WebResearchAgent
from app.agent.supervisor import Supervisor
from app.config import settings
from app.exceptions import PlanCreationError, ToolExecutionError
from app.tools.web_search import WebSearchTool


class MockPlannerForHardening:
    """A planner that raises validation or creation errors to test Supervisor."""
    
    def __init__(self, error_to_raise: Exception):
        self._error = error_to_raise

    def generate_plan(self, *args, **kwargs):
        raise self._error


def test_planner_catches_invalid_json():
    """Verify Supervisor gracefully catches PlanCreationError when LLM output is malformed."""
    registry = Mock()
    registry.get_all.return_value = [Mock()]
    planner = MockPlannerForHardening(PlanCreationError("LLM returned malformed JSON"))
    executor = Mock()
    
    supervisor = Supervisor(registry=registry, planner=planner, plan_executor=executor)
    
    request = AgentRequest(
        input_text="Do something", 
        context=AgentExecutionContext(task="test")
    )
    
    result = supervisor.execute(request)
    
    assert result.success is False
    assert result.error is not None
    assert "PlanCreationError" in result.error.details.get("error_type", "") or "Plan" in result.error.message


def test_planner_enforces_max_steps():
    """Verify LLMPlanner raises PlanCreationError when exceeding max_plan_steps."""
    # We mock the LLMProvider to return a huge plan
    mock_provider = Mock()
    huge_plan_steps = [{"step_id": f"s{i}", "description": "desc", "task_type": "web_search"} for i in range(settings.max_plan_steps + 1)]
    
    from app.llm.base import ToolCall
    mock_response = Mock()
    mock_response.tool_call = ToolCall(
        id="call_1",
        name="submit_plan",
        arguments={"steps": huge_plan_steps}
    )
    mock_provider.generate_with_tools.return_value = mock_response
    
    planner = LLMPlanner(provider=mock_provider)
    
    with pytest.raises(PlanCreationError, match="exceeds maximum allowed steps"):
        planner.generate_plan(goal="Do a lot of things", context=AgentExecutionContext(task="test"))


@patch("app.tools.web_search.settings")
@patch("urllib.request.urlopen")
def test_web_search_timeout(mock_urlopen, mock_settings):
    """Verify WebSearchTool safely handles timeouts."""
    mock_settings.web_search_provider = "tavily"
    mock_settings.web_search_api_key = "secret_key"
    mock_settings.web_search_max_results = 5
    mock_settings.web_search_timeout_seconds = 10
    mock_urlopen.side_effect = TimeoutError("Connection timed out")
    
    tool = WebSearchTool()
    
    with pytest.raises(ToolExecutionError, match="Web search request timed out."):
        tool.execute(query="test query")
        
    # Also verify it doesn't leak secrets in exceptions
    try:
        tool.execute(query="test query")
    except ToolExecutionError as e:
        assert "secret_key" not in str(e)


def test_agent_isolation():
    """Verify WebResearchAgent cannot route to or instantiate rag_search capabilities."""
    agent = WebResearchAgent()
    
    # 1. Check identity/capabilities statically
    assert "rag_search" not in agent.capabilities.task_types
    assert agent.capabilities.retrieval is False
    
    # 2. Check tool registry isolation
    assert agent._tool_registry.get("web_search") is not None
    
    from app.exceptions import ToolNotFoundError
    with pytest.raises(ToolNotFoundError):
        agent._tool_registry.get("rag_search")
