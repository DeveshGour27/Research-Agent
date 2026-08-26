"""Tests for Phase 6.4 end-to-end integration."""

import json
from unittest.mock import MagicMock

import pytest

from app.agent.communicator import InProcessCommunicator
from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext
from app.agent.execution_policy import ExecutionPolicy
from app.agent.executor import PlanExecutor
from app.agent.llm_planner import LLMPlanner
from app.agent.plan import Plan, PlanStep, PlanStepStatus
from app.agent.registry import AgentRegistry
from app.agent.replanning import ReplanningPolicy
from app.agent.routing import CapabilityRouter
from app.agent.specialized.rag_agent import RAGAgent
from app.agent.specialized.web_agent import WebResearchAgent
from app.agent.supervisor import Supervisor
from app.agent.validator import PlanValidator
from app.exceptions import AgentExecutionError, PlanCreationError
from app.llm.base import LLMProvider, LLMResponse, ToolCall
from app.retriever import Retriever, RetrievalResult
from rag.hybrid_search import Retrieved
from app.tools.web_search import WebSearchTool


def create_mock_llm_response(arguments_dict: dict) -> LLMResponse:
    return LLMResponse(
        model="test",
        tool_calls=[ToolCall(
            id="call_1",
            name="submit_plan",
            arguments=arguments_dict,
        )],
    )


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(WebResearchAgent())
    mock_retriever = MagicMock(spec=Retriever)
    reg.register(RAGAgent(retriever=mock_retriever))
    return reg


@pytest.fixture
def supervisor(registry: AgentRegistry) -> Supervisor:
    mock_provider = MagicMock(spec=LLMProvider)
    planner = LLMPlanner(provider=mock_provider)
    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )
    
    return Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
    )


def test_supervisor_receives_llm_planner(supervisor: Supervisor) -> None:
    assert isinstance(supervisor._planner, LLMPlanner)
    assert isinstance(supervisor._plan_executor, PlanExecutor)
    assert isinstance(supervisor._registry, AgentRegistry)


def test_end_to_end_web_research_flow(supervisor: Supervisor, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Mock the planner's LLMProvider
    mock_provider = supervisor._planner._provider
    mock_provider.generate.return_value = create_mock_llm_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "Find info on AI",
                "task_type": "web_search",
                "required_capabilities": ["tool_use"],
            }
        ]
    })

    # 2. Mock the WebSearchTool
    mock_tool_execute = MagicMock(return_value="AI is cool.")
    monkeypatch.setattr(WebSearchTool, "execute", mock_tool_execute)

    # 3. Execute Supervisor
    request = AgentRequest(input_text="Research AI")
    result = supervisor.execute(request)

    assert result.success is True
    assert "AI is cool." in result.output
    mock_provider.generate.assert_called_once()
    mock_tool_execute.assert_called_once_with(query="Find info on AI")


def test_end_to_end_rag_flow(supervisor: Supervisor, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Mock the planner's LLMProvider
    mock_provider = supervisor._planner._provider
    mock_provider.generate.return_value = create_mock_llm_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "Find internal docs",
                "task_type": "rag_search",
                "required_capabilities": ["retrieval"],
            }
        ]
    })

    # 2. Mock the Retriever
    rag_agent = supervisor._registry.get("rag_agent")
    rag_agent._retriever.retrieve.return_value = [
        Retrieved(chunk_id="chunk1", score=0.9, source_score=0.9, vector_score=0.9, metadata={"text": "Internal doc content"})
    ]

    # 3. Execute Supervisor
    request = AgentRequest(
        input_text="Find docs",
        context=AgentExecutionContext(task="Find docs", user_id="test_user")
    )
    result = supervisor.execute(request)

    assert result.success is True
    assert "Internal doc content" in result.output
    rag_agent._retriever.retrieve.assert_called_once_with(query="Find internal docs", user_id="test_user")


def test_multi_step_dependency_execution(supervisor: Supervisor, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Mock Planner: Web Search -> RAG Search
    mock_provider = supervisor._planner._provider
    mock_provider.generate.return_value = create_mock_llm_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "Search web",
                "task_type": "web_search",
                "required_capabilities": ["tool_use"],
            },
            {
                "step_id": "step_2",
                "description": "Search rag",
                "task_type": "rag_search",
                "required_capabilities": ["retrieval"],
                "dependencies": ["step_1"],
            }
        ]
    })

    # 2. Mock Tools
    mock_web = MagicMock(return_value="Web output")
    monkeypatch.setattr(WebSearchTool, "execute", mock_web)
    
    rag_agent = supervisor._registry.get("rag_agent")
    rag_agent._retriever.retrieve.return_value = [
        Retrieved(chunk_id="chunk1", score=0.9, source_score=0.9, vector_score=0.9, metadata={"text": "RAG output"})
    ]

    # 3. Execute
    request = AgentRequest(
        input_text="Do complex task",
        context=AgentExecutionContext(task="Complex", user_id="test_user")
    )
    result = supervisor.execute(request)

    assert result.success is True
    assert "Web output" in result.output
    assert "RAG output" in result.output

    # Check that both were called
    mock_web.assert_called_once()
    rag_agent = supervisor._registry.get("rag_agent")
    rag_agent._retriever.retrieve.assert_called_once()


def test_planner_failure_propagation(supervisor: Supervisor) -> None:
    mock_provider = supervisor._planner._provider
    mock_provider.generate.side_effect = Exception("LLM is down")

    request = AgentRequest(input_text="Do something")
    
    result = supervisor.execute(request)
    assert result.success is False
    assert result.error is not None
    assert "LLM is down" in result.error.details.get("message", "") or "LLM is down" in str(result.error.details)


def test_agent_failure_and_replanning_flow(supervisor: Supervisor, monkeypatch: pytest.MonkeyPatch) -> None:
    # Allow 1 replan
    supervisor._execution_policy = ExecutionPolicy(allow_replanning=True, max_replans=1)
    
    # First plan: Web Search
    # Second plan: RAG Search (fallback)
    mock_provider = supervisor._planner._provider
    mock_provider.generate.side_effect = [
        create_mock_llm_response({
            "steps": [
                {"step_id": "step_1", "description": "Search web", "task_type": "web_search"}
            ]
        }),
        create_mock_llm_response({
            "steps": [
                {"step_id": "step_1_alt", "description": "Search rag", "task_type": "rag_search"}
            ]
        })
    ]

    # Mock WebSearch to FAIL
    mock_web = MagicMock(side_effect=Exception("Web search error"))
    monkeypatch.setattr(WebSearchTool, "execute", mock_web)

    # Mock RAG to SUCCEED
    rag_agent = supervisor._registry.get("rag_agent")
    rag_agent._retriever.retrieve.return_value = [
        Retrieved(chunk_id="chunk1", score=0.9, source_score=0.9, vector_score=0.9, metadata={"text": "RAG success"})
    ]

    request = AgentRequest(
        input_text="Search something",
        context=AgentExecutionContext(task="Search something", user_id="test_user")
    )
    result = supervisor.execute(request)

    assert result.success is True
    assert "RAG success" in result.output
    assert mock_provider.generate.call_count == 2
    mock_web.assert_called_once()
    rag_agent._retriever.retrieve.assert_called_once()


def test_context_propagation(supervisor: Supervisor, monkeypatch: pytest.MonkeyPatch) -> None:
    # We will verify that context IDs propagate into the agent's execute()
    
    mock_provider = supervisor._planner._provider
    mock_provider.generate.return_value = create_mock_llm_response({
        "steps": [
            {"step_id": "s1", "description": "desc", "task_type": "web_search"}
        ]
    })
    
    web_agent = supervisor._registry.get("web_research_agent")
    original_execute = web_agent.execute
    
    executed_contexts = []
    
    def mock_agent_execute(request: AgentRequest):
        executed_contexts.append(request.context)
        return original_execute(request)
        
    monkeypatch.setattr(web_agent, "execute", mock_agent_execute)
    mock_web_tool = MagicMock(return_value="output")
    monkeypatch.setattr(WebSearchTool, "execute", mock_web_tool)
    
    context = AgentExecutionContext(task="test task")
    request = AgentRequest(input_text="test task", context=context)
    
    supervisor.execute(request)
    
    assert len(executed_contexts) == 1
    child_context = executed_contexts[0]
    
    assert child_context.execution_id == context.execution_id
    assert child_context.trace_id == context.trace_id
    assert child_context.run_id == context.run_id


def test_isolation_boundaries(registry: AgentRegistry) -> None:
    web_agent = registry.get("web_research_agent")
    assert isinstance(web_agent, WebResearchAgent)
    
    rag_agent = registry.get("rag_agent")
    assert isinstance(rag_agent, RAGAgent)
    
    # WebResearchAgent should have exactly 1 tool: web_search
    tools = web_agent._tool_registry._tools
    assert len(tools) == 1
    assert "web_search" in tools
    
    # RAGAgent does not even have a tool registry
    assert not hasattr(rag_agent, "_tool_registry")
    assert hasattr(rag_agent, "_retriever")
