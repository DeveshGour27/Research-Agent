"""Tests for Phase 6.5 — Evaluation & Golden Cases."""

import pytest
import json
from unittest.mock import Mock

from app.evaluation.contracts import GoldenCase, GoldenConstraints, DeterministicGateResult
from app.evaluation.engine import EvaluationEngine
from app.evaluation.replay import ReplayRecord, ReplayContext, RecordedLLMResponse, RecordedAgentResult
from app.agent.supervisor import Supervisor
from app.agent.llm_planner import LLMPlanner
from app.agent.executor import PlanExecutor
from app.agent.registry import AgentRegistry
from app.agent.routing import CapabilityRouter
from app.agent.communicator import InProcessCommunicator
from app.agent.specialized.web_agent import WebResearchAgent
from app.agent.specialized.rag_agent import RAGAgent
from app.llm.base import LLMProvider, ChatMessage, ChatResponse, LLMResponse, ToolCall

# ----------------------------------------------------------------------------
# Test Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def supervisor():
    registry = AgentRegistry()
    registry.register(WebResearchAgent())
    registry.register(RAGAgent())

    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )
    planner = LLMPlanner(provider=Mock(spec=LLMProvider))

    return Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
    )

def create_replay_record(llm_responses=None, agent_results=None) -> ReplayRecord:
    return ReplayRecord.from_dict({
        "context": {
            "trace_id": "test-trace", "run_id": "test-run", "span_id": "test-span", "parent_span_id": None,
            "execution_id": "test-exec", "correlation_id": None, "user_id": None, "chat_id": None,
            "task": "test task"
        },
        "llm_responses": llm_responses or [],
        "agent_results": agent_results or [],
        "tool_results": []
    })

def create_plan_response(steps: list) -> dict:
    return {
        "model": "test-model",
        "content": "",
        "tool_call": {
            "id": "tc1",
            "name": "submit_plan",
            "arguments": {"steps": steps}
        }
    }

# ----------------------------------------------------------------------------
# Case A: Web Search
# ----------------------------------------------------------------------------
def test_case_a_web_search(supervisor):
    llm_resp = create_plan_response([
        {
            "step_id": "step_1",
            "description": "Find info on web",
            "task_type": "web_search",
            "required_capabilities": ["tool_use"],
            "dependencies": []
        }
    ])
    
    # We mock the worker agent's result to avoid external calls
    agent_resp = {
        "request_input": "Find info on web",
        "output": "Web results found",
        "success": True
    }
    
    record = create_replay_record(llm_responses=[llm_resp], agent_results=[agent_resp])
    
    case = GoldenCase(
        case_id="case_a",
        task_input="Find info on web",
        constraints=GoldenConstraints(
            expected_success=True,
            expected_task_types=["web_search"],
            expected_capabilities=["tool_use"],
            expected_plan_steps=1,
            expected_agent="production-supervisor-agent"
        ),
        replay_record=record
    )
    
    engine = EvaluationEngine(agent_class=None, agent_kwargs={})
    engine._agent_class = lambda **kwargs: supervisor  # override initialization
    
    res = engine.evaluate_case(case)
    assert res.deterministic.passed is True, f"Failures: {res.deterministic.failures}"

# ----------------------------------------------------------------------------
# Case B: RAG Search
# ----------------------------------------------------------------------------
def test_case_b_rag_search(supervisor):
    llm_resp = create_plan_response([
        {
            "step_id": "step_1",
            "description": "Find internal docs",
            "task_type": "rag_search",
            "required_capabilities": ["retrieval"],
            "dependencies": []
        }
    ])
    
    agent_resp = {
        "request_input": "Find internal docs",
        "output": "Internal docs found",
        "success": True
    }
    
    record = create_replay_record(llm_responses=[llm_resp], agent_results=[agent_resp])
    
    case = GoldenCase(
        case_id="case_b",
        task_input="Find internal docs",
        constraints=GoldenConstraints(
            expected_success=True,
            expected_task_types=["rag_search"],
            expected_capabilities=["retrieval"],
            expected_plan_steps=1
        ),
        replay_record=record
    )
    
    engine = EvaluationEngine(agent_class=None, agent_kwargs={})
    engine._agent_class = lambda **kwargs: supervisor
    
    res = engine.evaluate_case(case)
    assert res.deterministic.passed is True, f"Failures: {res.deterministic.failures}"

# ----------------------------------------------------------------------------
# Case C: Multi-Step Dependency
# ----------------------------------------------------------------------------
def test_case_c_multi_step_dependency(supervisor):
    llm_resp = create_plan_response([
        {
            "step_id": "step_1",
            "description": "Web research",
            "task_type": "web_search",
            "required_capabilities": ["tool_use"],
            "dependencies": []
        },
        {
            "step_id": "step_2",
            "description": "RAG search based on web",
            "task_type": "rag_search",
            "required_capabilities": ["retrieval"],
            "dependencies": ["step_1"]
        }
    ])
    
    agent_resp_1 = {"request_input": "Web research", "output": "Web info", "success": True}
    agent_resp_2 = {"request_input": "RAG search based on web\n\nContext from prerequisite steps:\n- step_1: Web info", "output": "RAG info", "success": True}
    
    record = create_replay_record(llm_responses=[llm_resp], agent_results=[agent_resp_1, agent_resp_2])
    
    case = GoldenCase(
        case_id="case_c",
        task_input="Do multi step",
        constraints=GoldenConstraints(
            expected_success=True,
            expected_task_types=["web_search", "rag_search"],
            expected_capabilities=["tool_use", "retrieval"],
            expected_plan_steps=2
        ),
        replay_record=record
    )
    
    engine = EvaluationEngine(agent_class=None, agent_kwargs={})
    engine._agent_class = lambda **kwargs: supervisor
    
    res = engine.evaluate_case(case)
    assert res.deterministic.passed is True, f"Failures: {res.deterministic.failures}"

# ----------------------------------------------------------------------------
# Case D: Invalid Plan (Missing Capability)
# ----------------------------------------------------------------------------
def test_case_d_invalid_plan(supervisor):
    llm_resp = create_plan_response([
        {
            "step_id": "step_1",
            "description": "Find info on web",
            "task_type": "web_search",
            "required_capabilities": ["unknown_magic"],
            "dependencies": []
        }
    ])
    
    # We do not need agent_resp because the plan will fail validation
    record = create_replay_record(llm_responses=[llm_resp])
    
    case = GoldenCase(
        case_id="case_d",
        task_input="Find info on web",
        constraints=GoldenConstraints(
            expected_success=False  # Invalid plan causes execution to fail
        ),
        replay_record=record
    )
    
    engine = EvaluationEngine(agent_class=None, agent_kwargs={})
    engine._agent_class = lambda **kwargs: supervisor
    
    res = engine.evaluate_case(case)
    # The evaluation itself passes because it successfully rejected the bad plan
    assert res.deterministic.passed is True, f"Failures: {res.deterministic.failures}"

# ----------------------------------------------------------------------------
# Case E: Replanning
# ----------------------------------------------------------------------------
def test_case_e_replanning(supervisor):
    llm_resp_1 = create_plan_response([
        {
            "step_id": "step_1",
            "description": "Initial try",
            "task_type": "web_search",
            "required_capabilities": ["tool_use"],
            "dependencies": []
        }
    ])
    
    llm_resp_2 = create_plan_response([
        {
            "step_id": "step_1_alt",
            "description": "Second try",
            "task_type": "rag_search",
            "required_capabilities": ["retrieval"],
            "dependencies": []
        }
    ])
    
    agent_resp_fail = {"request_input": "Initial try", "output": "Failed to fetch", "success": False}
    agent_resp_success = {"request_input": "Second try", "output": "RAG success", "success": True}
    
    record = create_replay_record(
        llm_responses=[llm_resp_1, llm_resp_2], 
        agent_results=[agent_resp_fail, agent_resp_success]
    )
    
    case = GoldenCase(
        case_id="case_e",
        task_input="Try and fail then succeed",
        constraints=GoldenConstraints(
            expected_success=True,  # the second plan succeeds
            expected_task_types=["rag_search"], # We expect the executed plan to be the second plan
            expected_capabilities=["retrieval"],
            expected_plan_steps=1
        ),
        replay_record=record
    )
    
    engine = EvaluationEngine(agent_class=None, agent_kwargs={})
    engine._agent_class = lambda **kwargs: supervisor
    
    res = engine.evaluate_case(case)
    assert res.deterministic.passed is True, f"Failures: {res.deterministic.failures}"

# ----------------------------------------------------------------------------
# Case F: Deterministic Evaluation Checks
# ----------------------------------------------------------------------------
def test_case_f_deterministic_evaluation_fails_on_wrong_task(supervisor):
    llm_resp = create_plan_response([
        {
            "step_id": "step_1",
            "description": "Find info on web",
            "task_type": "web_search",
            "required_capabilities": ["tool_use"],
            "dependencies": []
        }
    ])
    
    agent_resp = {"request_input": "Find info on web", "output": "Web results", "success": True}
    record = create_replay_record(llm_responses=[llm_resp], agent_results=[agent_resp])
    
    case = GoldenCase(
        case_id="case_f",
        task_input="Find info on web",
        constraints=GoldenConstraints(
            expected_success=True,
            expected_task_types=["rag_search"]  # Deliberate mismatch
        ),
        replay_record=record
    )
    
    engine = EvaluationEngine(agent_class=None, agent_kwargs={})
    engine._agent_class = lambda **kwargs: supervisor
    
    res = engine.evaluate_case(case)
    assert res.deterministic.passed is False
    assert any("Expected task types" in f for f in res.deterministic.failures)
