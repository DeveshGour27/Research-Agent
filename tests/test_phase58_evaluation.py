"""Tests for Phase 5.8 Step 6 — Evaluation & Golden Dataset."""

import pytest
import json
from unittest.mock import Mock
from typing import Sequence, List, Dict, Any

from app.evaluation.contracts import GoldenCase, GoldenConstraints, DeterministicGateResult, JudgeResult
from app.evaluation.judge import LLMJudge
from app.evaluation.engine import EvaluationEngine
from app.agent.contracts import BaseAgent, AgentRequest, AgentResult, AgentIdentity, AgentCapabilities
from app.agent.state import AgentState
from app.llm.base import LLMProvider, ChatMessage, ChatResponse, LLMResponse, ToolCall
from app.evaluation.replay import ReplayRecord
from app.observability.events import AgentExecutionStartedEvent, ToolCalledEvent, HandoffInitiatedEvent

# ----------------------------------------------------------------------------
# 1. GoldenCase Validation Tests
# ----------------------------------------------------------------------------
def test_valid_golden_case_deserialization():
    data = {
        "case_id": "c1",
        "task_input": "do this",
        "constraints": {
            "expected_agent": "agent1",
            "required_tools": ["tool1", "tool2"],
            "must_not_handoff": True,
            "expected_success": True
        },
        "reference_answer": "done",
        "tags": ["test"]
    }
    
    case = GoldenCase.from_dict(data)
    assert case.case_id == "c1"
    assert case.constraints.expected_agent == "agent1"
    assert "tool1" in case.constraints.required_tools
    assert case.constraints.must_not_handoff is True
    assert case.reference_answer == "done"
    assert "test" in case.tags

def test_invalid_golden_case_missing_fields():
    data = {
        "case_id": "c1"
        # missing task_input, constraints
    }
    with pytest.raises(ValueError):
        GoldenCase.from_dict(data)

def test_golden_case_callable_rejection():
    def malicious(): pass
    
    data = {
        "case_id": "c1",
        "task_input": malicious,
        "constraints": {}
    }
    with pytest.raises(ValueError, match="Callables are not allowed"):
        GoldenCase.from_dict(data)

    data2 = {
        "case_id": "c1",
        "task_input": "task",
        "constraints": {
            "required_tools": [malicious]
        }
    }
    with pytest.raises(ValueError, match="Callables are not allowed"):
        GoldenCase.from_dict(data2)

# ----------------------------------------------------------------------------
# 2. LLMJudge Tests
# ----------------------------------------------------------------------------
class MockLLMProvider(LLMProvider):
    def __init__(self, responses: List[LLMResponse]):
        self.responses = responses
        self.index = 0

    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return ChatResponse(content="not implemented", model="test")

    def generate_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        res = self.responses[self.index]
        self.index += 1
        return res

def test_llm_judge_valid_tool_call():
    provider = MockLLMProvider([
        LLMResponse(
            model="test",
            tool_call=ToolCall(
                id="tc1",
                name="submit_evaluation",
                arguments={"correctness": 4.5, "relevance": 5.0, "instruction_adherence": 3.0, "reasoning": "Good"}
            )
        )
    ])
    judge = LLMJudge(provider)
    case = GoldenCase(case_id="1", task_input="t", constraints=GoldenConstraints())
    res = AgentResult(request=Mock(), state=Mock(), output="output", success=True)
    
    score = judge.evaluate(case, res)
    assert score.correctness == 4.5
    assert score.relevance == 5.0
    assert score.instruction_adherence == 3.0
    assert score.reasoning == "Good"
    assert score.error is None

def test_llm_judge_fallback_json_parsing():
    provider = MockLLMProvider([
        LLMResponse(
            model="test",
            content='Some text. {"correctness": 3.0, "relevance": 2.0, "instruction_adherence": 1.0, "reasoning": "Bad"} more text.'
        )
    ])
    judge = LLMJudge(provider)
    case = GoldenCase(case_id="1", task_input="t", constraints=GoldenConstraints())
    res = AgentResult(request=Mock(), state=Mock(), output="output", success=True)
    
    score = judge.evaluate(case, res)
    assert score.correctness == 3.0
    assert score.error is None

def test_llm_judge_malformed_json():
    provider = MockLLMProvider([
        LLMResponse(
            model="test",
            content='Some text. {"correctness": 3.0, relevance": Bad} more text.' # malformed
        )
    ])
    judge = LLMJudge(provider)
    case = GoldenCase(case_id="1", task_input="t", constraints=GoldenConstraints())
    res = AgentResult(request=Mock(), state=Mock(), output="output", success=True)
    
    score = judge.evaluate(case, res)
    assert score.error == "MALFORMED_OUTPUT"
    assert score.correctness == 0.0

def test_llm_judge_timeout_api_failure():
    class FailingProvider(LLMProvider):
        def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse: return ChatResponse("", "")
        def generate_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse: raise RuntimeError("API Timeout")
        
    judge = LLMJudge(FailingProvider())
    case = GoldenCase(case_id="1", task_input="t", constraints=GoldenConstraints())
    res = AgentResult(request=Mock(), state=Mock(), output="output", success=True)
    
    score = judge.evaluate(case, res)
    assert score.error == "API Timeout"
    assert score.correctness == 0.0

# ----------------------------------------------------------------------------
# 3. Evaluation Engine Tests
# ----------------------------------------------------------------------------
class LiveTestAgent(BaseAgent):
    def __init__(self, **kwargs):
        self._identity = AgentIdentity("live")
        self._capabilities = AgentCapabilities()
    
    @property
    def identity(self): return self._identity
    @property
    def capabilities(self): return self._capabilities
    
    def execute(self, request: AgentRequest) -> AgentResult:
        AgentExecutionStartedEvent(None, None, None, None, agent_name=self.identity.name, input_text=request.input_text).emit()
        
        if request.input_text == "fail":
            return AgentResult(request=request, state=Mock(), output="failed", success=False)
        if request.input_text == "crash":
            raise ValueError("Boom")
            
        if request.input_text == "use_tools":
            ToolCalledEvent(None, None, None, None, tool_name="tool1", arguments={}).emit()
            ToolCalledEvent(None, None, None, None, tool_name="tool2", arguments={}).emit()
            
        if request.input_text == "handoff":
            HandoffInitiatedEvent(None, None, None, None, target_task_type="sub", message="go").emit()
            
        return AgentResult(request=request, state=Mock(), output="live output", success=True)

def test_engine_live_evaluation_success():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="success", constraints=GoldenConstraints(expected_success=True))
    
    summary = engine.run_suite([case])
    assert summary.passed_deterministic == 1
    assert summary.failed_deterministic == 0
    assert summary.results[0].deterministic.passed is True

def test_engine_live_evaluation_mismatch_success():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="fail", constraints=GoldenConstraints(expected_success=True))
    
    summary = engine.run_suite([case])
    assert summary.failed_deterministic == 1
    assert "Expected success=True, got False" in summary.results[0].deterministic.failures

def test_engine_live_evaluation_crash():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="crash", constraints=GoldenConstraints(expected_success=True))
    
    summary = engine.run_suite([case])
    assert summary.failed_deterministic == 1
    assert "Execution resulted in a fatal error." in summary.results[0].deterministic.failures

def test_engine_live_evaluation_expected_agent_mismatch():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="success", constraints=GoldenConstraints(expected_agent="wrong_agent"))
    summary = engine.run_suite([case])
    assert summary.failed_deterministic == 1
    assert any("Expected agent wrong_agent" in f for f in summary.results[0].deterministic.failures)

def test_engine_live_evaluation_required_tools_success():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="use_tools", constraints=GoldenConstraints(required_tools=frozenset(["tool1"])))
    summary = engine.run_suite([case])
    assert summary.passed_deterministic == 1

def test_engine_live_evaluation_required_tools_failure():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="use_tools", constraints=GoldenConstraints(required_tools=frozenset(["tool3"])))
    summary = engine.run_suite([case])
    assert summary.failed_deterministic == 1
    assert any("Missing required tools" in f for f in summary.results[0].deterministic.failures)

def test_engine_live_evaluation_must_not_handoff_failure():
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={})
    case = GoldenCase(case_id="1", task_input="handoff", constraints=GoldenConstraints(must_not_handoff=True))
    summary = engine.run_suite([case])
    assert summary.failed_deterministic == 1
    assert any("Handoff initiated but must_not_handoff was True" in f for f in summary.results[0].deterministic.failures)

def test_engine_replay_evaluation_success():
    # Construct a valid replay record dict
    replay_data = {
        "context": {
            "trace_id": "t1", "run_id": "r1", "span_id": "s1", "parent_span_id": None,
            "execution_id": "e1", "correlation_id": None, "user_id": None, "chat_id": None,
            "task": "do work"
        },
        "tool_results": [],
        "agent_results": [],
        "llm_responses": [
            {"model": "test-model", "content": "hello", "tool_call": None}
        ]
    }
    
    # We use a mock agent that is compatible with ReplayEngine (needs nested provider structure)
    class DeepMockAgent(BaseAgent):
        def __init__(self):
            class Router:
                def __init__(self):
                    self._provider = MockLLMProvider([])
            class Loop:
                def __init__(self):
                    self._router = Router()
                    self._registry = Mock()
            self._router = Router()
            self._loop = Loop()
            self._communicator = Mock()
            self._identity = AgentIdentity("test-agent")
            self._capabilities = AgentCapabilities()
            
        @property
        def identity(self): return self._identity
        @property
        def capabilities(self): return self._capabilities
        
        def execute(self, request):
            resp = self._loop._router._provider.generate_with_tools([], [])
            return AgentResult(request=request, state=Mock(), output=resp.content, success=True)
            
    case = GoldenCase(case_id="1", task_input="do work", constraints=GoldenConstraints(), replay_record=ReplayRecord.from_dict(replay_data))
    
    engine = EvaluationEngine(agent_class=DeepMockAgent, agent_kwargs={})
    summary = engine.run_suite([case])
    
    assert summary.passed_deterministic == 1

def test_engine_replay_mismatch_failure():
    # Exhausted LLM causes mismatch
    replay_data = {
        "context": {
            "trace_id": "t1", "run_id": "r1", "span_id": "s1", "parent_span_id": None,
            "execution_id": "e1", "correlation_id": None, "user_id": None, "chat_id": None,
            "task": "do work"
        },
        "tool_results": [],
        "agent_results": [],
        "llm_responses": [] # empty
    }
    
    class ExhaustingAgent(BaseAgent):
        def __init__(self):
            class Router:
                def __init__(self): self._provider = MockLLMProvider([])
            class Loop:
                def __init__(self):
                    self._router = Router()
                    self._registry = Mock()
            self._router = Router()
            self._loop = Loop()
            self._communicator = Mock()
            self._identity = AgentIdentity("test-agent")
            self._capabilities = AgentCapabilities()
            
        @property
        def identity(self): return self._identity
        @property
        def capabilities(self): return self._capabilities
        
        def execute(self, request):
            self._loop._router._provider.generate_with_tools([], [])
            return AgentResult(request=request, state=Mock(), output="", success=True)
            
    case = GoldenCase(case_id="1", task_input="do work", constraints=GoldenConstraints(), replay_record=ReplayRecord.from_dict(replay_data))
    engine = EvaluationEngine(agent_class=ExhaustingAgent, agent_kwargs={})
    summary = engine.run_suite([case])
    
    assert summary.failed_deterministic == 1
    assert "Replay trace diverged from GoldenCase ReplayRecord." in summary.results[0].deterministic.failures

def test_probabilistic_judge_integration():
    provider = MockLLMProvider([
        LLMResponse(
            model="test",
            tool_call=ToolCall(
                id="tc1",
                name="submit_evaluation",
                arguments={"correctness": 4.5, "relevance": 5.0, "instruction_adherence": 3.0, "reasoning": "Good"}
            )
        )
    ])
    judge = LLMJudge(provider)
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={}, judge=judge)
    
    case = GoldenCase(case_id="1", task_input="success", constraints=GoldenConstraints())
    summary = engine.run_suite([case])
    
    assert summary.passed_deterministic == 1
    assert summary.results[0].probabilistic is not None
    assert summary.results[0].probabilistic.correctness == 4.5
    assert summary.average_scores["correctness"] == 4.5

def test_deterministic_failure_overrides_judge():
    # Judge thinks it's good, but deterministic constraints fail
    provider = MockLLMProvider([
        LLMResponse(
            model="test",
            tool_call=ToolCall(
                id="tc1",
                name="submit_evaluation",
                arguments={"correctness": 5.0, "relevance": 5.0, "instruction_adherence": 5.0, "reasoning": "Perfect"}
            )
        )
    ])
    judge = LLMJudge(provider)
    engine = EvaluationEngine(agent_class=LiveTestAgent, agent_kwargs={}, judge=judge)
    
    case = GoldenCase(case_id="1", task_input="fail", constraints=GoldenConstraints(expected_success=True))
    summary = engine.run_suite([case])
    
    assert summary.failed_deterministic == 1
    # Judge score recorded but deterministic result failed
    assert summary.results[0].deterministic.passed is False
    assert summary.results[0].probabilistic.correctness == 5.0
