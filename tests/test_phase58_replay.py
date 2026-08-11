"""Tests for Phase 5.8 Step 5 — Deterministic Replay Engine."""

import pytest

from app.evaluation.replay import (
    ReplayContext,
    ReplayRecord,
    RecordedToolResult,
    RecordedAgentResult,
    RecordedLLMResponse,
    ReplayToolRegistry,
    ReplayCommunicator,
    ReplayLLMProvider,
    ReplayEngine,
    ReplayMismatchError,
    ReplayStatus,
    ReplayResult
)
from app.agent.contracts import AgentRequest, AgentResult, AgentIdentity, AgentCapabilities, BaseAgent
from app.agent.execution_context import AgentExecutionContext
from app.agent.state import AgentState
from app.llm.base import ToolCall, ChatMessage

class MockAgent(BaseAgent):
    """Simple agent used for replay testing."""
    def __init__(self, _provider, _registry, _communicator):
        self._provider = _provider
        self._registry = _registry
        self._communicator = _communicator
        
        # In a real agent, provider is inside router and router is inside loop.
        # ReplayEngine patches agent._registry, agent._communicator, and agent._loop._router._provider.
        # We simulate the exact structure ReplayEngine expects.
        class Router:
            def __init__(self, provider):
                self._provider = provider
                
        class Loop:
            def __init__(self, router, registry):
                self._router = router
                self._registry = registry
                
        self._router = Router(_provider)
        self._loop = Loop(self._router, _registry)
        
        self._identity = AgentIdentity("test-agent")
        self._capabilities = AgentCapabilities()
        
    @property
    def identity(self):
        return self._identity
        
    @property
    def capabilities(self):
        return self._capabilities
        
    def execute(self, request: AgentRequest) -> AgentResult:
        # Generate some text from the deeply nested provider
        response = self._loop._router._provider.generate_with_tools([{"role": "user", "content": request.input_text}], [])
        
        # Make a tool call if the LLM returned one
        if response.tool_call:
            tool_res = self._loop._registry.execute(response.tool_call)
            output = tool_res.content
        else:
            output = response.content
            
        # Maybe delegate based on output
        if output == "delegate":
            res = self._communicator.send(AgentRequest(input_text="subtask"))
            output = res.output
            
        return AgentResult(
            request=request,
            state=AgentState(messages=[], tool_calls=[], iteration=1, finished=True, final_answer=output),
            output=output,
            success=True
        )

# ----------------------------------------------------------------------------
# 1. ReplayContext Tests
# ----------------------------------------------------------------------------
def test_replay_context_isolation():
    # Verify ReplayContext does NOT subclass AgentExecutionContext
    assert not issubclass(ReplayContext, AgentExecutionContext)

def test_replay_context_fields():
    ctx = ReplayContext(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id="s0",
        execution_id="e1", correlation_id="c1", user_id="u1", chat_id="ch1",
        task="test"
    )
    assert ctx.trace_id == "t1"
    assert ctx.span_id == "s1"
    assert ctx.task == "test"

# ----------------------------------------------------------------------------
# 2. ReplayRecord Validation Tests
# ----------------------------------------------------------------------------
def test_replay_record_valid_deserialization():
    data = {
        "context": {
            "trace_id": "t1", "run_id": "r1", "span_id": "s1", "parent_span_id": None,
            "execution_id": "e1", "correlation_id": None, "user_id": None, "chat_id": None,
            "task": "do work"
        },
        "tool_results": [
            {"tool_name": "calc", "tool_call_id": "tc1", "arguments": {"a": 1}, "content": "2"}
        ],
        "agent_results": [],
        "llm_responses": [
            {"model": "test-model", "content": "hello", "tool_call": None}
        ]
    }
    
    record = ReplayRecord.from_dict(data)
    assert record.context.trace_id == "t1"
    assert len(record.tool_results) == 1
    assert record.tool_results[0].content == "2"
    assert len(record.llm_responses) == 1
    assert record.llm_responses[0].content == "hello"

def test_replay_record_rejects_missing_fields():
    data = {
        "context": {
            # missing trace_id
            "run_id": "r1", "span_id": "s1", "parent_span_id": None,
            "execution_id": "e1", "task": "do work"
        }
    }
    with pytest.raises(Exception):
        ReplayRecord.from_dict(data)

def test_replay_record_rejects_callables():
    def malicious(): pass
    
    data = {
        "context": malicious,
    }
    with pytest.raises(ValueError):
        ReplayRecord.from_dict(data)
        
    data2 = {
        "context": {
            "trace_id": "t1", "run_id": "r1", "span_id": "s1", "parent_span_id": None,
            "execution_id": "e1", "correlation_id": None, "user_id": None, "chat_id": None,
            "task": "do work"
        },
        "tool_results": [
            {"tool_name": "calc", "tool_call_id": "tc1", "arguments": {"a": malicious}, "content": "2"}
        ]
    }
    with pytest.raises(ValueError, match="Callables are not allowed"):
        ReplayRecord.from_dict(data2)

# ----------------------------------------------------------------------------
# 3. Deterministic Providers Tests
# ----------------------------------------------------------------------------
def test_replay_tool_registry_success():
    rec = RecordedToolResult(
        tool_name="test_tool", tool_call_id="id1",
        arguments={}, content="result", is_error=False
    )
    registry = ReplayToolRegistry([rec])
    
    result = registry.execute(ToolCall(id="id1", name="test_tool", arguments={}))
    assert result.content == "result"
    assert result.is_error is False

def test_replay_tool_registry_mismatch():
    rec = RecordedToolResult(
        tool_name="test_tool", tool_call_id="id1",
        arguments={}, content="result", is_error=False
    )
    registry = ReplayToolRegistry([rec])
    
    with pytest.raises(ReplayMismatchError):
        # We only check name in the current execution implementation
        registry.execute(ToolCall(id="id2", name="wrong_tool", arguments={}))

def test_replay_tool_registry_exhausted():
    registry = ReplayToolRegistry([])
    with pytest.raises(ReplayMismatchError):
        registry.execute(ToolCall(id="id1", name="t", arguments={}))

def test_replay_llm_provider_success():
    rec = RecordedLLMResponse(model="test", content="hi", tool_call=None)
    provider = ReplayLLMProvider([rec])
    
    res = provider.generate_with_tools([], [])
    assert res.content == "hi"

def test_replay_llm_provider_exhausted():
    provider = ReplayLLMProvider([])
    with pytest.raises(ReplayMismatchError):
        provider.generate_with_tools([], [])

def test_replay_communicator_success():
    rec = RecordedAgentResult(request_input="sub", output="done", success=True)
    comm = ReplayCommunicator([rec])
    
    res = comm.send(AgentRequest(input_text="sub"))
    assert res.output == "done"
    assert res.success is True

# ----------------------------------------------------------------------------
# 4. ReplayEngine Core Tests
# ----------------------------------------------------------------------------
@pytest.fixture
def base_context():
    return ReplayContext(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        execution_id="e1", correlation_id=None, user_id=None, chat_id=None,
        task="test_task"
    )

def test_replay_engine_success(base_context):
    record = ReplayRecord(
        context=base_context,
        llm_responses=[RecordedLLMResponse(model="test", content="final answer", tool_call=None)]
    )
    
    agent = MockAgent(None, None, None)
    engine = ReplayEngine(agent, record)
    result = engine.execute(AgentRequest(input_text="do it"))
    
    assert result.status == ReplayStatus.REPLAY_SUCCESS
    assert result.details["output"] == "final answer"

def test_replay_engine_with_tool(base_context):
    record = ReplayRecord(
        context=base_context,
        llm_responses=[
            RecordedLLMResponse(model="test", content=None, tool_call={"id": "tc1", "name": "calc", "arguments": {}})
        ],
        tool_results=[
            RecordedToolResult(tool_name="calc", tool_call_id="tc1", arguments={}, content="42")
        ]
    )
    
    agent = MockAgent(None, None, None)
    engine = ReplayEngine(agent, record)
    result = engine.execute(AgentRequest(input_text="do it"))
    
    assert result.status == ReplayStatus.REPLAY_SUCCESS
    assert result.details["output"] == "42"

def test_replay_engine_mismatch_tool_exhausted(base_context):
    record = ReplayRecord(
        context=base_context,
        llm_responses=[
            RecordedLLMResponse(model="test", content=None, tool_call={"id": "tc1", "name": "calc", "arguments": {}})
        ],
        tool_results=[] # Missing tool result
    )
    
    agent = MockAgent(None, None, None)
    engine = ReplayEngine(agent, record)
    result = engine.execute(AgentRequest(input_text="do it"))
    
    assert result.status == ReplayStatus.REPLAY_MISMATCH
    assert "Missing recorded tool result" in result.details["error"]

def test_replay_engine_mismatch_llm_exhausted(base_context):
    record = ReplayRecord(
        context=base_context,
        llm_responses=[] # Missing LLM response
    )
    
    agent = MockAgent(None, None, None)
    engine = ReplayEngine(agent, record)
    result = engine.execute(AgentRequest(input_text="do it"))
    
    assert result.status == ReplayStatus.REPLAY_MISMATCH

def test_replay_engine_agent_delegation(base_context):
    record = ReplayRecord(
        context=base_context,
        llm_responses=[
            RecordedLLMResponse(model="test", content="delegate", tool_call=None)
        ],
        agent_results=[
            RecordedAgentResult(request_input="subtask", output="sub_done", success=True)
        ]
    )
    
    agent = MockAgent(None, None, None)
    engine = ReplayEngine(agent, record)
    result = engine.execute(AgentRequest(input_text="do it"))
    
    assert result.status == ReplayStatus.REPLAY_SUCCESS
    assert result.details["output"] == "sub_done"

# ----------------------------------------------------------------------------
# 5. Determinism & State Isolation
# ----------------------------------------------------------------------------
def test_replay_engine_is_deterministic(base_context):
    record = ReplayRecord(
        context=base_context,
        llm_responses=[RecordedLLMResponse(model="test", content="stable", tool_call=None)]
    )
    
    agent1 = MockAgent(None, None, None)
    engine1 = ReplayEngine(agent1, record)
    r1 = engine1.execute(AgentRequest(input_text="do it"))
    
    agent2 = MockAgent(None, None, None)
    engine2 = ReplayEngine(agent2, record)
    r2 = engine2.execute(AgentRequest(input_text="do it"))
    
    assert r1 == r2

def test_replay_fails_cleanly_on_internal_agent_error(base_context):
    class ErrorAgent(BaseAgent):
        @property
        def identity(self): return AgentIdentity("e")
        @property
        def capabilities(self): return AgentCapabilities()
        def execute(self, req): raise RuntimeError("Boom")
        
    record = ReplayRecord(context=base_context)
    agent = ErrorAgent()
    engine = ReplayEngine(agent, record)
    
    result = engine.execute(AgentRequest(input_text="do it"))
    assert result.status == ReplayStatus.FAILED
    assert "Boom" in result.details["error"]
