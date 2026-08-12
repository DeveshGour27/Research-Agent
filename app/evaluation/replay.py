import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from enum import Enum

from app.agent.contracts import AgentRequest, AgentResult, BaseAgent
from app.agent.execution_context import AgentExecutionContext
from app.agent.communicator import AgentCommunicator
from app.tools.registry import ToolRegistry
from app.tools.base import ToolResult
from app.llm.base import LLMProvider, ChatMessage, ChatResponse, LLMResponse, ToolCall
from app.agent.state import AgentState

def _reject_callables(data: Any) -> None:
    if callable(data):
        raise ValueError("Callables are not allowed in payload")
    elif isinstance(data, dict):
        for v in data.values():
            _reject_callables(v)
    elif isinstance(data, list):
        for v in data:
            _reject_callables(v)

class ReplayMismatchError(Exception):
    """Raised when replay execution diverges from the recorded trace."""
    pass

class ReplayStatus(str, Enum):
    REPLAY_SUCCESS = "REPLAY_SUCCESS"
    FAILED = "FAILED"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"

@dataclass
class ReplayResult:
    status: ReplayStatus
    details: Dict[str, Any]

@dataclass
class ReplayContext:
    trace_id: str
    run_id: str
    span_id: str
    parent_span_id: str | None
    execution_id: str
    correlation_id: str | None
    user_id: str | None
    chat_id: str | None
    task: str

@dataclass
class RecordedToolResult:
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any]
    content: str
    is_error: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedToolResult":
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        _reject_callables(data)
        return cls(
            tool_name=str(data["tool_name"]),
            tool_call_id=str(data["tool_call_id"]),
            arguments=dict(data.get("arguments", {})),
            content=str(data["content"]),
            is_error=bool(data.get("is_error", False))
        )

@dataclass
class RecordedAgentResult:
    request_input: str
    output: str | None
    success: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedAgentResult":
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        _reject_callables(data)
        return cls(
            request_input=str(data["request_input"]),
            output=str(data["output"]) if data.get("output") is not None else None,
            success=bool(data.get("success", False))
        )

@dataclass
class RecordedLLMResponse:
    model: str
    content: str | None
    tool_call: Dict[str, Any] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedLLMResponse":
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        _reject_callables(data)
        return cls(
            model=str(data["model"]),
            content=str(data["content"]) if data.get("content") is not None else None,
            tool_call=dict(data["tool_call"]) if data.get("tool_call") is not None else None,
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0))
        )

@dataclass
class ReplayRecord:
    context: ReplayContext
    tool_results: List[RecordedToolResult] = field(default_factory=list)
    agent_results: List[RecordedAgentResult] = field(default_factory=list)
    llm_responses: List[RecordedLLMResponse] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayRecord":
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        ctx_data = data.get("context", {})
        _reject_callables(ctx_data)
        if not isinstance(ctx_data, dict):
             raise ValueError("Invalid context data")
             
        ctx = ReplayContext(
            trace_id=str(ctx_data["trace_id"]),
            run_id=str(ctx_data["run_id"]),
            span_id=str(ctx_data["span_id"]),
            parent_span_id=str(ctx_data["parent_span_id"]) if ctx_data.get("parent_span_id") is not None else None,
            execution_id=str(ctx_data["execution_id"]),
            correlation_id=str(ctx_data["correlation_id"]) if ctx_data.get("correlation_id") is not None else None,
            user_id=str(ctx_data["user_id"]) if ctx_data.get("user_id") is not None else None,
            chat_id=str(ctx_data["chat_id"]) if ctx_data.get("chat_id") is not None else None,
            task=str(ctx_data["task"])
        )
        
        tools = [RecordedToolResult.from_dict(t) for t in data.get("tool_results", [])]
        agents = [RecordedAgentResult.from_dict(a) for a in data.get("agent_results", [])]
        llms = [RecordedLLMResponse.from_dict(l) for l in data.get("llm_responses", [])]
        
        return cls(
            context=ctx,
            tool_results=tools,
            agent_results=agents,
            llm_responses=llms
        )

class ReplayToolRegistry(ToolRegistry):
    def __init__(self, recorded_results: List[RecordedToolResult]):
        super().__init__()
        self._recorded_results = list(recorded_results)
        self._index = 0

    def execute(self, tool_call: ToolCall) -> ToolResult:
        if self._index >= len(self._recorded_results):
            raise ReplayMismatchError(f"Missing recorded tool result for {tool_call.name}")
        
        recorded = self._recorded_results[self._index]
        if recorded.tool_name != tool_call.name:
             raise ReplayMismatchError(f"Tool mismatch: expected {recorded.tool_name}, got {tool_call.name}")
        
        self._index += 1
        return ToolResult(
            tool_call_id=tool_call.id,
            content=recorded.content,
            is_error=recorded.is_error
        )

class ReplayCommunicator(AgentCommunicator):
    def __init__(self, recorded_results: List[RecordedAgentResult]):
        self._recorded_results = list(recorded_results)
        self._index = 0

    def send(self, request: AgentRequest) -> AgentResult:
        if self._index >= len(self._recorded_results):
            raise ReplayMismatchError(f"Missing recorded agent result for request")
            
        recorded = self._recorded_results[self._index]
        self._index += 1
        
        state = AgentState(
            messages=[],
            tool_calls=[],
            iteration=0,
            finished=True,
            final_answer=recorded.output
        )
        
        return AgentResult(
            request=request,
            state=state,
            output=recorded.output,
            success=recorded.success,
            context=request.context,
            metadata={}
        )

class ReplayLLMProvider(LLMProvider):
    def __init__(self, recorded_responses: List[RecordedLLMResponse]):
        self._recorded_responses = list(recorded_responses)
        self._index = 0

    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        if self._index >= len(self._recorded_responses):
            raise ReplayMismatchError("Missing recorded LLM response")
            
        recorded = self._recorded_responses[self._index]
        self._index += 1
        return ChatResponse(content=recorded.content or "", model=recorded.model)
        
    def generate_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        if self._index >= len(self._recorded_responses):
            raise ReplayMismatchError("Missing recorded LLM response")
            
        recorded = self._recorded_responses[self._index]
        self._index += 1
        
        tc = None
        if recorded.tool_call:
            tc = ToolCall(
                id=recorded.tool_call.get("id", ""),
                name=recorded.tool_call.get("name", ""),
                arguments=recorded.tool_call.get("arguments", {})
            )
            
        return LLMResponse(
            model=recorded.model,
            content=recorded.content,
            tool_call=tc,
            prompt_tokens=recorded.prompt_tokens,
            completion_tokens=recorded.completion_tokens
        )

class ReplayEngine:
    def __init__(self, agent: BaseAgent, record: ReplayRecord):
        self.agent = agent
        self.record = record
        
    def execute(self, request: AgentRequest) -> ReplayResult:
        ctx = AgentExecutionContext(
            task=self.record.context.task,
            user_id=self.record.context.user_id,
            chat_id=self.record.context.chat_id,
            execution_id=self.record.context.execution_id,
            correlation_id=self.record.context.correlation_id,
            trace_id=self.record.context.trace_id,
            run_id=self.record.context.run_id,
            span_id=self.record.context.span_id,
            parent_span_id=self.record.context.parent_span_id
        )
        
        new_request = AgentRequest(
            input_text=request.input_text,
            metadata=request.metadata,
            request_id=request.request_id,
            context=ctx,
            correlation_id=request.correlation_id,
            sender_id=request.sender_id,
            is_required=request.is_required
        )
        
        mock_registry = ReplayToolRegistry(self.record.tool_results)
        mock_provider = ReplayLLMProvider(self.record.llm_responses)
        mock_communicator = ReplayCommunicator(self.record.agent_results)
        
        original_state = {}
        
        # Phase 6 Supervisor path
        if self.agent.__class__.__name__ == "Supervisor":
            if hasattr(self.agent, "_planner") and hasattr(self.agent._planner, "_provider"):
                original_state["supervisor_planner_provider"] = self.agent._planner._provider
                self.agent._planner._provider = mock_provider
                
            if hasattr(self.agent, "_plan_executor") and hasattr(self.agent._plan_executor, "_communicator"):
                original_state["supervisor_executor_communicator"] = self.agent._plan_executor._communicator
                self.agent._plan_executor._communicator = mock_communicator
        else:
            # Phase 5 Legacy path
            if hasattr(self.agent, "_registry"):
                original_state["_registry"] = self.agent._registry
                self.agent._registry = mock_registry
                
            if hasattr(self.agent, "_loop"):
                original_state["_loop_registry"] = getattr(self.agent._loop, "_registry", None)
                if hasattr(self.agent._loop, "_registry"):
                    self.agent._loop._registry = mock_registry
                    
            if hasattr(self.agent, "_router") and hasattr(self.agent._router, "_provider"):
                original_state["_provider"] = self.agent._router._provider
                self.agent._router._provider = mock_provider
                
            if hasattr(self.agent, "_loop") and hasattr(self.agent._loop, "_router") and hasattr(self.agent._loop._router, "_provider"):
                original_state["_loop_provider"] = self.agent._loop._router._provider
                self.agent._loop._router._provider = mock_provider
                
            if hasattr(self.agent, "_communicator"):
                original_state["_communicator"] = self.agent._communicator
                self.agent._communicator = mock_communicator
            
        try:
            result = self.agent.execute(new_request)
            return ReplayResult(
                status=ReplayStatus.REPLAY_SUCCESS, 
                details={
                    "output": result.output, 
                    "success": result.success,
                    "metadata": result.metadata,
                }
            )
        except ReplayMismatchError as e:
            return ReplayResult(
                status=ReplayStatus.REPLAY_MISMATCH, 
                details={"error": str(e)}
            )
        except Exception as e:
            from app.agent.contracts import AgentExecutionError
            metadata = {}
            if isinstance(e, AgentExecutionError) and getattr(e, "details", None):
                metadata = e.details
                
            return ReplayResult(
                status=ReplayStatus.FAILED, 
                details={
                    "error": str(e), 
                    "traceback": traceback.format_exc(),
                    "metadata": metadata
                }
            )
        finally:
            # Phase 6 restore
            if "supervisor_planner_provider" in original_state:
                self.agent._planner._provider = original_state["supervisor_planner_provider"]
            if "supervisor_executor_communicator" in original_state:
                self.agent._plan_executor._communicator = original_state["supervisor_executor_communicator"]
                
            # Phase 5 restore
            if "_registry" in original_state:
                self.agent._registry = original_state["_registry"]
            if "_loop_registry" in original_state:
                self.agent._loop._registry = original_state["_loop_registry"]
            if "_provider" in original_state:
                self.agent._router._provider = original_state["_provider"]
            if "_loop_provider" in original_state:
                self.agent._loop._router._provider = original_state["_loop_provider"]
            if "_communicator" in original_state:
                self.agent._communicator = original_state["_communicator"]
