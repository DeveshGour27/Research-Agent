"""Structured event schemas for Phase 5.8 Observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.logger import get_logger
from app.observability.redactor import Redactor

_logger = get_logger("observability")
_redactor = Redactor()


@dataclass
class BaseEvent:
    """Base schema for all deterministic observability events."""

    event_type: str
    trace_id: str | None
    run_id: str | None
    span_id: str | None
    parent_span_id: str | None
    event_data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def emit(self) -> None:
        """
        Safely redacts event data and emits it as a structured log.
        Failure isolation: Never raises an exception on the critical path.
        """
        try:
            safe_data = _redactor.redact(self.event_data)
            
            payload = {
                "event_type": self.event_type,
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "timestamp": self.timestamp,
                "event_data": safe_data,
            }
            
            # Write to the logger as extra JSON kwargs
            _logger.info(self.event_type, extra={"event": payload})
        except Exception:
            # Observability must NEVER become a control-flow dependency.
            pass


@dataclass
class AgentExecutionStartedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, agent_name: str, input_text: str):
        super().__init__(
            event_type="AgentExecutionStarted",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"agent_name": agent_name, "input_text": input_text},
        )


@dataclass
class AgentExecutionCompletedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, agent_name: str, success: bool, output: str | None):
        super().__init__(
            event_type="AgentExecutionCompleted",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"agent_name": agent_name, "success": success, "output": output},
        )


@dataclass
class PlanGeneratedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, plan_id: str, step_count: int, is_replan: bool):
        super().__init__(
            event_type="PlanGenerated",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"plan_id": plan_id, "step_count": step_count, "is_replan": is_replan},
        )


@dataclass
class PlanStepCompletedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, plan_id: str, step_id: str, status: str):
        super().__init__(
            event_type="PlanStepCompleted",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"plan_id": plan_id, "step_id": step_id, "status": status},
        )


@dataclass
class HandoffInitiatedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, target_task_type: str, message: str):
        super().__init__(
            event_type="HandoffInitiated",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"target_task_type": target_task_type, "message": message},
        )


@dataclass
class HandoffResolvedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, target_task_type: str, success: bool, messages_exchanged: int):
        super().__init__(
            event_type="HandoffResolved",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"target_task_type": target_task_type, "success": success, "messages_exchanged": messages_exchanged},
        )


@dataclass
class RetryAttemptedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, error_type: str, attempt_number: int):
        super().__init__(
            event_type="RetryAttempted",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"error_type": error_type, "attempt_number": attempt_number},
        )


@dataclass
class TimeoutCancellationEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, reason: str):
        super().__init__(
            event_type="TimeoutCancellation",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"reason": reason},
        )


@dataclass
class ToolCalledEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, tool_name: str, arguments: dict[str, Any]):
        super().__init__(
            event_type="ToolCalled",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"tool_name": tool_name, "arguments": arguments},
        )


@dataclass
class ToolCompletedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, tool_name: str, success: bool, output: Any):
        super().__init__(
            event_type="ToolCompleted",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"tool_name": tool_name, "success": success, "output": output},
        )


@dataclass
class MemoryRetrievedEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, query: str, top_k: int, result_count: int):
        super().__init__(
            event_type="MemoryRetrieved",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"query": query, "top_k": top_k, "result_count": result_count},
        )
