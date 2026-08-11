"""Tests for Phase 5.8 Step 3 — Execution Instrumentation."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.agent.collaboration import (
    CollaborationPolicy,
    CollaborationSession,
)
from app.agent.communicator import InProcessCommunicator
from app.agent.contracts import (
    AgentCapabilities,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
    CollaborationRequest,
)
from app.agent.execution_context import AgentExecutionContext, ExecutionStatus
from app.agent.execution_policy import ExecutionPolicy
from app.agent.executor import PlanExecutor
from app.agent.plan import Plan, PlanStep, PlanStatus, PlanStepStatus
from app.agent.registry import AgentRegistry
from app.agent.retry import RetryPolicy
from app.agent.routing import CapabilityRouter
from app.agent.state import AgentState
from app.agent.supervisor import Supervisor
from app.exceptions import (
    AgentCancellationError,
    AgentTimeoutError,
    CommunicationError,
)
from app.observability.tracer import extract_span_info


# ------------------------------------------------------------------ #
# Test helpers / fakes
# ------------------------------------------------------------------ #

class FakeAgent(BaseAgent):
    """Minimal agent for instrumentation tests."""

    def __init__(
        self,
        name: str,
        task_types: frozenset[str] | None = None,
        output: str = "done",
        success: bool = True,
        should_raise: Exception | None = None,
        handoff_to: str | None = None,
        handoff_message: str = "help",
    ) -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities(
            task_types=task_types or frozenset(),
        )
        self.output = output
        self.success = success
        self.should_raise = should_raise
        self.handoff_to = handoff_to
        self.handoff_message = handoff_message
        self.call_count = 0

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        self.call_count += 1
        if self.should_raise:
            raise self.should_raise

        collab_req = None
        if self.handoff_to and self.call_count == 1:
            collab_req = CollaborationRequest(
                requested_task_type=self.handoff_to,
                message=self.handoff_message,
            )

        return AgentResult(
            request=request,
            state=AgentState(),
            output=self.output,
            success=self.success,
            collaboration_request=collab_req,
        )


class FakeCommunicator:
    """Minimal communicator that tracks attempts and can fail."""

    def __init__(self, fails: int = 0, error_type: type = CommunicationError):
        self.fails = fails
        self.error_type = error_type
        self.attempts = 0

    def send(self, request: AgentRequest) -> AgentResult:
        self.attempts += 1
        if self.attempts <= self.fails:
            raise self.error_type("Failed")
        from app.agent.state import AgentState
        return AgentResult(
            request=request, state=AgentState(),
            output="Success", success=True,
        )


def _collect_events(caplog_records: list) -> list[dict]:
    """Extract observability events from captured log records."""
    events = []
    for record in caplog_records:
        ev = getattr(record, "event", None)
        if ev and isinstance(ev, dict) and "event_type" in ev:
            events.append(ev)
    return events


def _event_types(events: list[dict]) -> list[str]:
    return [e["event_type"] for e in events]


def _setup_collab_session(
    policy: CollaborationPolicy | None = None,
) -> tuple[CollaborationSession, AgentRegistry]:
    """Create a standard collaboration session for tests."""
    registry = AgentRegistry()
    researcher = FakeAgent("researcher", frozenset(["research"]), handoff_to="analysis")
    analyst = FakeAgent("analyst", frozenset(["analysis"]))
    registry.register(researcher)
    registry.register(analyst)

    router = CapabilityRouter()
    communicator = InProcessCommunicator(registry)
    context = AgentExecutionContext(task="test")

    request = AgentRequest(
        input_text="Do research",
        metadata={"required_task_type": "research"},
        context=context,
    )

    session = CollaborationSession(
        step_id="s1",
        initial_request=request,
        router=router,
        registry=registry,
        communicator=communicator,
        retry_policy=RetryPolicy(),
        policy=policy,
    )

    return session, registry


# ------------------------------------------------------------------ #
# 1. Executor emits execution-start event
# ------------------------------------------------------------------ #

def test_executor_emits_execution_started(caplog) -> None:
    caplog.set_level(logging.INFO)
    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    ctx = AgentExecutionContext(task="test")
    plan = Plan(goal="test goal")
    plan.steps["1"] = PlanStep(step_id="1", description="do work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    types = _event_types(events)
    assert "AgentExecutionStarted" in types


# ------------------------------------------------------------------ #
# 2. Executor emits plan-generation event (via supervisor planning path)
# ------------------------------------------------------------------ #

def test_supervisor_emits_plan_generated(caplog) -> None:
    caplog.set_level(logging.INFO)

    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    communicator = InProcessCommunicator(registry)

    # Build a plan that the planner will return
    plan = Plan(goal="test")
    plan.steps["1"] = PlanStep(step_id="1", description="work", task_type="work")

    planner = MagicMock()
    planner.generate_plan.return_value = plan

    executor = PlanExecutor(CapabilityRouter(), registry, communicator)

    supervisor = Supervisor(
        agents=[agent],
        registry=registry,
        communicator=communicator,
        planner=planner,
        plan_executor=executor,
    )

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)
    result = supervisor.execute(request)

    assert result.success

    events = _collect_events(caplog.records)
    types = _event_types(events)
    assert "PlanGenerated" in types


# ------------------------------------------------------------------ #
# 3. Executor emits step lifecycle events
# ------------------------------------------------------------------ #

def test_executor_emits_step_lifecycle_events(caplog) -> None:
    caplog.set_level(logging.INFO)
    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    ctx = AgentExecutionContext(task="test")
    plan = Plan(goal="test")
    plan.steps["1"] = PlanStep(step_id="1", description="do work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    types = _event_types(events)
    assert "PlanStepCompleted" in types

    # Verify step event contains the right step_id
    step_events = [e for e in events if e["event_type"] == "PlanStepCompleted"]
    assert len(step_events) == 1
    assert step_events[0]["event_data"]["step_id"] == "1"
    assert step_events[0]["event_data"]["status"] == "completed"


# ------------------------------------------------------------------ #
# 4. Supervisor emits routing/delegation events
# ------------------------------------------------------------------ #

def test_supervisor_emits_execution_events(caplog) -> None:
    caplog.set_level(logging.INFO)

    agent = FakeAgent("worker", frozenset(["work"]))
    supervisor = Supervisor(agents=[agent])

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)
    result = supervisor.execute(request)

    assert result.success

    events = _collect_events(caplog.records)
    types = _event_types(events)

    # Should have started and completed events
    assert "AgentExecutionStarted" in types
    assert "AgentExecutionCompleted" in types


# ------------------------------------------------------------------ #
# 5. Retry produces a retry event
# ------------------------------------------------------------------ #

def test_supervisor_retry_emits_event(caplog) -> None:
    caplog.set_level(logging.INFO)

    failing_agent = FakeAgent("fail-agent", should_raise=Exception("boom"))
    success_agent = FakeAgent("ok-agent")

    supervisor = Supervisor(agents=[failing_agent, success_agent], max_attempts=2)
    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)
    result = supervisor.execute(request)

    assert result.success

    events = _collect_events(caplog.records)
    types = _event_types(events)
    assert "RetryAttempted" in types


# ------------------------------------------------------------------ #
# 6. Collaboration emits handoff events
# ------------------------------------------------------------------ #

def test_collaboration_emits_handoff_events(caplog) -> None:
    caplog.set_level(logging.INFO)

    session, _ = _setup_collab_session()
    result = session.execute()

    assert result.success

    events = _collect_events(caplog.records)
    types = _event_types(events)

    assert "HandoffInitiated" in types
    assert "HandoffResolved" in types


# ------------------------------------------------------------------ #
# 7. Failure produces appropriate failure event
# ------------------------------------------------------------------ #

def test_supervisor_failure_emits_completed_event(caplog) -> None:
    caplog.set_level(logging.INFO)

    failing_agent = FakeAgent("fail-agent", success=False, output=None)
    supervisor = Supervisor(agents=[failing_agent])

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)

    with pytest.raises(Exception):
        supervisor.execute(request)

    events = _collect_events(caplog.records)
    types = _event_types(events)

    assert "AgentExecutionStarted" in types
    assert "AgentExecutionCompleted" in types

    completed_events = [e for e in events if e["event_type"] == "AgentExecutionCompleted"]
    assert any(e["event_data"]["success"] is False for e in completed_events)


# ------------------------------------------------------------------ #
# 8. Cancellation produces cancellation telemetry
# ------------------------------------------------------------------ #

def test_executor_cancellation_emits_event(caplog) -> None:
    caplog.set_level(logging.INFO)

    ctx = AgentExecutionContext(task="test")
    ctx.mark_cancelled()

    plan = Plan()
    plan.steps["1"] = PlanStep(step_id="1", description="s1")

    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), FakeCommunicator())
    with pytest.raises(AgentCancellationError):
        executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    types = _event_types(events)
    assert "TimeoutCancellation" in types

    cancel_events = [e for e in events if e["event_type"] == "TimeoutCancellation"]
    assert any(e["event_data"]["reason"] == "cancelled" for e in cancel_events)


# ------------------------------------------------------------------ #
# 9. Timeout produces timeout telemetry
# ------------------------------------------------------------------ #

def test_executor_timeout_emits_event(caplog) -> None:
    caplog.set_level(logging.INFO)

    ctx = AgentExecutionContext(task="test")
    ctx.mark_timed_out()

    plan = Plan()
    plan.steps["1"] = PlanStep(step_id="1", description="s1")

    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), FakeCommunicator())
    with pytest.raises(AgentTimeoutError):
        executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    types = _event_types(events)
    assert "TimeoutCancellation" in types

    timeout_events = [e for e in events if e["event_type"] == "TimeoutCancellation"]
    assert any(e["event_data"]["reason"] == "timed_out" for e in timeout_events)


# ------------------------------------------------------------------ #
# 10. Partial success produces partial-success telemetry
# ------------------------------------------------------------------ #

def test_executor_partial_success_emits_event(caplog) -> None:
    caplog.set_level(logging.INFO)

    ctx = AgentExecutionContext(task="test")
    plan = Plan()
    # Required step completed, optional step failed
    plan.steps["1"] = PlanStep(step_id="1", description="s1", is_required=True, status=PlanStepStatus.COMPLETED)
    plan.steps["2"] = PlanStep(step_id="2", description="s2", is_required=False, status=PlanStepStatus.FAILED)

    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), FakeCommunicator())
    result_plan = executor.execute(plan, ctx, ExecutionPolicy())
    assert result_plan.status == PlanStatus.PARTIAL_SUCCESS

    events = _collect_events(caplog.records)
    completed_events = [e for e in events if e["event_type"] == "AgentExecutionCompleted"]

    assert len(completed_events) >= 1
    assert any(e["event_data"].get("output") == "partial_success" for e in completed_events)


# ------------------------------------------------------------------ #
# 11. Trace IDs propagate correctly
# ------------------------------------------------------------------ #

def test_trace_ids_propagate_in_events(caplog) -> None:
    caplog.set_level(logging.INFO)
    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    ctx = AgentExecutionContext(task="test")
    plan = Plan(goal="test")
    plan.steps["1"] = PlanStep(step_id="1", description="do work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    for event in events:
        assert event["trace_id"] == ctx.trace_id
        assert event["run_id"] == ctx.run_id
        assert event["span_id"] == ctx.span_id


# ------------------------------------------------------------------ #
# 12. Parent/child span relationships are correct
# ------------------------------------------------------------------ #

def test_span_parent_child_in_events(caplog) -> None:
    caplog.set_level(logging.INFO)

    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    parent_ctx = AgentExecutionContext(task="parent")
    child_ctx = parent_ctx.create_child_span(task="child")

    plan = Plan(goal="child work")
    plan.steps["1"] = PlanStep(step_id="1", description="do work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    executor.execute(plan, child_ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    assert len(events) > 0

    for event in events:
        assert event["trace_id"] == parent_ctx.trace_id
        assert event["span_id"] == child_ctx.span_id
        assert event["parent_span_id"] == parent_ctx.span_id


# ------------------------------------------------------------------ #
# 13. Events appear in valid lifecycle order
# ------------------------------------------------------------------ #

def test_events_in_valid_lifecycle_order(caplog) -> None:
    caplog.set_level(logging.INFO)

    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    ctx = AgentExecutionContext(task="test")
    plan = Plan(goal="test")
    plan.steps["1"] = PlanStep(step_id="1", description="work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    types = _event_types(events)

    # AgentExecutionStarted must come before AgentExecutionCompleted
    started_idx = types.index("AgentExecutionStarted")
    completed_idx = types.index("AgentExecutionCompleted")
    assert started_idx < completed_idx

    # PlanStepCompleted should be between started and completed
    step_idx = types.index("PlanStepCompleted")
    assert started_idx < step_idx < completed_idx


# ------------------------------------------------------------------ #
# 14. Sensitive values are redacted
# ------------------------------------------------------------------ #

def test_sensitive_values_are_redacted(caplog) -> None:
    caplog.set_level(logging.INFO)

    agent = FakeAgent("worker", output="done with sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6")
    supervisor = Supervisor(agents=[agent])

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)
    result = supervisor.execute(request)

    assert result.success

    events = _collect_events(caplog.records)
    # Check that no event leaks the raw API key
    for event in events:
        serialized = str(event)
        assert "sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" not in serialized


# ------------------------------------------------------------------ #
# 15. Telemetry failure does not fail execution
# ------------------------------------------------------------------ #

def test_telemetry_failure_does_not_fail_execution(monkeypatch) -> None:
    """If event emission raises, the agent still succeeds."""
    import app.observability.events as events_module

    original_emit = events_module.BaseEvent.emit

    def broken_emit(self):
        raise RuntimeError("Telemetry is broken!")

    monkeypatch.setattr(events_module.BaseEvent, "emit", broken_emit)

    agent = FakeAgent("worker")
    supervisor = Supervisor(agents=[agent])

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)
    result = supervisor.execute(request)

    assert result.success is True
    assert result.output == "done"


# ------------------------------------------------------------------ #
# 16. Telemetry failure does not change execution status
# ------------------------------------------------------------------ #

def test_telemetry_failure_preserves_execution_status(monkeypatch) -> None:
    import app.observability.events as events_module

    def broken_emit(self):
        raise RuntimeError("Telemetry crash!")

    monkeypatch.setattr(events_module.BaseEvent, "emit", broken_emit)

    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    ctx = AgentExecutionContext(task="test")
    plan = Plan(goal="test")
    plan.steps["1"] = PlanStep(step_id="1", description="work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    result_plan = executor.execute(plan, ctx, ExecutionPolicy())

    assert result_plan.status == PlanStatus.COMPLETED


# ------------------------------------------------------------------ #
# 17. Existing Phase 5.6 collaboration behavior remains unchanged
# ------------------------------------------------------------------ #

def test_collaboration_behavior_unchanged() -> None:
    session, registry = _setup_collab_session()
    result = session.execute()

    assert result.success is True

    researcher = registry.get("researcher")
    analyst = registry.get("analyst")
    assert researcher.call_count == 1
    assert analyst.call_count == 1
    assert "researcher" in session.participating_agents
    assert "analyst" in session.participating_agents


# ------------------------------------------------------------------ #
# 18. Existing Phase 5.7 recovery behavior remains unchanged
# ------------------------------------------------------------------ #

def test_recovery_behavior_unchanged() -> None:
    """First agent fails, second succeeds — supervisor recovers."""
    failing = FakeAgent("fail", should_raise=Exception("boom"))
    success = FakeAgent("ok")

    supervisor = Supervisor(agents=[failing, success], max_attempts=2)
    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)

    result = supervisor.execute(request)
    assert result.success is True
    assert result.output == "done"


# ------------------------------------------------------------------ #
# 19. No duplicate lifecycle events are emitted
# ------------------------------------------------------------------ #

def test_no_duplicate_lifecycle_events(caplog) -> None:
    caplog.set_level(logging.INFO)

    registry = AgentRegistry()
    agent = FakeAgent("worker", frozenset(["work"]))
    registry.register(agent)

    ctx = AgentExecutionContext(task="test")
    plan = Plan(goal="test")
    plan.steps["1"] = PlanStep(step_id="1", description="work", task_type="work")

    executor = PlanExecutor(CapabilityRouter(), registry, InProcessCommunicator(registry))
    executor.execute(plan, ctx, ExecutionPolicy())

    events = _collect_events(caplog.records)
    types = _event_types(events)

    # Exactly one started, one completed for the executor
    assert types.count("AgentExecutionStarted") == 1
    assert types.count("AgentExecutionCompleted") == 1
    assert types.count("PlanStepCompleted") == 1


# ------------------------------------------------------------------ #
# 20. Normal execution still produces the expected final result
# ------------------------------------------------------------------ #

def test_normal_execution_produces_expected_result() -> None:
    agent = FakeAgent("worker", output="research complete")
    supervisor = Supervisor(agents=[agent])

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="research AI agents", context=context)
    result = supervisor.execute(request)

    assert result.success is True
    assert result.output == "research complete"
    assert context.status == ExecutionStatus.COMPLETED
