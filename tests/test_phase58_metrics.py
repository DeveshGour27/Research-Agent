"""Tests for Phase 5.8 Step 4 — Metrics."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import time
from unittest.mock import MagicMock, patch

from app.observability.events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    PlanGeneratedEvent,
    PlanStepCompletedEvent,
    HandoffInitiatedEvent,
    HandoffResolvedEvent,
    RetryAttemptedEvent,
    TimeoutCancellationEvent,
)
from app.observability.metrics import MetricsRegistry


def test_basic_counters() -> None:
    registry = MetricsRegistry()

    # 1. Execution Starts
    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", input_text="test"
    ))
    
    # 2. Handoff Initiated
    registry.record_event(HandoffInitiatedEvent(
        trace_id="t1", run_id="r1", span_id="s2", parent_span_id="s1",
        target_task_type="research", message="do research"
    ))
    
    # 3. Handoff Resolved
    registry.record_event(HandoffResolvedEvent(
        trace_id="t1", run_id="r1", span_id="s2", parent_span_id="s1",
        target_task_type="research", success=True, messages_exchanged=2
    ))

    # 4. Retry
    registry.record_event(RetryAttemptedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        error_type="SomeError", attempt_number=1
    ))

    # 5. Execution Completed
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", success=True, output="done"
    ))

    snapshot = registry.snapshot()
    assert snapshot["executions"]["total"] == 1
    assert snapshot["executions"]["successful"] == 1
    assert snapshot["executions"]["failed"] == 0
    
    assert snapshot["handoffs"]["total"] == 1
    assert snapshot["handoffs"]["successful"] == 1
    assert snapshot["handoffs"]["failed"] == 0
    
    assert snapshot["retries"]["total"] == 1


def test_correctness_duplicate_lifecycle_events() -> None:
    registry = MetricsRegistry()

    started = AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", input_text="test"
    )
    completed = AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", success=True, output="done"
    )

    registry.record_event(started)
    registry.record_event(started) # duplicate
    registry.record_event(completed)
    registry.record_event(completed) # duplicate

    snapshot = registry.snapshot()
    assert snapshot["executions"]["total"] == 1
    assert snapshot["executions"]["successful"] == 1


def test_correctness_cancellation_cannot_become_success() -> None:
    registry = MetricsRegistry()

    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", input_text="test"
    ))
    
    registry.record_event(TimeoutCancellationEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        reason="cancelled"
    ))
    
    # Buggy code sequence emits completion after cancellation
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", success=True, output="done"
    ))

    snapshot = registry.snapshot()
    assert snapshot["executions"]["cancelled"] == 1
    assert snapshot["executions"]["successful"] == 0


def test_partial_success_distinct_from_success() -> None:
    registry = MetricsRegistry()

    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", input_text="test"
    ))
    
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="Supervisor", success=True, output="partial_success"
    ))

    snapshot = registry.snapshot()
    assert snapshot["executions"]["successful"] == 0
    assert snapshot["executions"]["partial_success"] == 1


def test_agent_metrics_tracking() -> None:
    registry = MetricsRegistry()

    # Agent A executes and succeeds
    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", input_text="test"
    ))
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", success=True, output="done"
    ))

    # Agent B executes (say, as child) and fails
    # It might only have a completion event if legacy supervisor is used
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t2", run_id="r2", span_id="s2", parent_span_id="s1",
        agent_name="AgentB", success=False, output="error"
    ))

    snapshot = registry.snapshot()
    assert "AgentA" in snapshot["agents"]
    assert snapshot["agents"]["AgentA"]["executions"] == 1
    assert snapshot["agents"]["AgentA"]["failures"] == 0

    assert "AgentB" in snapshot["agents"]
    assert snapshot["agents"]["AgentB"]["executions"] == 0 # we didn't see started event
    assert snapshot["agents"]["AgentB"]["failures"] == 1


def test_latency_calculated_correctly() -> None:
    registry = MetricsRegistry()

    # Create events with explicit old timestamps to guarantee > 0 latency
    start_event = AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", input_text="test"
    )
    start_event.timestamp = "2023-01-01T12:00:00Z"
    
    end_event = AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", success=True, output="done"
    )
    end_event.timestamp = "2023-01-01T12:00:05Z" # 5 seconds later
    
    registry.record_event(start_event)
    registry.record_event(end_event)
    
    snapshot = registry.snapshot()
    assert snapshot["latency"]["execution"]["count"] == 1
    assert snapshot["latency"]["execution"]["total_ms"] == 5000.0


def test_malformed_timestamp_does_not_crash() -> None:
    registry = MetricsRegistry()

    start_event = AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", input_text="test"
    )
    start_event.timestamp = "not-a-timestamp"
    
    end_event = AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", success=True, output="done"
    )
    end_event.timestamp = "still-not-a-timestamp"
    
    # Should not raise exception
    registry.record_event(start_event)
    registry.record_event(end_event)
    
    snapshot = registry.snapshot()
    assert snapshot["latency"]["execution"]["count"] == 1


def test_failure_isolation() -> None:
    registry = MetricsRegistry()

    # Pass an object that will crash internal processing
    # (e.g. missing getattr attributes entirely, though Python getattr allows this.
    # We will trigger an error by patching a method it calls, or similar).
    
    class EvilEvent:
        @property
        def event_type(self):
            raise RuntimeError("Boom!")

    # Should safely catch and ignore the RuntimeError
    registry.record_event(EvilEvent())


def test_snapshot_is_safely_copied() -> None:
    registry = MetricsRegistry()

    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", input_text="test"
    ))
    
    snap1 = registry.snapshot()
    
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", success=True, output="done"
    ))
    
    snap2 = registry.snapshot()
    
    # Mutating snap1 should not affect anything
    snap1["executions"]["total"] = 999
    
    assert snap2["executions"]["total"] == 1
    assert snap2["executions"]["successful"] == 1


def test_step_failure_counting() -> None:
    registry = MetricsRegistry()
    registry.record_event(PlanStepCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        plan_id="p1", step_id="s1", status="failed"
    ))
    snapshot = registry.snapshot()
    assert snapshot["step_failures"] == 1


def test_handoff_failure() -> None:
    registry = MetricsRegistry()
    registry.record_event(HandoffResolvedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        target_task_type="research", success=False, messages_exchanged=1
    ))
    snapshot = registry.snapshot()
    assert snapshot["handoffs"]["failed"] == 1
    assert snapshot["handoffs"]["successful"] == 0


def test_concurrency() -> None:
    registry = MetricsRegistry()
    
    events = [
        RetryAttemptedEvent(
            trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
            error_type="Error", attempt_number=i
        ) for i in range(100)
    ]
    
    def record(event):
        registry.record_event(event)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(record, events)
        
    snapshot = registry.snapshot()
    assert snapshot["retries"]["total"] == 100


def test_failed_execution_remains_failed() -> None:
    registry = MetricsRegistry()

    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", input_text="test"
    ))
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", success=False, output="error"
    ))
    
    # Simulate a bug where success event is emitted after failure
    registry.record_event(AgentExecutionCompletedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", success=True, output="done"
    ))
    
    snapshot = registry.snapshot()
    assert snapshot["executions"]["failed"] == 1
    assert snapshot["executions"]["successful"] == 0


def test_timeout_cancellation_distinct() -> None:
    registry = MetricsRegistry()

    registry.record_event(AgentExecutionStartedEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        agent_name="AgentA", input_text="test"
    ))
    registry.record_event(TimeoutCancellationEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id=None,
        reason="timed_out"
    ))
    
    snapshot = registry.snapshot()
    assert snapshot["executions"]["timed_out"] == 1
    assert snapshot["executions"]["cancelled"] == 0
