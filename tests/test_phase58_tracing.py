"""Tests for Phase 5.8 Step 2 — Context / Span Propagation."""

from __future__ import annotations

import pytest

from app.agent.execution_context import AgentExecutionContext, ExecutionStatus
from app.observability.tracer import SpanInfo, extract_span_info


# ------------------------------------------------------------------ #
# 1. Context generates valid tracing identifiers
# ------------------------------------------------------------------ #

def test_context_generates_valid_tracing_identifiers() -> None:
    ctx = AgentExecutionContext(task="research")

    assert isinstance(ctx.trace_id, str)
    assert len(ctx.trace_id) > 0
    assert isinstance(ctx.run_id, str)
    assert len(ctx.run_id) > 0
    assert isinstance(ctx.span_id, str)
    assert len(ctx.span_id) > 0


# ------------------------------------------------------------------ #
# 2. trace_id, run_id, and span_id are distinct concepts
# ------------------------------------------------------------------ #

def test_tracing_ids_are_distinct() -> None:
    ctx = AgentExecutionContext(task="research")

    # All three should be different UUIDs
    assert ctx.trace_id != ctx.run_id
    assert ctx.trace_id != ctx.span_id
    assert ctx.run_id != ctx.span_id

    # They should also be distinct from the existing execution_id
    assert ctx.trace_id != ctx.execution_id
    assert ctx.run_id != ctx.execution_id
    assert ctx.span_id != ctx.execution_id


# ------------------------------------------------------------------ #
# 3. Root context has no parent span
# ------------------------------------------------------------------ #

def test_root_context_has_no_parent_span() -> None:
    ctx = AgentExecutionContext(task="research")

    assert ctx.parent_span_id is None


# ------------------------------------------------------------------ #
# 4. Child context preserves the parent's trace_id
# ------------------------------------------------------------------ #

def test_child_preserves_parent_trace_id() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    assert child.trace_id == parent.trace_id


# ------------------------------------------------------------------ #
# 5. Child context preserves the parent's run_id
# ------------------------------------------------------------------ #

def test_child_preserves_parent_run_id() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    assert child.run_id == parent.run_id


# ------------------------------------------------------------------ #
# 6. Child receives a new span_id
# ------------------------------------------------------------------ #

def test_child_receives_new_span_id() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    assert child.span_id != parent.span_id
    assert isinstance(child.span_id, str)
    assert len(child.span_id) > 0


# ------------------------------------------------------------------ #
# 7. Child's parent_span_id equals the parent's span_id
# ------------------------------------------------------------------ #

def test_child_parent_span_id_matches_parent_span_id() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    assert child.parent_span_id == parent.span_id


# ------------------------------------------------------------------ #
# 8. Multiple children have distinct span IDs
# ------------------------------------------------------------------ #

def test_multiple_children_have_distinct_span_ids() -> None:
    parent = AgentExecutionContext(task="parent task")
    child_a = parent.create_child_span(task="child A")
    child_b = parent.create_child_span(task="child B")
    child_c = parent.create_child_span(task="child C")

    span_ids = {child_a.span_id, child_b.span_id, child_c.span_id}
    assert len(span_ids) == 3

    # All children share the parent's trace and run
    assert child_a.trace_id == child_b.trace_id == child_c.trace_id == parent.trace_id
    assert child_a.run_id == child_b.run_id == child_c.run_id == parent.run_id

    # All children point back to the same parent span
    assert child_a.parent_span_id == parent.span_id
    assert child_b.parent_span_id == parent.span_id
    assert child_c.parent_span_id == parent.span_id


# ------------------------------------------------------------------ #
# 9. Parent and child do not accidentally share mutable state
# ------------------------------------------------------------------ #

def test_parent_and_child_do_not_share_mutable_state() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    # Each gets its own execution_id
    assert parent.execution_id != child.execution_id

    # Modifying child artifacts does not affect parent
    child.set_artifact("child_data", "value")
    assert parent.has_artifact("child_data") is False

    # Modifying parent metadata does not affect child
    parent.set_metadata("parent_key", "parent_value")
    assert child.metadata.get("parent_key") is None

    # Publishing output on parent does not appear on child
    parent.publish_agent_output(
        agent_id="agent-1",
        output="result",
        success=True,
    )
    assert child.get_agent_output("agent-1") is None

    # Output histories are separate
    assert len(parent.agent_output_history) == 1
    assert len(child.agent_output_history) == 0


# ------------------------------------------------------------------ #
# 10. Existing cancellation semantics remain intact
# ------------------------------------------------------------------ #

def test_cancellation_semantics_intact() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    # Child starts in CREATED, not cancelled
    assert child.is_cancelled is False
    assert child.status == ExecutionStatus.CREATED

    # Cancelling the parent does not automatically cancel the child
    # (the existing architecture uses a single shared context, not copies)
    parent.mark_cancelled()
    assert parent.is_cancelled is True
    assert child.is_cancelled is False

    # Child can be cancelled independently
    child.mark_cancelled()
    assert child.is_cancelled is True
    assert child.status == ExecutionStatus.CANCELLED


# ------------------------------------------------------------------ #
# 11. Existing timeout semantics remain intact
# ------------------------------------------------------------------ #

def test_timeout_semantics_intact() -> None:
    parent = AgentExecutionContext(task="parent task")
    child = parent.create_child_span(task="child task")

    # Child starts clean
    assert child.is_timed_out is False

    # Timing out the parent does not automatically time out the child
    parent.mark_timed_out()
    assert parent.is_timed_out is True
    assert child.is_timed_out is False

    # Child can time out independently
    child.mark_timed_out()
    assert child.is_timed_out is True
    assert child.status == ExecutionStatus.TIMED_OUT


# ------------------------------------------------------------------ #
# 12. Existing execution_id and correlation_id behavior remains intact
# ------------------------------------------------------------------ #

def test_execution_id_and_correlation_id_preserved() -> None:
    parent = AgentExecutionContext(
        task="parent task",
        correlation_id="user-session-42",
    )
    child = parent.create_child_span(task="child task")

    # execution_id is unique per context instance
    assert parent.execution_id != child.execution_id

    # correlation_id is inherited
    assert child.correlation_id == "user-session-42"

    # trace_id is NOT the same as correlation_id
    assert parent.trace_id != parent.correlation_id

    # run_id is NOT the same as execution_id
    assert parent.run_id != parent.execution_id


# ------------------------------------------------------------------ #
# 13. Tracer/span creation does not modify execution status
# ------------------------------------------------------------------ #

def test_span_creation_does_not_modify_status() -> None:
    parent = AgentExecutionContext(task="parent task")
    parent.mark_running()

    child = parent.create_child_span(task="child task")

    # Parent's status is unchanged
    assert parent.status == ExecutionStatus.RUNNING

    # Child starts in CREATED
    assert child.status == ExecutionStatus.CREATED


# ------------------------------------------------------------------ #
# 14. Tracing failure cannot become an execution failure
# ------------------------------------------------------------------ #

def test_extract_span_info_returns_none_for_none_context() -> None:
    result = extract_span_info(None)
    assert result is None


def test_extract_span_info_returns_none_for_incompatible_object() -> None:
    result = extract_span_info({"not": "a context"})
    assert result is None


def test_extract_span_info_returns_none_for_partial_context() -> None:
    """An object with only some tracing fields should not crash."""
    class PartialContext:
        trace_id = "abc"
        run_id = None  # Missing
        span_id = "def"

    result = extract_span_info(PartialContext())
    assert result is None


def test_extract_span_info_success() -> None:
    ctx = AgentExecutionContext(task="test")
    info = extract_span_info(ctx)

    assert info is not None
    assert isinstance(info, SpanInfo)
    assert info.trace_id == ctx.trace_id
    assert info.run_id == ctx.run_id
    assert info.span_id == ctx.span_id
    assert info.parent_span_id is None


def test_extract_span_info_from_child() -> None:
    parent = AgentExecutionContext(task="parent")
    child = parent.create_child_span(task="child")

    info = extract_span_info(child)
    assert info is not None
    assert info.trace_id == parent.trace_id
    assert info.run_id == parent.run_id
    assert info.span_id == child.span_id
    assert info.parent_span_id == parent.span_id


def test_extract_span_info_never_raises() -> None:
    """Even a completely broken object should not crash extract_span_info."""
    class BrokenContext:
        @property
        def trace_id(self):
            raise RuntimeError("boom")

    result = extract_span_info(BrokenContext())
    assert result is None


# ------------------------------------------------------------------ #
# 15. Existing Phase 5.6/5.7 collaboration behavior remains unaffected
# ------------------------------------------------------------------ #

def test_existing_context_behavior_unaffected() -> None:
    """Verify that all pre-existing context operations still work."""
    ctx = AgentExecutionContext(task="test")

    # Lifecycle transitions
    ctx.mark_running()
    assert ctx.status == ExecutionStatus.RUNNING

    ctx.mark_executing()
    assert ctx.status == ExecutionStatus.EXECUTING

    ctx.mark_completed()
    assert ctx.status == ExecutionStatus.COMPLETED

    # Artifacts
    ctx2 = AgentExecutionContext(task="test2")
    ctx2.set_artifact("key", "value")
    assert ctx2.get_artifact("key") == "value"

    # Agent outputs
    ctx2.publish_agent_output(
        agent_id="agent-1",
        output="done",
        success=True,
    )
    output = ctx2.get_agent_output("agent-1")
    assert output is not None
    assert output.output == "done"

    # Metadata
    ctx2.set_metadata("priority", "high")
    assert ctx2.metadata["priority"] == "high"


def test_child_inherits_user_and_chat_ids() -> None:
    parent = AgentExecutionContext(
        task="parent",
        user_id="user-123",
        chat_id="chat-456",
    )
    child = parent.create_child_span(task="child")

    assert child.user_id == "user-123"
    assert child.chat_id == "chat-456"


# ------------------------------------------------------------------ #
# SpanInfo immutability
# ------------------------------------------------------------------ #

def test_span_info_is_immutable() -> None:
    info = SpanInfo(
        trace_id="t1",
        run_id="r1",
        span_id="s1",
        parent_span_id=None,
    )

    with pytest.raises(AttributeError):
        info.trace_id = "modified"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        info.span_id = "modified"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# Grandchild spans
# ------------------------------------------------------------------ #

def test_grandchild_span_propagation() -> None:
    root = AgentExecutionContext(task="root")
    child = root.create_child_span(task="child")
    grandchild = child.create_child_span(task="grandchild")

    # All share the same trace_id and run_id
    assert grandchild.trace_id == child.trace_id == root.trace_id
    assert grandchild.run_id == child.run_id == root.run_id

    # Grandchild's parent is the child
    assert grandchild.parent_span_id == child.span_id

    # Child's parent is the root
    assert child.parent_span_id == root.span_id

    # Root has no parent
    assert root.parent_span_id is None

    # All span_ids are unique
    assert len({root.span_id, child.span_id, grandchild.span_id}) == 3
