"""Tests for the Phase 5.2 shared execution context."""

from __future__ import annotations

from app.agent.execution_context import (
    AgentExecutionContext,
    ExecutionStatus,
)


def test_context_has_unique_execution_id() -> None:
    first = AgentExecutionContext(task="Task one")
    second = AgentExecutionContext(task="Task two")

    assert first.execution_id != second.execution_id


def test_context_starts_created() -> None:
    context = AgentExecutionContext(task="Research AI agents")

    assert context.status == ExecutionStatus.CREATED
    assert context.task == "Research AI agents"


def test_context_lifecycle() -> None:
    context = AgentExecutionContext(task="Research AI agents")

    context.mark_running()
    assert context.status == ExecutionStatus.RUNNING

    context.mark_completed()
    assert context.status == ExecutionStatus.COMPLETED


def test_context_can_store_shared_artifact() -> None:
    context = AgentExecutionContext(task="Research AI")

    context.set_artifact(
        "research_results",
        ["result one", "result two"],
    )

    assert context.has_artifact("research_results") is True
    assert context.get_artifact("research_results") == [
        "result one",
        "result two",
    ]


def test_context_can_remove_artifact() -> None:
    context = AgentExecutionContext(task="Research AI")
    context.set_artifact("research_results", ["result"])

    removed = context.remove_artifact("research_results")

    assert removed == ["result"]
    assert context.has_artifact("research_results") is False


def test_context_rejects_empty_artifact_key() -> None:
    context = AgentExecutionContext(task="Research AI")

    try:
        context.set_artifact("", "value")
    except ValueError as error:
        assert str(error) == "Artifact key must not be empty."
    else:
        raise AssertionError("Expected ValueError")


def test_context_can_publish_agent_output() -> None:
    context = AgentExecutionContext(task="Research AI")

    context.publish_agent_output(
        agent_id="researcher",
        output="Research completed.",
        success=True,
        metadata={"sources": 5},
    )

    result = context.get_agent_output("researcher")

    assert result is not None
    assert result.agent_id == "researcher"
    assert result.output == "Research completed."
    assert result.success is True
    assert result.metadata["sources"] == 5


def test_context_keeps_agent_outputs_separate() -> None:
    context = AgentExecutionContext(task="Research AI")

    context.publish_agent_output(
        agent_id="researcher",
        output="Research result",
        success=True,
    )

    context.publish_agent_output(
        agent_id="analyst",
        output="Analysis result",
        success=True,
    )

    assert context.get_agent_output("researcher").output == "Research result"
    assert context.get_agent_output("analyst").output == "Analysis result"


def test_context_metadata_is_shared() -> None:
    context = AgentExecutionContext(task="Research AI")

    context.set_metadata("priority", "high")

    assert context.metadata["priority"] == "high"


def test_contexts_are_isolated() -> None:
    first = AgentExecutionContext(task="Task one")
    second = AgentExecutionContext(task="Task two")

    first.set_artifact("data", "first")

    assert first.get_artifact("data") == "first"
    assert second.get_artifact("data") is None