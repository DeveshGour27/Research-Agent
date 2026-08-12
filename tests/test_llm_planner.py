"""Tests for Phase 6.3 LLMPlanner."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agent.execution_context import AgentExecutionContext
from app.agent.llm_planner import ALLOWED_CAPABILITIES, ALLOWED_TASK_TYPES, LLMPlanner
from app.agent.plan import Plan, PlanStepStatus, StepResult
from app.exceptions import PlanCreationError
from app.llm.base import ChatResponse, LLMProvider, LLMResponse, ToolCall
from app.observability.events import PlanGeneratedEvent


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    return provider


@pytest.fixture
def context() -> AgentExecutionContext:
    return AgentExecutionContext(task="Test Task")


@pytest.fixture
def planner(mock_provider: MagicMock) -> LLMPlanner:
    return LLMPlanner(provider=mock_provider)


def create_mock_response(arguments_dict: dict | None = None, malformed_json: bool = False, no_tool_call: bool = False) -> LLMResponse:
    if no_tool_call:
        return LLMResponse(model="test", content="I cannot do that.")
    
    if malformed_json:
        args = {"__parse_error__": "Invalid JSON", "__raw_arguments__": "{ malformed"}
    else:
        args = arguments_dict or {}

    return LLMResponse(
        model="test",
        tool_call=ToolCall(id="call_1", name="submit_plan", arguments=args),
    )


def test_valid_single_step_plan(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "Do web search",
                "task_type": "web_search",
                "required_capabilities": ["web_search"],
                "dependencies": []
            }
        ]
    })

    with patch.object(PlanGeneratedEvent, 'emit') as mock_emit:
        plan = planner.generate_plan("Do something", context)
        
        assert plan.goal == "Do something"
        assert len(plan.steps) == 1
        assert "step_1" in plan.steps
        
        step = plan.steps["step_1"]
        assert step.step_id == "step_1"
        assert step.task_type == "web_search"
        assert "web_search" in step.required_capabilities
        assert step.status == PlanStepStatus.PENDING
        
        mock_emit.assert_called_once()
        # Ensure context ids are passed
        assert mock_emit.call_args[0] == ()
        # Event is constructed with context IDs


def test_valid_multi_step_dag(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "First step",
                "task_type": "web_search",
                "required_capabilities": [],
                "dependencies": []
            },
            {
                "step_id": "step_2",
                "description": "Second step",
                "task_type": "rag_search",
                "required_capabilities": ["retrieval"],
                "dependencies": ["step_1"]
            }
        ]
    })

    plan = planner.generate_plan("Do something", context)
    assert len(plan.steps) == 2
    assert "step_1" in plan.steps["step_2"].dependencies


def test_missing_tool_call(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response(no_tool_call=True)
    with pytest.raises(PlanCreationError, match="submit_plan"):
        planner.generate_plan("Do something", context)


def test_malformed_json(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response(malformed_json=True)
    with pytest.raises(PlanCreationError, match="malformed JSON arguments"):
        planner.generate_plan("Do something", context)


def test_empty_tool_call_arguments(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({})
    with pytest.raises(PlanCreationError, match="missing 'steps'"):
        planner.generate_plan("Do something", context)


def test_missing_required_plan_fields(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {
                "description": "No step_id"
            }
        ]
    })
    with pytest.raises(PlanCreationError, match="missing a valid 'step_id'"):
        planner.generate_plan("Do something", context)


def test_duplicate_step_ids(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "A",
                "task_type": "web_search",
            },
            {
                "step_id": "step_1",
                "description": "B",
                "task_type": "rag_search",
            }
        ]
    })
    with pytest.raises(PlanCreationError, match="Duplicate step ID"):
        planner.generate_plan("Do something", context)


def test_missing_dependency(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "A",
                "task_type": "web_search",
                "dependencies": ["missing_step"]
            }
        ]
    })
    with pytest.raises(PlanCreationError, match="unknown step 'missing_step'"):
        planner.generate_plan("Do something", context)


def test_cyclic_dependency(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {
                "step_id": "step_1",
                "description": "A",
                "task_type": "web_search",
                "dependencies": ["step_2"]
            },
            {
                "step_id": "step_2",
                "description": "B",
                "task_type": "rag_search",
                "dependencies": ["step_1"]
            }
        ]
    })
    with pytest.raises(PlanCreationError, match="cyclic dependencies"):
        planner.generate_plan("Do something", context)


def test_empty_plan(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({"steps": []})
    with pytest.raises(PlanCreationError, match="at least one step"):
        planner.generate_plan("Do something", context)


def test_empty_goal(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [{"step_id": "1", "description": "1", "task_type": "web_search"}]
    })
    with pytest.raises(PlanCreationError, match="goal must not be empty"):
        planner.generate_plan("   ", context)


def test_unknown_task_type(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [{"step_id": "1", "description": "1", "task_type": "hack_mainframe"}]
    })
    with pytest.raises(PlanCreationError, match="unknown task type: 'hack_mainframe'"):
        planner.generate_plan("Do something", context)


def test_unknown_capability(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [{"step_id": "1", "description": "1", "task_type": "web_search", "required_capabilities": ["hack_mainframe"]}]
    })
    with pytest.raises(PlanCreationError, match="unknown capability: 'hack_mainframe'"):
        planner.generate_plan("Do something", context)


def test_excessive_step_count(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    planner._max_plan_steps = 2
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {"step_id": "1", "description": "1", "task_type": "web_search"},
            {"step_id": "2", "description": "2", "task_type": "rag_search"},
            {"step_id": "3", "description": "3", "task_type": "web_search"}
        ]
    })
    with pytest.raises(PlanCreationError, match="exceeds maximum allowed steps"):
        planner.generate_plan("Do something", context)


def test_llm_provider_exception(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.side_effect = Exception("API Error")
    with pytest.raises(PlanCreationError, match="LLM provider failed"):
        planner.generate_plan("Do something", context)


def test_replanning_context(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [{"step_id": "1", "description": "1", "task_type": "web_search"}]
    })
    
    prev_plan = Plan(goal="Do something")
    failure_ctx = [StepResult(step_id="old_1", success=False, error="Tool crashed")]
    
    plan = planner.generate_plan("Do something", context, prev_plan, failure_ctx)
    
    # Verify the prompt includes replanning instructions
    calls = mock_provider.generate_with_tools.call_args_list
    assert len(calls) == 1
    messages = calls[0][0][0]
    user_prompt = messages[1]["content"]
    assert "REPLANNING CONTEXT" in user_prompt
    assert "old_1" in user_prompt
    assert "Tool crashed" in user_prompt


def test_event_not_emitted_on_validation_failure(planner: LLMPlanner, mock_provider: MagicMock, context: AgentExecutionContext) -> None:
    # E.g. cyclic dependency
    mock_provider.generate_with_tools.return_value = create_mock_response({
        "steps": [
            {"step_id": "1", "description": "1", "task_type": "web_search", "dependencies": ["2"]},
            {"step_id": "2", "description": "2", "task_type": "web_search", "dependencies": ["1"]},
        ]
    })
    
    with patch.object(PlanGeneratedEvent, 'emit') as mock_emit:
        with pytest.raises(PlanCreationError):
            planner.generate_plan("Do something", context)
        
        mock_emit.assert_not_called()
