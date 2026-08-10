"""Tests for Phase 5.5 plan contracts and validation."""

import pytest

from app.agent.plan import Plan, PlanStep, PlanStepStatus, PlanStatus, StepResult
from app.agent.validator import PlanValidator
from app.exceptions import PlanValidationError


# ------------------------------------------------------------------ #
# Plan contract tests
# ------------------------------------------------------------------ #

def test_plan_step_defaults() -> None:
    step = PlanStep(step_id="s1", description="Do research")
    assert step.status == PlanStepStatus.PENDING
    assert step.dependencies == ()
    assert step.required_capabilities == frozenset()
    assert step.task_type is None
    assert step.result is None


def test_plan_defaults() -> None:
    plan = Plan(goal="Test goal")
    assert plan.status == PlanStatus.PENDING
    assert plan.steps == {}
    assert plan.plan_id  # auto-generated UUID


def test_step_result_wraps_agent_result() -> None:
    result = StepResult(step_id="s1", success=True, error=None)
    assert result.success is True
    assert result.agent_result is None


# ------------------------------------------------------------------ #
# PlanValidator tests
# ------------------------------------------------------------------ #

def _make_plan(**kwargs) -> Plan:
    defaults = {"goal": "Test goal"}
    defaults.update(kwargs)
    return Plan(**defaults)


def test_validator_accepts_valid_plan() -> None:
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Step 1"),
        "s2": PlanStep(step_id="s2", description="Step 2", dependencies=("s1",)),
    })
    PlanValidator().validate(plan)  # Should not raise


def test_validator_rejects_empty_goal() -> None:
    plan = _make_plan(goal="", steps={"s1": PlanStep(step_id="s1", description="X")})
    with pytest.raises(PlanValidationError, match="goal must not be empty"):
        PlanValidator().validate(plan)


def test_validator_rejects_empty_steps() -> None:
    plan = _make_plan(steps={})
    with pytest.raises(PlanValidationError, match="at least one step"):
        PlanValidator().validate(plan)


def test_validator_rejects_missing_dependency() -> None:
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Step 1", dependencies=("missing",)),
    })
    with pytest.raises(PlanValidationError, match="unknown step 'missing'"):
        PlanValidator().validate(plan)


def test_validator_rejects_cyclic_dependencies() -> None:
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Step 1", dependencies=("s2",)),
        "s2": PlanStep(step_id="s2", description="Step 2", dependencies=("s1",)),
    })
    with pytest.raises(PlanValidationError, match="cyclic dependencies"):
        PlanValidator().validate(plan)


def test_validator_rejects_self_dependency() -> None:
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Step 1", dependencies=("s1",)),
    })
    with pytest.raises(PlanValidationError, match="cyclic dependencies"):
        PlanValidator().validate(plan)


def test_validator_rejects_invalid_initial_status() -> None:
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Step 1", status=PlanStepStatus.COMPLETED),
    })
    with pytest.raises(PlanValidationError, match="invalid initial status"):
        PlanValidator().validate(plan)


def test_validator_accepts_branching_dag() -> None:
    """
    s1 ──┐
         ├──→ s3
    s2 ──┘
    """
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Research"),
        "s2": PlanStep(step_id="s2", description="Verification"),
        "s3": PlanStep(step_id="s3", description="Synthesis", dependencies=("s1", "s2")),
    })
    PlanValidator().validate(plan)  # Should not raise


def test_validator_accepts_diamond_dag() -> None:
    """
         s1
        / \
       s2   s3
        \ /
         s4
    """
    plan = _make_plan(steps={
        "s1": PlanStep(step_id="s1", description="Start"),
        "s2": PlanStep(step_id="s2", description="Branch A", dependencies=("s1",)),
        "s3": PlanStep(step_id="s3", description="Branch B", dependencies=("s1",)),
        "s4": PlanStep(step_id="s4", description="Merge", dependencies=("s2", "s3")),
    })
    PlanValidator().validate(plan)  # Should not raise
