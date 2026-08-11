"""Planning contracts for structured multi-step execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from app.agent.contracts import AgentResult


class PlanStepStatus(str, Enum):
    """Lifecycle status of an individual plan step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStatus(str, Enum):
    """Lifecycle status of an entire plan."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


@dataclass(slots=True)
class StepResult:
    """Result of executing a single plan step."""

    step_id: str
    success: bool
    agent_result: AgentResult | None = None
    error: str | None = None


@dataclass(slots=True)
class PlanStep:
    """A single step in an execution plan.

    Steps describe required work via task_type and required_capabilities,
    NOT via hard-coded agent IDs. The Router decides which agent executes.
    """

    step_id: str
    description: str
    task_type: str | None = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: StepResult | None = None
    is_required: bool = True


@dataclass(slots=True)
class Plan:
    """A structured execution plan containing dependent steps."""

    plan_id: str = field(default_factory=lambda: str(uuid4()))
    goal: str = ""
    steps: dict[str, PlanStep] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.PENDING
    metadata: dict[str, str] = field(default_factory=dict)
