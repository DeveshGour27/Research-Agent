"""LLM-backed plan generation for structured multi-step execution."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from app.agent.execution_context import AgentExecutionContext
from app.agent.plan import Plan, PlanStep, StepResult
from app.exceptions import PlanCreationError
from app.logger import get_logger

logger = get_logger(__name__)


class Planner(ABC):
    """Provider-neutral plan generator.

    Responsible for producing a structured Plan from a goal.
    Must NOT access Registry, execute agents, or own retry logic.
    """

    @abstractmethod
    def generate_plan(
        self,
        goal: str,
        context: AgentExecutionContext,
        previous_plan: Plan | None = None,
        failure_context: list[StepResult] | None = None,
    ) -> Plan:
        """Generate a plan for the given goal.

        Args:
            goal: The task to plan for.
            context: The shared execution context.
            previous_plan: An optional failed plan for replanning.
            failure_context: Optional list of failed step results for replanning.

        Returns:
            A structured Plan ready for validation and execution.

        Raises:
            PlanCreationError: If the plan cannot be generated.
        """
