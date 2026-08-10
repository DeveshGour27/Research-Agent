"""Deterministic structural validation for execution plans."""

from __future__ import annotations

from collections import deque

from app.agent.plan import Plan, PlanStepStatus
from app.exceptions import PlanValidationError


class PlanValidator:
    """Validates a Plan for structural correctness before execution.

    Checks performed:
    - Plan must have a non-empty goal
    - Plan must have at least one step
    - No duplicate step IDs (enforced by dict keys)
    - All dependency references must resolve to existing step IDs
    - No cyclic dependencies (topological sort via Kahn's algorithm)
    - Steps with no dependencies must be marked PENDING (not already terminal)
    """

    def validate(self, plan: Plan) -> None:
        """Validate a plan. Raises PlanValidationError on any structural issue."""
        self._validate_not_empty(plan)
        self._validate_dependencies_exist(plan)
        self._validate_no_cycles(plan)
        self._validate_initial_states(plan)

    @staticmethod
    def _validate_not_empty(plan: Plan) -> None:
        if not plan.goal or not plan.goal.strip():
            raise PlanValidationError(
                "Plan goal must not be empty.",
                details={"plan_id": plan.plan_id},
            )
        if not plan.steps:
            raise PlanValidationError(
                "Plan must contain at least one step.",
                details={"plan_id": plan.plan_id},
            )

    @staticmethod
    def _validate_dependencies_exist(plan: Plan) -> None:
        step_ids = set(plan.steps.keys())
        for step_id, step in plan.steps.items():
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    raise PlanValidationError(
                        f"Step '{step_id}' depends on unknown step '{dep_id}'.",
                        details={
                            "plan_id": plan.plan_id,
                            "step_id": step_id,
                            "missing_dependency": dep_id,
                        },
                    )

    @staticmethod
    def _validate_no_cycles(plan: Plan) -> None:
        in_degree: dict[str, int] = {sid: 0 for sid in plan.steps}
        adjacency: dict[str, list[str]] = {sid: [] for sid in plan.steps}

        for step_id, step in plan.steps.items():
            for dep_id in step.dependencies:
                adjacency[dep_id].append(step_id)
                in_degree[step_id] += 1

        queue: deque[str] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )
        visited_count = 0

        while queue:
            current = queue.popleft()
            visited_count += 1
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(plan.steps):
            raise PlanValidationError(
                "Plan contains cyclic dependencies.",
                details={"plan_id": plan.plan_id},
            )

    @staticmethod
    def _validate_initial_states(plan: Plan) -> None:
        for step_id, step in plan.steps.items():
            if step.status not in (PlanStepStatus.PENDING, PlanStepStatus.READY):
                raise PlanValidationError(
                    f"Step '{step_id}' has invalid initial status '{step.status.value}'.",
                    details={
                        "plan_id": plan.plan_id,
                        "step_id": step_id,
                        "status": step.status.value,
                    },
                )
