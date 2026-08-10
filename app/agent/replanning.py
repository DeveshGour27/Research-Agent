"""Replanning policy for plan execution failure recovery."""

from __future__ import annotations

from app.agent.execution_policy import ExecutionPolicy
from app.agent.plan import Plan, PlanStatus
from app.logger import get_logger

logger = get_logger(__name__)


class ReplanningPolicy:
    """Decides whether replanning is permitted after a plan failure.

    Evaluates the execution result against the ExecutionPolicy
    and historical replan count. Exposes a single question:
    should_replan() -> bool.
    """

    def __init__(self, policy: ExecutionPolicy) -> None:
        self._policy = policy
        self._replan_count: int = 0

    @property
    def replan_count(self) -> int:
        return self._replan_count

    def should_replan(self, plan: Plan) -> bool:
        """Determine whether replanning should be attempted.

        Args:
            plan: The plan that was just executed (expected FAILED status).

        Returns:
            True if replanning is allowed and within limits.
        """
        if not self._policy.allow_replanning:
            logger.info(
                "Replanning disabled by policy",
                extra={"plan_id": plan.plan_id},
            )
            return False

        if plan.status != PlanStatus.FAILED:
            logger.info(
                "Replanning not needed, plan did not fail",
                extra={
                    "plan_id": plan.plan_id,
                    "status": plan.status.value,
                },
            )
            return False

        if self._replan_count >= self._policy.max_replans:
            logger.warning(
                "Maximum replans reached",
                extra={
                    "plan_id": plan.plan_id,
                    "replan_count": self._replan_count,
                    "max_replans": self._policy.max_replans,
                },
            )
            return False

        logger.info(
            "Replanning permitted",
            extra={
                "plan_id": plan.plan_id,
                "replan_count": self._replan_count,
                "max_replans": self._policy.max_replans,
            },
        )
        return True

    def record_replan(self) -> None:
        """Record that a replan has been attempted."""
        self._replan_count += 1
