"""Coordination-level retry boundary for agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from app.agent.communicator import AgentCommunicator
from app.agent.contracts import AgentRequest, AgentResult
from app.exceptions import CommunicationError, RetryableError, FatalError
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    Policy for retrying agent communication at the coordination boundary.
    """
    max_attempts: int = 3
    max_repeated_failures: int = 5
    max_failures_per_task: int = 10
    retryable_errors: tuple[Type[Exception], ...] = (CommunicationError, RetryableError)


class RetryBoundary:
    """
    Applies a retry policy around an AgentCommunicator.
    """

    def __init__(
        self,
        policy: RetryPolicy,
        communicator: AgentCommunicator,
    ) -> None:
        self._policy = policy
        self._communicator = communicator
        self.total_failures = 0
        self.repeated_failures = 0
        self.task_failures: dict[str, int] = {}

    def send(self, request: AgentRequest) -> AgentResult:
        """
        Send a request, retrying on transient errors according to policy.
        """
        attempts_left = self._policy.max_attempts
        last_error: Exception | None = None
        task_id = request.request_id or "unknown"

        while attempts_left > 0:
            # Check budgets before attempt
            if self.repeated_failures >= self._policy.max_repeated_failures:
                raise FatalError("Max repeated failures budget exceeded.", request=request)
                
            if self.task_failures.get(task_id, 0) >= self._policy.max_failures_per_task:
                raise FatalError("Max failures per task budget exceeded.", request=request)

            attempts_left -= 1
            attempt_number = self._policy.max_attempts - attempts_left

            try:
                result = self._communicator.send(request)
                self.repeated_failures = 0  # reset on success
                return result
            except Exception as error:
                last_error = error
                self.total_failures += 1
                self.repeated_failures += 1
                self.task_failures[task_id] = self.task_failures.get(task_id, 0) + 1
                
                if isinstance(error, FatalError):
                    logger.warning("Fatal error encountered, aborting retry.", extra={"error_type": type(error).__name__})
                    raise

                # Check if this error type is retryable
                is_retryable = any(
                    isinstance(error, retryable_type)
                    for retryable_type in self._policy.retryable_errors
                )

                if not is_retryable or attempts_left == 0:
                    logger.warning(
                        "Retry boundary aborting",
                        extra={
                            "attempt": attempt_number,
                            "correlation_id": request.correlation_id,
                            "request_id": request.request_id,
                            "error_type": type(error).__name__,
                            "retryable": is_retryable,
                        },
                    )
                    raise

                logger.info(
                    "Retrying transient communication error",
                    extra={
                        "attempt": attempt_number,
                        "correlation_id": request.correlation_id,
                        "request_id": request.request_id,
                        "error_type": type(error).__name__,
                    },
                )

        # Should never reach here if max_attempts > 0 because we raise in loop
        if last_error:
            raise last_error # pragma: no cover
        raise RuntimeError("Retry policy configured with 0 max attempts.")
