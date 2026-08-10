"""Coordination-level retry boundary for agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from app.agent.communicator import AgentCommunicator
from app.agent.contracts import AgentRequest, AgentResult
from app.exceptions import CommunicationError
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    Policy for retrying agent communication at the coordination boundary.

    Only transient infrastructure/transport errors should be retried.
    Contract validation or routing errors are deterministic and will fail again.
    """
    max_attempts: int = 3
    retryable_errors: tuple[Type[Exception], ...] = (CommunicationError,)


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

    def send(self, request: AgentRequest) -> AgentResult:
        """
        Send a request, retrying on transient errors according to policy.

        Raises:
            Exception: Re-raises the last exception if max_attempts is exhausted
                       or if the exception is not retryable.
        """
        attempts_left = self._policy.max_attempts
        last_error: Exception | None = None

        while attempts_left > 0:
            attempts_left -= 1
            attempt_number = self._policy.max_attempts - attempts_left

            try:
                return self._communicator.send(request)
            except Exception as error:
                last_error = error
                
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
