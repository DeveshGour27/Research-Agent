"""
Custom exception hierarchy for the Production AI Research Agent.

Responsibilities:
    - Define a base exception so callers can catch all agent errors with
      a single ``except AgentError`` clause
    - Group exceptions by subsystem for precise, targeted error handling
    - Carry structured context (not just a message string) so that the
      logging layer can emit rich, searchable records

Design Decision:
    Every exception stores a ``details`` dict. This avoids the anti-pattern
    of embedding dynamic data inside the message string, which makes log
    parsing harder and forces brittle regex on the consumer side.

Usage:
    raise ConfigurationError(
        "OPENAI_API_KEY is missing",
        details={"env_var": "OPENAI_API_KEY"},
    )
"""

from __future__ import annotations

from typing import Any


# ------------------------------------------------------------------ #
# Base
# ------------------------------------------------------------------ #

class AgentError(Exception):
    """
    Base exception for the Production AI Research Agent.

    All custom exceptions inherit from this class, enabling broad
    ``except AgentError`` catches at the top of the call stack.

    Args:
        message: Human-readable description of what went wrong.
        details: Optional mapping of structured context (logged as JSON).
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = details or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.message!r}, details={self.details!r})"


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

class ConfigurationError(AgentError):
    """
    Raised when required configuration is missing or invalid.

    Examples:
        - OPENAI_API_KEY not set in production
        - Invalid log level string
        - chunk_overlap >= chunk_size
    """


# ------------------------------------------------------------------ #
# Tool execution
# ------------------------------------------------------------------ #

class ToolError(AgentError):
    """Base class for errors that originate inside a Tool."""


class ToolExecutionError(ToolError):
    """
    Raised when a tool fails to complete its task.

    Args:
        message: Description of the failure.
        tool_name: The tool that raised the error.
        details: Any additional structured context.
    """

    def __init__(
        self,
        message: str,
        tool_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details={"tool_name": tool_name, **(details or {})})
        self.tool_name: str = tool_name


class ToolNotFoundError(ToolError):
    """Raised when the Tool Manager cannot locate a requested tool by name."""


class ToolInputValidationError(ToolError):
    """Raised when the arguments passed to a tool fail validation."""


class ToolOutputValidationError(ToolError):
    """Raised when a tool returns output that does not match its schema."""


# ------------------------------------------------------------------ #
# Retrieval / RAG
# ------------------------------------------------------------------ #

class RetrievalError(AgentError):
    """Base class for errors in the retrieval (RAG) pipeline."""


class EmbeddingError(RetrievalError):
    """Raised when generating or storing embeddings fails."""


class VectorStoreError(RetrievalError):
    """Raised when the vector database operation fails."""


class DocumentChunkingError(RetrievalError):
    """Raised when a document cannot be split into chunks."""


# ------------------------------------------------------------------ #
# Memory
# ------------------------------------------------------------------ #

class MemoryError(AgentError):
    """Base class for memory subsystem errors."""


class MemoryReadError(MemoryError):
    """Raised when reading from memory fails."""


class MemoryWriteError(MemoryError):
    """Raised when writing to memory fails."""


# ------------------------------------------------------------------ #
# Planning
# ------------------------------------------------------------------ #

class PlannerError(AgentError):
    """Base class for planner errors."""


class PlanCreationError(PlannerError):
    """Raised when the planner cannot produce an execution plan."""


class PlanExecutionError(PlannerError):
    """Raised when a previously valid plan fails during execution."""


class PlanValidationError(PlannerError):
    """Raised when a plan fails structural validation."""


# ------------------------------------------------------------------ #
# Reflection
# ------------------------------------------------------------------ #

class ReflectionError(AgentError):
    """Raised when the reflection step cannot complete."""


# ------------------------------------------------------------------ #
# Output / Report generation
# ------------------------------------------------------------------ #

class OutputError(AgentError):
    """Base class for report and output generation errors."""


class ReportGenerationError(OutputError):
    """Raised when formatting or writing the final report fails."""


# ------------------------------------------------------------------ #
# LLM / API
# ------------------------------------------------------------------ #

class LLMError(AgentError):
    """Base class for errors related to the LLM provider."""


class LLMAPIError(LLMError):
    """
    Raised when the LLM API call fails (network, rate-limit, server error).

    Args:
        message: Description of the failure.
        status_code: HTTP status code returned by the API, if available.
        details: Additional structured context.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details={"status_code": status_code, **(details or {})})
        self.status_code: int | None = status_code


class LLMResponseParseError(LLMError):
    """Raised when the LLM response cannot be parsed into the expected format."""


# ------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------ #

class InputValidationError(AgentError):
    """Raised when user-supplied input fails validation before processing."""


# ------------------------------------------------------------------ #
# Agent Coordination
# ------------------------------------------------------------------ #

class AgentNotFoundError(AgentError):
    """Raised when the Agent Registry cannot locate a requested agent by ID."""


class RoutingError(AgentError):
    """Raised when the Router cannot select an appropriate agent for a request."""


class CommunicationError(AgentError):
    """Raised when the Communicator fails to deliver a message or request due to transport/infrastructure issues."""


class ContractValidationError(AgentError):
    """Raised when an AgentRequest or AgentResult fails validation."""


class ExecutionStateError(AgentError):
    """Raised when an illegal execution lifecycle transition is attempted."""


class AgentExecutionError(AgentError):
    """
    Raised when an agent cannot execute a request successfully.
    This acts as a base for more specific failure categories while preserving backward compatibility.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details={"request_id": request.request_id if request else None, **(details or {})})
        self.request = request


# ------------------------------------------------------------------ #
# Phase 5.7: Multi-Agent Failure Categories
# ------------------------------------------------------------------ #

class RetryableError(AgentExecutionError):
    """The operation failed in a way that may safely be retried."""

class RecoverableError(AgentExecutionError):
    """The operation failed but the supervisor may choose an alternative recovery path."""

class FatalError(AgentExecutionError):
    """The operation or system cannot safely recover."""

class AgentTimeoutError(AgentExecutionError):
    """A bounded execution exceeded its allowed time."""

class AgentCancellationError(AgentExecutionError):
    """The execution was explicitly cancelled."""


class InvalidStateTransitionError(AgentError):
    """Raised when an invalid job state transition is attempted."""

    def __init__(
        self,
        message: str,
        *,
        current_state: str,
        target_state: str,
        job_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            details={
                "current_state": current_state,
                "target_state": target_state,
                "job_id": job_id,
            },
        )
        self.current_state = current_state
        self.target_state = target_state

