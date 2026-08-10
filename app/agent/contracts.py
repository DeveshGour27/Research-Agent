"""Typed contracts for agent execution in the Production AI Research Agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.agent.state import AgentState
from app.exceptions import AgentError
from app.agent.execution_context import AgentExecutionContext


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """Stable identity information for an agent implementation."""

    name: str
    version: str = "1.0.0"
    description: str = ""


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Declarative capabilities exposed by an agent implementation."""

    tool_use: bool = True
    memory: bool = True
    multi_turn: bool = True
    retrieval: bool = True
    task_types: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Structured input provided to an agent for one execution."""

    input_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    context: AgentExecutionContext | None = None
    correlation_id: str | None = None
    sender_id: str | None = None

@dataclass(slots=True)
class AgentResult:
    """Structured result returned by a completed agent execution."""

    request: AgentRequest
    state: AgentState
    output: str | None
    success: bool
    context: AgentExecutionContext | None = None
    error: "AgentExecutionError | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentExecutionError(AgentError):
    """Raised when an agent cannot execute a request successfully."""

    def __init__(
        self,
        message: str,
        *,
        request: AgentRequest | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details or {})
        self.request = request


class BaseAgent(ABC):
    """Minimal abstraction for provider-neutral agent implementations."""

    @property
    @abstractmethod
    def identity(self) -> AgentIdentity:
        """Return the agent's stable identity information."""

    @property
    @abstractmethod
    def capabilities(self) -> AgentCapabilities:
        """Return the capabilities the agent exposes."""

    @property
    def description(self) -> str:
        """Return the agent description from its identity metadata."""
        return self.identity.description

    @abstractmethod
    def execute(self, request: AgentRequest) -> AgentResult:
        """Execute a structured request and return a structured result."""
