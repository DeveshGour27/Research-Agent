"""Shared execution context for multi-agent task coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from app.agent.delegation import AgentDelegation
from app.agent.messaging import AgentMessage

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.agent.collaboration import Artifact


class ExecutionStatus(str, Enum):
    """Lifecycle status of a shared agent execution."""

    CREATED = "created"
    RUNNING = "running"
    EXECUTING = "executing"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class AgentOutput:
    """Structured output published by one agent."""

    agent_id: str
    output: str | None
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentExecutionContext:
    """
    Task-level context shared by participating agents.

    Private AgentState objects are never stored here.
    """

    task: str
    user_id: str | None = None
    chat_id: str | None = None

    execution_id: str = field(
        default_factory=lambda: str(uuid4())
    )
    correlation_id: str | None = None

    status: ExecutionStatus = ExecutionStatus.CREATED

    artifacts: dict[str, Any] = field(default_factory=dict)

    # Backward-compatible latest output per stable agent identity.
    agent_outputs: dict[str, AgentOutput] = field(default_factory=dict)

    # Complete output history. Prevents duplicate executions from losing data.
    agent_output_history: list[AgentOutput] = field(default_factory=list)

    # Unique message IDs prevent collisions.
    messages: dict[str, AgentMessage] = field(default_factory=dict)

    # Unique delegation IDs prevent collisions.
    delegations: dict[str, AgentDelegation] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Phase 5.8: Tracing identifiers.
    # These are deliberately separate from execution_id and correlation_id.
    # execution_id identifies the shared execution context instance.
    # correlation_id is an optional user-supplied or supervisor-supplied correlator.
    # trace_id groups all spans within a single top-level user request.
    # run_id identifies one execution run within the trace.
    # span_id identifies this specific execution unit within the run.
    # parent_span_id links to the parent execution unit.
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None

    def create_child_span(self, *, task: str) -> "AgentExecutionContext":
        """Create a child execution context that inherits the trace lineage.

        The child receives:
        - The same ``trace_id`` and ``run_id`` (same logical trace).
        - A new ``span_id`` (its own execution unit).
        - ``parent_span_id`` set to this context's ``span_id``.
        - Fresh mutable collections (no shared state).
        - Its own ``execution_id`` (new context instance).
        - Inherited scalar fields: ``user_id``, ``chat_id``, ``correlation_id``.

        The child starts in CREATED status. Cancellation and timeout state
        are NOT copied — the existing architecture propagates those signals
        through the authoritative shared context object, not through copies.
        """
        return AgentExecutionContext(
            task=task,
            user_id=self.user_id,
            chat_id=self.chat_id,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            run_id=self.run_id,
            parent_span_id=self.span_id,
        )

    def mark_running(self) -> None:
        self.status = ExecutionStatus.RUNNING
        self._touch()

    def mark_executing(self) -> None:
        self.status = ExecutionStatus.EXECUTING
        self._touch()

    def mark_completed(self) -> None:
        self.status = ExecutionStatus.COMPLETED
        self._touch()
        
    def mark_partial_success(self) -> None:
        self.status = ExecutionStatus.PARTIAL_SUCCESS
        self._touch()

    def mark_failed(self) -> None:
        self.status = ExecutionStatus.FAILED
        self._touch()

    def mark_cancelled(self) -> None:
        self.status = ExecutionStatus.CANCELLED
        self._touch()
        
    def mark_timed_out(self) -> None:
        self.status = ExecutionStatus.TIMED_OUT
        self._touch()

    @property
    def is_cancelled(self) -> bool:
        return self.status == ExecutionStatus.CANCELLED
        
    @property
    def is_timed_out(self) -> bool:
        return self.status == ExecutionStatus.TIMED_OUT

    def publish_artifact(self, artifact: "Artifact") -> None:
        """Publish an immutable structured Artifact to the context."""
        
        normalized_key = artifact.artifact_id.strip()
        
        if not normalized_key:
            raise ValueError("Artifact ID must not be empty.")
            
        if normalized_key in self.artifacts:
            raise ValueError(f"Artifact ID '{normalized_key}' already exists. Artifacts are immutable.")
            
        self.artifacts[normalized_key] = artifact
        self._touch()
        
    def set_artifact(self, key: str, value: Any) -> None:
        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError("Artifact key must not be empty.")

        if normalized_key in self.artifacts:
            # For strict immutability, we must prevent standard `set_artifact` overwrites too
            raise ValueError(f"Artifact key '{normalized_key}' already exists. Artifacts are immutable.")

        self.artifacts[normalized_key] = value
        self._touch()

    def get_artifact(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.artifacts.get(key, default)

    def has_artifact(self, key: str) -> bool:
        return key in self.artifacts

    def remove_artifact(self, key: str) -> Any:
        value = self.artifacts.pop(key)
        self._touch()
        return value

    def publish_agent_output(
        self,
        *,
        agent_id: str,
        output: str | None,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish an agent result without exposing private AgentState."""

        normalized_agent_id = agent_id.strip()

        if not normalized_agent_id:
            raise ValueError("Agent ID must not be empty.")

        agent_output = AgentOutput(
            agent_id=normalized_agent_id,
            output=output,
            success=success,
            metadata=dict(metadata or {}),
        )

        # Preserve existing behavior: latest output for each agent.
        self.agent_outputs[normalized_agent_id] = agent_output

        # Preserve every publication as well.
        self.agent_output_history.append(agent_output)

        self._touch()

    def get_agent_output(
        self,
        agent_id: str,
    ) -> AgentOutput | None:
        """Return the latest published output for an agent."""

        return self.agent_outputs.get(agent_id)

    def publish_message(self, message: AgentMessage) -> None:
        """Publish a unique agent message."""

        if message.execution_id != self.execution_id:
            raise ValueError(
                "Message execution_id does not match this context."
            )

        self.messages[message.message_id] = message
        self._touch()

    def get_message(
        self,
        message_id: str,
    ) -> AgentMessage | None:
        return self.messages.get(message_id)

    def get_messages_for_agent(
        self,
        agent_id: str,
    ) -> list[AgentMessage]:
        return [
            message
            for message in self.messages.values()
            if message.recipient_agent_id == agent_id
        ]

    def add_delegation(
        self,
        delegation: AgentDelegation,
    ) -> None:
        """Register a delegation in this execution context."""

        if delegation.execution_id != self.execution_id:
            raise ValueError(
                "Delegation execution_id does not match this context."
            )

        self.delegations[delegation.delegation_id] = delegation
        self._touch()

    def get_delegation(
        self,
        delegation_id: str,
    ) -> AgentDelegation | None:
        return self.delegations.get(delegation_id)

    def set_metadata(self, key: str, value: Any) -> None:
        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError("Metadata key must not be empty.")

        self.metadata[normalized_key] = value
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)