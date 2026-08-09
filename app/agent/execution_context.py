"""Shared execution context for multi-agent task coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from app.agent.delegation import AgentDelegation
from app.agent.messaging import AgentMessage


class ExecutionStatus(str, Enum):
    """Lifecycle status of a shared agent execution."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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

    def mark_running(self) -> None:
        self.status = ExecutionStatus.RUNNING
        self._touch()

    def mark_completed(self) -> None:
        self.status = ExecutionStatus.COMPLETED
        self._touch()

    def mark_failed(self) -> None:
        self.status = ExecutionStatus.FAILED
        self._touch()

    def set_artifact(self, key: str, value: Any) -> None:
        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError("Artifact key must not be empty.")

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