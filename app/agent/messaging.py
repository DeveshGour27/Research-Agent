"""Typed agent-to-agent messaging contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A structured message exchanged between agents."""

    sender_agent_id: str
    recipient_agent_id: str
    content: str
    execution_id: str
    message_type: str = "information"
    message_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.sender_agent_id.strip():
            raise ValueError("sender_agent_id must not be empty.")

        if not self.recipient_agent_id.strip():
            raise ValueError("recipient_agent_id must not be empty.")

        if not self.execution_id.strip():
            raise ValueError("execution_id must not be empty.")

        if not self.content.strip():
            raise ValueError("content must not be empty.")

        if not self.message_type.strip():
            raise ValueError("message_type must not be empty.")


def create_agent_message(
    *,
    sender_agent_id: str,
    recipient_agent_id: str,
    execution_id: str,
    content: str,
    message_type: str = "information",
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentMessage:
    """Create a validated agent message."""

    return AgentMessage(
        sender_agent_id=sender_agent_id,
        recipient_agent_id=recipient_agent_id,
        execution_id=execution_id,
        content=content,
        message_type=message_type,
        correlation_id=correlation_id,
        metadata=dict(metadata or {}),
    )