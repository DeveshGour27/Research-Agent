"""Delegation contracts for supervisor-driven agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class DelegationStatus(str, Enum):
    """Lifecycle state of a delegated task."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class AgentDelegation:
    """A structured task delegated from one agent to another."""

    requesting_agent_id: str
    target_agent_id: str
    task: str
    execution_id: str
    delegation_id: str = field(default_factory=lambda: str(uuid4()))
    status: DelegationStatus = DelegationStatus.CREATED
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def mark_running(self) -> None:
        self.status = DelegationStatus.RUNNING
        self._touch()

    def mark_completed(self, result: str | None) -> None:
        self.status = DelegationStatus.COMPLETED
        self.result = result
        self.error = None
        self._touch()

    def mark_failed(self, error: str) -> None:
        self.status = DelegationStatus.FAILED
        self.error = error
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)