"""Execution policy for plan execution bounds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Deterministic bounds for plan execution.

    Controls execution limits and replanning permissions.
    Does NOT contain replanning logic itself.
    """

    max_steps: int = 50
    max_failures: int = 3
    max_replans: int = 2
    allow_replanning: bool = True
