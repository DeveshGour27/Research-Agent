"""Lightweight tracer for span creation and propagation.

Responsibilities:
    - Create root and child spans from an AgentExecutionContext.
    - Extract tracing identifiers for use in observability events.
    - Provide a safe, isolated abstraction that Step 3 can use
      to instrument executor, supervisor, and collaboration.

This module does NOT:
    - Execute agents.
    - Perform retries or routing.
    - Modify execution status.
    - Become a required dependency for agent execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SpanInfo:
    """Immutable snapshot of tracing identifiers for a single span.

    Designed to be extracted from an AgentExecutionContext and passed
    to observability event constructors without coupling events to
    the mutable context object.
    """

    trace_id: str
    run_id: str
    span_id: str
    parent_span_id: str | None


def extract_span_info(context: Any) -> SpanInfo | None:
    """Safely extract tracing identifiers from an AgentExecutionContext.

    Returns ``None`` if the context is ``None`` or lacks the expected
    tracing fields. Never raises — observability failures must not
    affect agent execution.

    Args:
        context: An ``AgentExecutionContext`` or compatible object.

    Returns:
        A ``SpanInfo`` snapshot, or ``None`` on failure.
    """
    try:
        if context is None:
            return None

        trace_id = getattr(context, "trace_id", None)
        run_id = getattr(context, "run_id", None)
        span_id = getattr(context, "span_id", None)
        parent_span_id = getattr(context, "parent_span_id", None)

        if trace_id is None or run_id is None or span_id is None:
            return None

        return SpanInfo(
            trace_id=str(trace_id),
            run_id=str(run_id),
            span_id=str(span_id),
            parent_span_id=str(parent_span_id) if parent_span_id is not None else None,
        )
    except Exception:
        # Observability must NEVER become a control-flow dependency.
        return None
