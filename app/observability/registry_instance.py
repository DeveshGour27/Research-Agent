"""Canonical singleton MetricsRegistry instance for the application.

All components that need to record or query metrics should import
``metrics_registry`` from this module rather than constructing their own
``MetricsRegistry`` instance.  This guarantees a single, process-wide
registry whose snapshot is exposed by the ``/metrics`` health endpoint.
"""

from __future__ import annotations

from app.observability.metrics import MetricsRegistry

# Single application-wide instance.
metrics_registry = MetricsRegistry()


def get_metrics_snapshot() -> dict:
    """Return the current deterministic metrics snapshot.

    Convenience wrapper used by the ``/metrics`` endpoint.
    """
    return metrics_registry.snapshot()
