"""Health, readiness, and metrics endpoints for Phase 7.5.

All routes in this module are **public** — they do NOT require ``X-API-Key``
authentication.  They are designed for orchestrators, load-balancers, and
monitoring systems.

Security invariants:
    - No database URLs, credentials, API keys, stack traces, filesystem
      paths, or raw exception details are ever exposed in responses.
    - The ``/metrics`` snapshot is sourced from the existing deterministic
      ``MetricsRegistry`` and contains only counters and latencies — never
      raw user goals, LLM responses, or authorization headers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.database import engine
from app.observability.registry_instance import get_metrics_snapshot

health_router = APIRouter(tags=["System"])


# ------------------------------------------------------------------
# GET /health  — Liveness
# ------------------------------------------------------------------
@health_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Liveness Health Check",
)
def get_health() -> dict[str, str]:
    """Lightweight liveness probe.  Always returns 200 if the process is up."""
    return {"status": "pass"}


# ------------------------------------------------------------------
# GET /ready   — Readiness
# ------------------------------------------------------------------
@health_router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
)
def get_ready(request: Request) -> JSONResponse:
    """Deep readiness check: database connectivity + job-manager state.

    Returns 200 when all subsystems are healthy, 503 otherwise.
    Never leaks credentials, connection strings, or stack traces.
    """
    checks: dict[str, str] = {}
    overall_pass = True

    # --- Database ---
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "connected"
    except Exception:
        checks["database"] = "unavailable"
        overall_pass = False

    # --- AsyncJobManager ---
    job_manager = getattr(request.app.state, "job_manager", None)
    if job_manager is not None and getattr(job_manager, "is_ready", False):
        checks["job_manager"] = "ready"
    else:
        checks["job_manager"] = "unavailable"
        overall_pass = False

    status_code = status.HTTP_200_OK if overall_pass else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        content={"status": "pass" if overall_pass else "fail", **checks},
        status_code=status_code,
    )


# ------------------------------------------------------------------
# GET /metrics — Operational Metrics
# ------------------------------------------------------------------
@health_router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Operational Metrics",
)
def get_metrics() -> JSONResponse:
    """Expose the existing deterministic ``MetricsRegistry`` snapshot as JSON.

    Does NOT add Prometheus dependencies or invent new counters.
    """
    snapshot = get_metrics_snapshot()
    return JSONResponse(content=snapshot)
