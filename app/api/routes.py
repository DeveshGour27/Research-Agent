"""API routes for Phase 7.1 service boundary (hardened in Phase 7.6)."""

from __future__ import annotations

from uuid import uuid4
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.models import (
    ResearchJobCancelResponse,
    ResearchJobCreateRequest,
    ResearchJobCreateResponse,
    ResearchJobStatusResponse,
    QueueStatsResponse,
)
from app.db.database import get_db
from app.db.models import User
from app.db.repository import SQLJobRepository
from app.api.auth import get_current_user
from app.exceptions import InvalidStateTransitionError
from app.config import settings

router = APIRouter()

_MAX_IDEMPOTENCY_KEY_LENGTH = 256


def _error_response(
    request: Request,
    code: str,
    message: str,
    status_code: int,
) -> JSONResponse:
    """Build a deterministic API error response with request-ID."""
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


@router.post(
    "/api/v1/research/jobs",
    response_model=ResearchJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Research Job",
    tags=["Research Jobs"],
)
def create_research_job(
    request_data: ResearchJobCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    x_idempotency_key: str | None = Header(None, alias="X-Idempotency-Key"),
) -> ResearchJobCreateResponse | JSONResponse:
    """Accept a new research goal request and return a job identifier.

    Supports idempotent creation via the ``X-Idempotency-Key`` header.
    """
    # --- Rate limiting ---
    rate_limiter = getattr(getattr(request.app, "state", None), "rate_limiter", None)
    if rate_limiter is not None:
        allowed, retry_after = rate_limiter.is_allowed(user.user_id)
        if not allowed:
            return _error_response(
                request,
                code="RATE_LIMIT_EXCEEDED",
                message="Too many requests. Please retry later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

    # --- Validate idempotency key ---
    if x_idempotency_key is not None:
        x_idempotency_key = x_idempotency_key.strip()
        if not x_idempotency_key:
            return _error_response(
                request,
                code="INVALID_IDEMPOTENCY_KEY",
                message="X-Idempotency-Key must not be empty or whitespace.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if len(x_idempotency_key) > _MAX_IDEMPOTENCY_KEY_LENGTH:
            return _error_response(
                request,
                code="INVALID_IDEMPOTENCY_KEY",
                message=f"X-Idempotency-Key must not exceed {_MAX_IDEMPOTENCY_KEY_LENGTH} characters.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    repo = SQLJobRepository(db)

    # --- Idempotency check ---
    if x_idempotency_key:
        existing = repo.get_job_by_idempotency_key(user.user_id, x_idempotency_key)
        if existing:
            return ResearchJobCreateResponse(
                job_id=existing.job_id,
                status=existing.status,
            )

    job_id = f"job_{uuid4().hex[:12]}"

    # Persist the job
    try:
        job = repo.create_job(
            job_id=job_id,
            user_id=user.user_id,
            goal=request_data.goal,
            idempotency_key=x_idempotency_key,
        )
    except IntegrityError:
        # Concurrent duplicate — the unique constraint caught it
        db.rollback()
        if x_idempotency_key:
            existing = repo.get_job_by_idempotency_key(user.user_id, x_idempotency_key)
            if existing:
                return ResearchJobCreateResponse(
                    job_id=existing.job_id,
                    status=existing.status,
                )
        return _error_response(
            request,
            code="DUPLICATE_REQUEST",
            message="A conflicting request was already processed.",
            status_code=status.HTTP_409_CONFLICT,
        )

    # Submit to the background execution manager
    job_manager = request.app.state.job_manager
    job_manager.submit_job(job_id=job.job_id, user_id=user.user_id, goal=request_data.goal)

    return ResearchJobCreateResponse(
        job_id=job.job_id,
        status=job.status,
    )


@router.get(
    "/api/v1/research/jobs/stats",
    response_model=QueueStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Queue Statistics",
    tags=["Research Jobs"],
)
def get_queue_stats(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueueStatsResponse:
    """Retrieve job queue statistics scoped to the authenticated user."""
    repo = SQLJobRepository(db)
    stale_threshold = settings.job_stale_after_seconds
    stats = repo.get_queue_stats(user.user_id, stale_threshold)
    
    return QueueStatsResponse(**stats)


@router.get(
    "/api/v1/research/jobs/{job_id}",
    response_model=ResearchJobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Research Job Status",
    tags=["Research Jobs"],
)
def get_research_job_status(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResearchJobStatusResponse | JSONResponse:
    """Retrieve the status and results of a research job by ID."""
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)

    if not job:
        # Deliberate non-enumerating 404 for cross-user isolation
        return _error_response(
            request,
            code="JOB_NOT_FOUND",
            message="Research job was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    execution_time_seconds = None
    if job.started_at and job.completed_at:
        execution_time_seconds = (job.completed_at - job.started_at).total_seconds()

    return ResearchJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        execution_time_seconds=execution_time_seconds,
        error=job.error_message,
        attempt_count=job.attempt_count,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.post(
    "/api/v1/research/jobs/{job_id}/cancel",
    response_model=ResearchJobCancelResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Research Job",
    tags=["Research Jobs"],
)
async def cancel_research_job(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResearchJobCancelResponse | JSONResponse:
    """Request cancellation of an active or pending research job."""
    # --- Rate limiting ---
    rate_limiter = getattr(getattr(request.app, "state", None), "rate_limiter", None)
    if rate_limiter is not None:
        allowed, retry_after = rate_limiter.is_allowed(user.user_id)
        if not allowed:
            return _error_response(
                request,
                code="RATE_LIMIT_EXCEEDED",
                message="Too many requests. Please retry later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

    job_manager = request.app.state.job_manager
    try:
        await job_manager.cancel_job(job_id=job_id, user_id=user.user_id)
    except ValueError:
        return _error_response(
            request,
            code="JOB_NOT_FOUND",
            message="Research job was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # Fetch updated job to return status
    db.expire_all()
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)

    return ResearchJobCancelResponse(
        job_id=job.job_id,
        status=job.status,
    )
