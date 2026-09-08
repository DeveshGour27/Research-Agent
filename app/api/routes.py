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
    ResearchResultResponse,
    ResearchJobEventsResponse,
)
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
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
    "/api/v1/research",
    response_model=ResearchJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Research Job",
    tags=["Research Jobs"],
)
@router.post(
    "/api/v1/research/jobs",
    response_model=ResearchJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Research Job (Deprecated Alias)",
    tags=["Research Jobs"],
    deprecated=True,
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
    "/api/v1/jobs/stats",
    response_model=QueueStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Queue Statistics",
    tags=["Research Jobs"],
)
@router.get(
    "/api/v1/research/jobs/stats",
    response_model=QueueStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Queue Statistics (Deprecated Alias)",
    tags=["Research Jobs"],
    deprecated=True,
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
    "/api/v1/jobs/{job_id}",
    response_model=ResearchJobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Research Job Status",
    tags=["Research Jobs"],
)
@router.get(
    "/api/v1/research/jobs/{job_id}",
    response_model=ResearchJobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Research Job Status (Deprecated Alias)",
    tags=["Research Jobs"],
    deprecated=True,
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
    "/api/v1/research/{job_id}/cancel",
    response_model=ResearchJobCancelResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Research Job",
    tags=["Research Jobs"],
)
@router.post(
    "/api/v1/research/jobs/{job_id}/cancel",
    response_model=ResearchJobCancelResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Research Job (Deprecated Alias)",
    tags=["Research Jobs"],
    deprecated=True,
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


@router.get(
    "/api/v1/research/{job_id}/hitl",
    status_code=status.HTTP_200_OK,
    summary="Get HITL Requests for Job",
    tags=["Research Jobs", "HITL"],
)
@router.get(
    "/api/v1/research/jobs/{job_id}/hitl",
    status_code=status.HTTP_200_OK,
    summary="Get HITL Requests for Job (Deprecated Alias)",
    tags=["Research Jobs", "HITL"],
    deprecated=True,
)
def get_hitl_requests(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retrieve HITL requests for a specific job."""
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)
    if not job:
        return _error_response(request, "JOB_NOT_FOUND", "Research job was not found.", status.HTTP_404_NOT_FOUND)

    from app.db.models import HITLRequest
    from sqlalchemy import select
    stmt = select(HITLRequest).where(HITLRequest.job_id == job_id).order_by(HITLRequest.created_at.desc())
    requests = db.execute(stmt).scalars().all()

    return {
        "job_id": job_id,
        "hitl_requests": [
            {
                "request_id": r.request_id,
                "request_type": r.request_type,
                "component_name": r.component_name,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "decided_at": r.decided_at.isoformat() if r.decided_at else None,
                "decided_by": r.decided_by,
                "decision_reason": r.decision_reason,
            }
            for r in requests
        ]
    }


@router.post(
    "/api/v1/research/{job_id}/hitl/{request_id}/approve",
    status_code=status.HTTP_200_OK,
    summary="Approve HITL Request",
    tags=["Research Jobs", "HITL"],
)
@router.post(
    "/api/v1/research/jobs/{job_id}/hitl/{request_id}/approve",
    status_code=status.HTTP_200_OK,
    summary="Approve HITL Request (Deprecated Alias)",
    tags=["Research Jobs", "HITL"],
    deprecated=True,
)
def approve_hitl_request(
    job_id: str,
    request_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Approve a HITL request."""
    hitl_service = getattr(getattr(request.app, "state", None), "hitl_service", None)
    if not hitl_service:
        return _error_response(request, "INTERNAL_ERROR", "HITL service not configured.", status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Note: HITL service uses its own session factory. We pass user ID as decider.
    # The service internally validates if the job is waiting and request is pending.
    
    # First, verify job ownership
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)
    if not job:
        return _error_response(request, "JOB_NOT_FOUND", "Research job was not found.", status.HTTP_404_NOT_FOUND)
        
    success = hitl_service.approve_request(request_id, job_id, decided_by=user.user_id)
    if not success:
        return _error_response(request, "INVALID_STATE", "Request cannot be approved (already decided or job not waiting).", status.HTTP_400_BAD_REQUEST)

    return {"status": "success", "message": "HITL request approved."}


@router.post(
    "/api/v1/research/{job_id}/hitl/{request_id}/reject",
    status_code=status.HTTP_200_OK,
    summary="Reject HITL Request",
    tags=["Research Jobs", "HITL"],
)
@router.post(
    "/api/v1/research/jobs/{job_id}/hitl/{request_id}/reject",
    status_code=status.HTTP_200_OK,
    summary="Reject HITL Request (Deprecated Alias)",
    tags=["Research Jobs", "HITL"],
    deprecated=True,
)
def reject_hitl_request(
    job_id: str,
    request_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Reject a HITL request."""
    hitl_service = getattr(getattr(request.app, "state", None), "hitl_service", None)
    if not hitl_service:
        return _error_response(request, "INTERNAL_ERROR", "HITL service not configured.", status.HTTP_500_INTERNAL_SERVER_ERROR)

    # First, verify job ownership
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)
    if not job:
        return _error_response(request, "JOB_NOT_FOUND", "Research job was not found.", status.HTTP_404_NOT_FOUND)
        
    success = hitl_service.reject_request(request_id, job_id, decided_by=user.user_id)
    if not success:
        return _error_response(request, "INVALID_STATE", "Request cannot be rejected (already decided or job not waiting).", status.HTTP_400_BAD_REQUEST)

    return {"status": "success", "message": "HITL request rejected."}


@router.get(
    "/api/v1/research/{job_id}/result",
    response_model=ResearchResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Research Job Result",
    tags=["Research Jobs"],
)
def get_research_job_result(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResearchResultResponse | JSONResponse:
    """Retrieve the final result of a completed research job."""
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)
    if not job:
        return _error_response(request, "JOB_NOT_FOUND", "Research job was not found.", status.HTTP_404_NOT_FOUND)
        
    if job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
        return _error_response(
            request, 
            "RESULT_UNAVAILABLE", 
            f"Result is not yet available. Current status: {job.status}", 
            status.HTTP_409_CONFLICT
        )
        
    return ResearchResultResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result if job.status == "COMPLETED" else None,
        error=job.error_message if job.status == "FAILED" else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get(
    "/api/v1/jobs/{job_id}/events",
    response_model=ResearchJobEventsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Job Events",
    tags=["Research Jobs", "Events"],
)
def get_research_job_events(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResearchJobEventsResponse | JSONResponse:
    """Retrieve the deterministic event timeline for a research job."""
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)
    if not job:
        return _error_response(request, "JOB_NOT_FOUND", "Research job was not found.", status.HTTP_404_NOT_FOUND)
        
    from app.api.events import get_job_events
    events = get_job_events(db, job)
    
    return ResearchJobEventsResponse(job_id=job.job_id, events=events)


@router.get(
    "/api/v1/jobs/{job_id}/events/stream",
    summary="Stream Job Events (SSE)",
    tags=["Research Jobs", "Events"],
)
async def stream_research_job_events(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
):
    """Stream events for a research job using Server-Sent Events (SSE)."""
    # Verify ownership immediately
    repo = SQLJobRepository(db)
    job = repo.get_job(job_id=job_id, user_id=user.user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Research job was not found.")

    async def event_generator():
        from app.api.events import get_job_events
        import json
        import time
        
        last_yielded_seq = -1
        stream_started_at = time.monotonic()
        max_sse_duration = 900  # 15 minutes maximum lifetime for an active SSE stream
        
        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                break

            # Enforce max connection lifetime
            if time.monotonic() - stream_started_at > max_sse_duration:
                yield "event: timeout\ndata: {\"error\": \"SSE stream reached maximum duration limit.\"}\n\n"
                break
                
            # Fetch fresh state
            # db is synchronous, but we can call it in this polling generator safely enough for Phase 12 requirements
            db.expire_all()
            current_job = repo.get_job(job_id=job_id, user_id=user.user_id)
            if not current_job:
                break
                
            all_events = get_job_events(db, current_job)
            
            # If Last-Event-ID was provided, find its sequence to resume properly
            start_seq = 0
            if last_event_id and last_yielded_seq == -1:
                found = False
                for e in all_events:
                    if e.event_id == last_event_id:
                        start_seq = e.sequence + 1
                        found = True
                        break
                if not found:
                    # Deterministic fallback: start from 0 if we can't find it
                    start_seq = 0
                last_yielded_seq = start_seq - 1
            
            for ev in all_events:
                if ev.sequence > last_yielded_seq:
                    # Yield SSE formatted string
                    yield f"id: {ev.event_id}\n"
                    yield f"event: {ev.event_type}\n"
                    yield f"data: {json.dumps(ev.model_dump())}\n\n"
                    last_yielded_seq = ev.sequence
                    
            if current_job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                # Job is terminal. Allow buffers to flush to client before closing socket.
                await asyncio.sleep(0.5)
                break
                
            # Send an SSE comment as a heartbeat to keep the connection alive
            # through browsers, proxies and load balancers. Clients ignore comments.
            yield ": heartbeat\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

