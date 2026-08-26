"""Pydantic API request and response models for Phase 7.1 service boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchJobCreateRequest(BaseModel):
    """Request schema for creating a new research job."""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The research goal or prompt for the agent to execute.",
    )
    chat_id: str | None = Field(
        default=None,
        description="Optional chat session identifier.",
    )

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Research goal must not be empty or whitespace-only.")
        return stripped


class ResearchJobCreateResponse(BaseModel):
    """Response schema returned upon job creation acceptance."""

    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(default="PENDING", description="Initial job status.")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of job creation.")


class ResearchJobStatusResponse(BaseModel):
    """Response schema representing the status and output of a research job."""

    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(..., description="Current job execution status.")
    result: str | None = Field(default=None, description="Final answer or output if completed successfully.")
    execution_time_seconds: float | None = Field(
        default=None, description="Total execution duration in seconds."
    )
    error: str | None = Field(default=None, description="Error message if execution failed.")
    attempt_count: int = Field(default=0, description="Number of execution attempts.")
    created_at: str = Field(..., description="ISO 8601 timestamp of job creation.")
    started_at: str | None = Field(default=None, description="ISO 8601 timestamp when job started.")
    completed_at: str | None = Field(default=None, description="ISO 8601 timestamp when job completed.")


class QueueStatsResponse(BaseModel):
    """Response schema for queue statistics."""

    pending: int = Field(default=0, description="Number of pending jobs.")
    running: int = Field(default=0, description="Number of currently running jobs.")
    completed: int = Field(default=0, description="Number of completed jobs.")
    failed: int = Field(default=0, description="Number of failed jobs.")
    cancelled: int = Field(default=0, description="Number of cancelled jobs.")
    stale: int = Field(default=0, description="Number of stale running jobs.")


class ResearchJobCancelResponse(BaseModel):
    """Response schema returned upon job cancellation request."""

    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(default="CANCELLED", description="Updated job status after cancellation.")


class HealthResponse(BaseModel):
    """Response schema for application liveness health check."""

    status: str = Field(default="pass", description="System liveness status.")


class APIErrorDetail(BaseModel):
    """Structured error detail for deterministic API error responses."""

    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    request_id: str | None = Field(default=None, description="Correlation request ID.")


class APIErrorResponse(BaseModel):
    """Wrapper for structured API errors."""

    error: APIErrorDetail


class ResearchResultResponse(BaseModel):
    """Response schema explicitly for job results."""

    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(..., description="Current job execution status.")
    result: str | None = Field(default=None, description="Final answer or output if completed successfully.")
    error: str | None = Field(default=None, description="Error message if execution failed.")
    completed_at: str | None = Field(default=None, description="ISO 8601 timestamp when job completed.")


class ResearchJobEvent(BaseModel):
    """Schema for a deterministic, frontend-safe job event."""

    event_id: str = Field(..., description="Globally unique identifier for the event (e.g., jobstep:<id>).")
    job_id: str = Field(..., description="Job identifier.")
    event_type: str = Field(..., description="Type of event (e.g., JOB_CREATED, JOB_STARTED, TOOL_CALL, HITL_REQUESTED, JOB_COMPLETED).")
    timestamp: str = Field(..., description="ISO 8601 timestamp.")
    sequence: int = Field(..., description="Deterministic ordering sequence number.")
    source: str = Field(..., description="Source of the event (job, jobstep, hitl).")
    payload: dict[str, Any] = Field(default_factory=dict, description="Safe event details.")


class ResearchJobEventsResponse(BaseModel):
    """Response schema for job events list."""

    job_id: str = Field(..., description="Job identifier.")
    events: list[ResearchJobEvent] = Field(..., description="Deterministically ordered list of events.")


class HITLRequestResponse(BaseModel):
    """Response schema for a single HITL request."""
    
    request_id: str = Field(..., description="Unique request identifier.")
    request_type: str = Field(..., description="Type of request (TOOL or PLAN).")
    component_name: str = Field(..., description="Name of the component requiring approval.")
    status: str = Field(..., description="Current status of the request (PENDING, APPROVED, REJECTED, EXPIRED).")
    created_at: str = Field(..., description="ISO 8601 timestamp.")
    expires_at: str | None = Field(default=None, description="ISO 8601 timestamp.")
    decided_at: str | None = Field(default=None, description="ISO 8601 timestamp.")
    decided_by: str | None = Field(default=None, description="User ID of the decider.")
    decision_reason: str | None = Field(default=None, description="Reason provided for the decision.")


class HITLRequestsListResponse(BaseModel):
    """Response schema for listing HITL requests."""
    
    job_id: str = Field(..., description="Job identifier.")
    hitl_requests: list[HITLRequestResponse] = Field(..., description="List of HITL requests for the job.")

