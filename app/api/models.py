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
