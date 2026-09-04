"""Event adapter for translating database records into deterministic API events."""

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models import Job, JobStep, HITLRequest
from app.api.models import ResearchJobEvent


def get_job_events(db: Session, job: Job) -> list[ResearchJobEvent]:
    """
    Construct a deterministic timeline of events for a given job,
    fetching from Job status changes, JobStep history, and HITLRequests.
    """
    events: list[ResearchJobEvent] = []

    # 1. Job Created Event
    events.append(ResearchJobEvent(
        event_id=f"job:{job.job_id}:created",
        job_id=job.job_id,
        event_type="JOB_CREATED",
        timestamp=job.created_at.isoformat(),
        sequence=0,
        source="job",
        payload={"goal": job.goal}
    ))

    # 2. Job Started Event
    if job.started_at:
        events.append(ResearchJobEvent(
            event_id=f"job:{job.job_id}:started",
            job_id=job.job_id,
            event_type="JOB_STARTED",
            timestamp=job.started_at.isoformat(),
            sequence=1,
            source="job",
            payload={}
        ))

    # 3. JobSteps -> Events
    # JobStep status could be PENDING, COMPLETED, FAILED.
    # We will map task_type to event_type loosely, e.g. task_type is TOOL_CALL, PLAN_CREATED, etc.
    # To keep it deterministic, we order by created_at then step_id.
    stmt = select(JobStep).where(JobStep.job_id == job.job_id).order_by(JobStep.created_at, JobStep.step_id)
    steps = db.execute(stmt).scalars().all()

    for i, step in enumerate(steps, start=1000):
        # We can map step.task_type directly or use it as event_type if it matches the known ones.
        event_type = step.task_type.upper()
        
        events.append(ResearchJobEvent(
            event_id=f"jobstep:{step.step_id}",
            job_id=job.job_id,
            event_type=event_type,
            timestamp=step.created_at.isoformat(),
            sequence=i,
            source="jobstep",
            payload={
                "status": step.status,
                "output": step.output,
                "execution_time_seconds": step.execution_time_seconds,
            }
        ))

    # 4. HITL Requests -> Events
    stmt_hitl = select(HITLRequest).where(HITLRequest.job_id == job.job_id).order_by(HITLRequest.created_at, HITLRequest.request_id)
    hitl_reqs = db.execute(stmt_hitl).scalars().all()

    for i, req in enumerate(hitl_reqs, start=2000):
        events.append(ResearchJobEvent(
            event_id=f"hitl:{req.request_id}:requested",
            job_id=job.job_id,
            event_type="HITL_REQUESTED",
            timestamp=req.created_at.isoformat(),
            sequence=i,
            source="hitl",
            payload={
                "request_type": req.request_type,
                "component_name": req.component_name,
                "status": req.status,
            }
        ))
        
        # If decided, add a decision event
        if req.decided_at:
            decision_type = "HITL_APPROVED" if req.status == "APPROVED" else "HITL_REJECTED"
            events.append(ResearchJobEvent(
                event_id=f"hitl:{req.request_id}:decided",
                job_id=job.job_id,
                event_type=decision_type,
                timestamp=req.decided_at.isoformat(),
                sequence=i + 10000, # Push decisions later in sequence baseline
                source="hitl",
                payload={
                    "decided_by": req.decided_by,
                    "reason": req.decision_reason,
                }
            ))

    # 5. Terminal Job Events
    if job.completed_at:
        if job.status == "COMPLETED":
            events.append(ResearchJobEvent(
                event_id=f"job:{job.job_id}:completed",
                job_id=job.job_id,
                event_type="JOB_COMPLETED",
                timestamp=job.completed_at.isoformat(),
                sequence=90000,
                source="job",
                payload={"output": job.result}
            ))
        elif job.status == "FAILED":
            events.append(ResearchJobEvent(
                event_id=f"job:{job.job_id}:failed",
                job_id=job.job_id,
                event_type="JOB_FAILED",
                timestamp=job.completed_at.isoformat(),
                sequence=90001,
                source="job",
                payload={"error": job.error_message}
            ))
        elif job.status == "CANCELLED":
            events.append(ResearchJobEvent(
                event_id=f"job:{job.job_id}:cancelled",
                job_id=job.job_id,
                event_type="JOB_CANCELLED",
                timestamp=job.completed_at.isoformat(),
                sequence=90002,
                source="job",
                payload={}
            ))
            
    # Sort events by timestamp and then by sequence
    events.sort(key=lambda e: (e.timestamp, e.sequence))
    
    # Re-assign sequence to be strictly monotonous after sort
    for idx, e in enumerate(events):
        e.sequence = idx
        
    return events
