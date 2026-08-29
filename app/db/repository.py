"""Repository layer for interacting with the SQL persistence store."""

from __future__ import annotations

from typing import Sequence
import datetime

from sqlalchemy import select, update, func
from sqlalchemy.orm import Session
from app.exceptions import InvalidStateTransitionError

from app.db.models import ApiKey, Job, JobStep, User


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# Valid job state transitions. Terminal states have no outgoing edges.
VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"RUNNING", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED", "WAITING_FOR_HUMAN"},
    "WAITING_FOR_HUMAN": {"PENDING", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

TERMINAL_STATES: set[str] = {"COMPLETED", "FAILED", "CANCELLED"}

class SQLJobRepository:
    """Provides deterministic CRUD operations for jobs and users."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ----------------------------------------------------------------------
    # User & API Key Operations
    # ----------------------------------------------------------------------
    def create_user(self, email: str | None = None, user_id: str | None = None) -> User:
        """Create and return a new User."""
        user = User(email=email)
        if user_id:
            user.user_id = user_id
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_user(self, user_id: str) -> User | None:
        """Retrieve a User by their ID."""
        return self.db.get(User, user_id)

    def create_api_key(self, user_id: str, key_hash: str) -> ApiKey:
        """Create a new API key record for a user."""
        api_key = ApiKey(key_hash=key_hash, user_id=user_id)
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        return api_key

    def get_api_key(self, key_hash: str) -> ApiKey | None:
        """Retrieve an ApiKey by its hash."""
        return self.db.get(ApiKey, key_hash)

    # ----------------------------------------------------------------------
    # Job Operations
    # ----------------------------------------------------------------------
    def create_job(
        self,
        job_id: str,
        user_id: str,
        goal: str,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Job:
        """Create and persist a new Job."""
        job = Job(
            job_id=job_id,
            user_id=user_id,
            goal=goal,
            status="PENDING",
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Job | None:
        """Look up an existing job by user-scoped idempotency key."""
        stmt = select(Job).where(
            Job.user_id == user_id,
            Job.idempotency_key == idempotency_key,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_job(self, job_id: str, user_id: str) -> Job | None:
        """Retrieve a Job by ID, enforcing user ownership."""
        stmt = select(Job).where(Job.job_id == job_id, Job.user_id == user_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def claim_job(self, job_id: str, worker_id: str) -> bool:
        """Atomically claim a PENDING job."""
        now = _utc_now()
        stmt = (
            update(Job)
            .where(
                Job.job_id == job_id,
                Job.status == "PENDING",
                Job.worker_id.is_(None)
            )
            .values(
                status="RUNNING",
                worker_id=worker_id,
                started_at=now,
                heartbeat_at=now,
                attempt_count=Job.attempt_count + 1,
            )
        )
        res = self.db.execute(stmt)
        self.db.commit()
        return res.rowcount > 0

    def heartbeat_job(self, job_id: str, worker_id: str) -> bool:
        """Atomically heartbeat a RUNNING job for a specific worker."""
        stmt = (
            update(Job)
            .where(
                Job.job_id == job_id,
                Job.status == "RUNNING",
                Job.worker_id == worker_id,
            )
            .values(heartbeat_at=_utc_now())
        )
        res = self.db.execute(stmt)
        self.db.commit()
        return res.rowcount > 0

    def get_queue_stats(
        self, user_id: str, stale_threshold_seconds: int
    ) -> dict[str, int]:
        """Calculate queue statistics strictly scoped by user_id."""
        # 1. Base counts by status
        stmt_counts = (
            select(Job.status, func.count(Job.job_id))
            .where(Job.user_id == user_id)
            .group_by(Job.status)
        )
        res_counts = self.db.execute(stmt_counts).all()
        
        stats = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "stale": 0,
        }
        
        for status, count in res_counts:
            s = status.lower()
            if s in stats:
                stats[s] = count
                
        # 2. Stale count (consistent with recover_stale_jobs)
        stale_cutoff = _utc_now() - datetime.timedelta(seconds=stale_threshold_seconds)
        stmt_stale = (
            select(func.count(Job.job_id))
            .where(
                Job.user_id == user_id,
                Job.status == "RUNNING",
                Job.heartbeat_at < stale_cutoff,
            )
        )
        stale_count = self.db.execute(stmt_stale).scalar() or 0
        stats["stale"] = stale_count
        
        return stats

    def recover_stale_jobs(
        self, stale_threshold_seconds: int, max_attempts: int
    ) -> list[str]:
        """Atomically recover stale RUNNING jobs by returning them to PENDING or FAILED if exhausted."""
        stale_cutoff = _utc_now() - datetime.timedelta(seconds=stale_threshold_seconds)
        
        # 1. Recoverable jobs (attempt_count < max_attempts)
        stmt_recoverable = (
            update(Job)
            .where(
                Job.status == "RUNNING",
                Job.heartbeat_at < stale_cutoff,
                Job.attempt_count < max_attempts,
            )
            .values(
                status="PENDING",
                worker_id=None,
                heartbeat_at=None,
            )
            .returning(Job.job_id)
        )
        res_recoverable = self.db.execute(stmt_recoverable)
        recovered_ids = [row[0] for row in res_recoverable.all()]

        # 2. Poison-pill / exhausted jobs (attempt_count >= max_attempts)
        error_msg = f"Job exceeded maximum execution attempts ({max_attempts})."
        stmt_exhausted = (
            update(Job)
            .where(
                Job.status == "RUNNING",
                Job.heartbeat_at < stale_cutoff,
                Job.attempt_count >= max_attempts,
            )
            .values(
                status="FAILED",
                worker_id=None,
                heartbeat_at=None,
                error_message=error_msg,
                completed_at=_utc_now(),
            )
        )
        self.db.execute(stmt_exhausted)
        
        self.db.commit()
        return recovered_ids

    def poll_pending_jobs(self, limit: int) -> list[str]:
        """Fetch a batch of PENDING jobs that haven't been claimed."""
        stmt = (
            select(Job.job_id)
            .where(
                Job.status == "PENDING",
                Job.worker_id.is_(None),
            )
            .limit(limit)
        )
        res = self.db.execute(stmt)
        return list(res.scalars().all())

    def update_job_status(
        self,
        job_id: str,
        user_id: str,
        status: str,
        *,
        expected_states: set[str] | None = None,
        worker_id: str | None = None,
    ) -> Job | None:
        """Atomically update job status with state-machine validation.

        Uses a conditional UPDATE to avoid check-then-act races.
        Raises ``InvalidStateTransitionError`` if the transition is not valid.
        Returns ``None`` if the job is not found or the expected state no longer matches.
        """
        # Determine which current states allow this transition
        if expected_states is None:
            expected_states = [
                src for src, targets in VALID_TRANSITIONS.items() if status in targets
            ]

        if not expected_states:
            raise InvalidStateTransitionError(
                f"No valid source state for target '{status}'",
                current_state="UNKNOWN",
                target_state=status,
                job_id=job_id,
            )

        completed_at = _utc_now() if status in TERMINAL_STATES else None

        where_conditions = [
            Job.job_id == job_id,
            Job.user_id == user_id,
            Job.status.in_(expected_states),
        ]
        if worker_id is not None:
            where_conditions.append(Job.worker_id == worker_id)

        stmt = (
            update(Job)
            .where(*where_conditions)
            .values(status=status, completed_at=completed_at)
        )
        result = self.db.execute(stmt)
        self.db.commit()

        if result.rowcount == 0:
            # Either not found or state already changed
            job = self.get_job(job_id, user_id)
            if job and job.status not in expected_states:
                raise InvalidStateTransitionError(
                    f"Cannot transition from '{job.status}' to '{status}'",
                    current_state=job.status,
                    target_state=status,
                    job_id=job_id,
                )
            return None

        return self.get_job(job_id, user_id)

    def update_job_result(
        self,
        job_id: str,
        user_id: str,
        status: str,
        result: str | None = None,
        error_message: str | None = None,
        *,
        worker_id: str | None = None,
    ) -> Job | None:
        """Atomically update a job's result and status with state-machine validation.

        Uses a conditional UPDATE to protect terminal states from being overwritten.
        """
        expected_states = [
            src for src, targets in VALID_TRANSITIONS.items() if status in targets
        ]

        values: dict = {
            "status": status,
            "result": result,
            "error_message": error_message,
            "completed_at": _utc_now(),
        }

        where_conditions = [
            Job.job_id == job_id,
            Job.user_id == user_id,
            Job.status.in_(expected_states),
        ]
        if worker_id is not None:
            where_conditions.append(Job.worker_id == worker_id)

        stmt = (
            update(Job)
            .where(*where_conditions)
            .values(**values)
        )
        res = self.db.execute(stmt)
        self.db.commit()

        if res.rowcount == 0:
            return None

        return self.get_job(job_id, user_id)

    # ----------------------------------------------------------------------
    # Job Step Operations
    # ----------------------------------------------------------------------
    def create_job_step(
        self,
        step_id: str,
        job_id: str,
        task_type: str,
    ) -> JobStep:
        """Create and persist a new Job Step."""
        step = JobStep(
            step_id=step_id,
            job_id=job_id,
            task_type=task_type,
            status="PENDING",
        )
        self.db.add(step)
        self.db.commit()
        self.db.refresh(step)
        return step

    def update_job_step(
        self,
        step_id: str,
        job_id: str,
        status: str,
        output: str | None = None,
        execution_time_seconds: float | None = None,
    ) -> JobStep | None:
        """Update a job step's status and output."""
        stmt = select(JobStep).where(JobStep.step_id == step_id, JobStep.job_id == job_id)
        step = self.db.execute(stmt).scalar_one_or_none()
        if not step:
            return None

        step.status = status
        if output is not None:
            step.output = output
        if execution_time_seconds is not None:
            step.execution_time_seconds = execution_time_seconds

        self.db.commit()
        self.db.refresh(step)
        return step

    def get_job_steps(self, job_id: str, user_id: str) -> Sequence[JobStep]:
        """Retrieve all steps for a job, verifying user ownership first."""
        job = self.get_job(job_id, user_id)
        if not job:
            return []
            
        stmt = select(JobStep).where(JobStep.job_id == job_id).order_by(JobStep.created_at)
        return self.db.execute(stmt).scalars().all()

    # ----------------------------------------------------------------------
    # HITL Request Operations
    # ----------------------------------------------------------------------
    def create_hitl_request(
        self,
        job_id: str,
        run_id: str,
        request_type: str,
        component_name: str,
        invocation_fingerprint: str,
        payload: str,
        plan_fingerprint: str | None = None,
    ) -> "HITLRequest":
        from app.db.models import HITLRequest
        req = HITLRequest(
            job_id=job_id,
            run_id=run_id,
            request_type=request_type,
            component_name=component_name,
            invocation_fingerprint=invocation_fingerprint,
            plan_fingerprint=plan_fingerprint,
            payload=payload,
            status="PENDING",
        )
        self.db.add(req)
        self.db.commit()
        self.db.refresh(req)
        return req

    def get_hitl_request(self, request_id: str) -> "HITLRequest | None":
        from app.db.models import HITLRequest
        return self.db.get(HITLRequest, request_id)

    def get_pending_hitl_request_by_fingerprint(
        self, job_id: str, run_id: str, fingerprint: str
    ) -> "HITLRequest | None":
        from app.db.models import HITLRequest
        stmt = select(HITLRequest).where(
            HITLRequest.job_id == job_id,
            HITLRequest.run_id == run_id,
            HITLRequest.invocation_fingerprint == fingerprint,
            HITLRequest.status == "PENDING"
        )
        return self.db.execute(stmt).scalar_one_or_none()
    
    def get_approved_hitl_request_by_fingerprint(
        self, job_id: str, run_id: str, fingerprint: str
    ) -> "HITLRequest | None":
        from app.db.models import HITLRequest
        stmt = select(HITLRequest).where(
            HITLRequest.job_id == job_id,
            HITLRequest.run_id == run_id,
            HITLRequest.invocation_fingerprint == fingerprint,
            HITLRequest.status == "APPROVED"
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def update_hitl_request_status(
        self,
        request_id: str,
        status: str,
        decided_by: str | None = None,
        decision_reason: str | None = None,
    ) -> bool:
        from app.db.models import HITLRequest
        stmt = (
            update(HITLRequest)
            .where(
                HITLRequest.request_id == request_id,
                HITLRequest.status == "PENDING"
            )
            .values(
                status=status,
                decided_by=decided_by,
                decision_reason=decision_reason,
                decided_at=_utc_now()
            )
        )
        res = self.db.execute(stmt)
        # Flush, but let caller commit if needed
        self.db.flush()
        return res.rowcount > 0
    def get_conversation(self, chat_id: str, user_id: str) -> "Conversation | None":
        from app.db.models import Conversation
        stmt = select(Conversation).where(
            Conversation.chat_id == chat_id,
            Conversation.user_id == user_id
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_message(self, message_id: str, user_id: str) -> "Message | None":
        from app.db.models import Message, Conversation
        stmt = (
            select(Message)
            .join(Conversation)
            .where(
                Message.message_id == message_id,
                Conversation.user_id == user_id
            )
        )
        return self.db.execute(stmt).scalar_one_or_none()
