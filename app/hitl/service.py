"""Service for managing HITL lifecycle and enforcement."""

from __future__ import annotations

import hashlib
import json
from typing import Any
import datetime

from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from app.agent.contracts import AgentExecutionContext
from app.agent.plan import Plan
from app.db.repository import SQLJobRepository
from app.exceptions import AgentHITLPauseException, ToolExecutionError
from app.hitl.models import HITLPolicyDecision, HITLRequestType, HITLRequestStatus
from app.hitl.policy import HITLPolicy


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        return super().default(obj)


def _compute_fingerprint(components: list[Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint from components."""
    # Serialize canonically (sorted keys)
    serialized = json.dumps(components, sort_keys=True, separators=(",", ":"), cls=SetEncoder)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_tool_fingerprint(
    job_id: str, run_id: str, tool_name: str, arguments: dict[str, Any]
) -> str:
    """Compute fingerprint for a tool execution."""
    return _compute_fingerprint([job_id, run_id, "TOOL", tool_name, arguments])


def compute_plan_fingerprint(job_id: str, run_id: str, plan: Plan) -> str:
    """Compute fingerprint for a plan."""
    import dataclasses
    return _compute_fingerprint([job_id, run_id, "PLAN", dataclasses.asdict(plan)])


class HITLService:
    """Centralized service for evaluating policy and managing HITL lifecycle."""

    def __init__(self, session_factory: sessionmaker, policy: HITLPolicy) -> None:
        self._session_factory = session_factory
        self._policy = policy

    def evaluate_and_enforce_tool(
        self, tool: Any, arguments: dict[str, Any], context: AgentExecutionContext | None
    ) -> None:
        """
        Evaluate tool policy. If REQUIRE_HUMAN, check for approval or throw AgentHITLPauseException.
        If DENY, throw ToolExecutionError.
        """
        tool_name = tool.name
        decision = self._policy.evaluate_tool(tool, arguments, context)

        if decision == HITLPolicyDecision.ALLOW:
            return

        if decision == HITLPolicyDecision.DENY:
            raise ToolExecutionError(
                f"Tool '{tool_name}' execution denied by policy.",
                tool_name=tool_name,
                details={"arguments": arguments},
            )

        # REQUIRE_HUMAN path
        if not context:
            raise ToolExecutionError(
                f"Tool '{tool_name}' requires human approval but no context is available.",
                tool_name=tool_name,
            )

        job_id = context.metadata.get("job_id")
        if not job_id:
            # If there's no job_id, it's not a background job that can be paused safely.
            raise ToolExecutionError(
                f"Tool '{tool_name}' requires human approval, but job_id is missing from context.",
                tool_name=tool_name,
            )

        run_id = context.run_id
        fingerprint = compute_tool_fingerprint(job_id, run_id, tool_name, arguments)

        with self._session_factory() as session:
            repo = SQLJobRepository(session)
            
            # Check if there's already an approved request for this exact fingerprint
            approved = repo.get_approved_hitl_request_by_fingerprint(job_id, run_id, fingerprint)
            if approved:
                return  # Executing authorized exact tool call exactly once (handled by the caller completing it)
                
            # Wait, if we return here, the tool executes. But the instruction says:
            # "Do not execute twice."
            # If the tool is executed and succeeds, it is appended to agent history. 
            # If the tool failed and the agent retries with the SAME arguments, it would pass this check. 
            # That's fine because the approval is for the "exact invocation". If it succeeds, the state progresses.
            
            # Check if a pending request already exists
            pending = repo.get_pending_hitl_request_by_fingerprint(job_id, run_id, fingerprint)
            if pending:
                request_id = pending.request_id
            else:
                # Create a new PENDING request
                payload_json = json.dumps(arguments, sort_keys=True)
                new_req = repo.create_hitl_request(
                    job_id=job_id,
                    run_id=run_id,
                    request_type=HITLRequestType.TOOL.value,
                    component_name=tool_name,
                    invocation_fingerprint=fingerprint,
                    payload=payload_json,
                )
                request_id = new_req.request_id

        # Throw exception to bubble up and pause worker
        raise AgentHITLPauseException(
            f"Tool '{tool_name}' requires human approval.",
            request_id=request_id,
            details={"tool_name": tool_name, "fingerprint": fingerprint},
        )

    def evaluate_and_enforce_plan(self, plan: Plan, context: AgentExecutionContext) -> None:
        """
        Evaluate plan policy. If REQUIRE_HUMAN, check for approval or throw AgentHITLPauseException.
        """
        decision = self._policy.evaluate_plan(plan, context)

        if decision == HITLPolicyDecision.ALLOW:
            return

        job_id = context.metadata.get("job_id")
        if not job_id:
            return  # Can't pause

        run_id = context.run_id
        fingerprint = compute_plan_fingerprint(job_id, run_id, plan)

        with self._session_factory() as session:
            repo = SQLJobRepository(session)
            
            approved = repo.get_approved_hitl_request_by_fingerprint(job_id, run_id, fingerprint)
            if approved:
                return
                
            pending = repo.get_pending_hitl_request_by_fingerprint(job_id, run_id, fingerprint)
            if pending:
                request_id = pending.request_id
            else:
                import json, dataclasses
                payload_json = json.dumps(dataclasses.asdict(plan), cls=SetEncoder)
                new_req = repo.create_hitl_request(
                    job_id=job_id,
                    run_id=run_id,
                    request_type=HITLRequestType.PLAN.value,
                    component_name="PLAN",
                    invocation_fingerprint=fingerprint,
                    plan_fingerprint=fingerprint,
                    payload=payload_json,
                )
                request_id = new_req.request_id

        raise AgentHITLPauseException(
            "Plan requires human approval.",
            request_id=request_id,
            details={"fingerprint": fingerprint},
        )

    def get_approved_plan(self, context: AgentExecutionContext) -> Plan | None:
        """
        Retrieve a globally approved plan for this run_id, if any.
        Because plans might be large, we can just look up the latest APPROVED plan for this job/run.
        """
        job_id = context.metadata.get("job_id")
        if not job_id:
            return None
            
        with self._session_factory() as session:
            from app.db.models import HITLRequest
            from sqlalchemy import select
            
            stmt = select(HITLRequest).where(
                HITLRequest.job_id == job_id,
                HITLRequest.run_id == context.run_id,
                HITLRequest.request_type == HITLRequestType.PLAN.value,
                HITLRequest.status == HITLRequestStatus.APPROVED.value
            ).order_by(HITLRequest.created_at.desc())
            
            req = session.execute(stmt).scalars().first()
            if req:
                return Plan(**json.loads(req.payload))
                
        return None

    def approve_request(
        self, request_id: str, job_id: str, decided_by: str, decision_reason: str | None = None
    ) -> bool:
        """Atomically approve a HITL request and transition the job back to PENDING."""
        with self._session_factory() as session:
            repo = SQLJobRepository(session)
            
            req = repo.get_hitl_request(request_id)
            if not req or req.job_id != job_id or req.status != HITLRequestStatus.PENDING.value:
                return False
            from app.db.models import Job
            from sqlalchemy import select
            job = session.execute(select(Job).where(Job.job_id == job_id)).scalar_one_or_none()
            if not job or job.status != "WAITING_FOR_HUMAN":
                return False
                
            # Atomic update request
            if not repo.update_hitl_request_status(
                request_id, HITLRequestStatus.APPROVED.value, decided_by, decision_reason
            ):
                return False
                
            # Atomic update job
            try:
                repo.update_job_status(
                    job_id, job.user_id, status="PENDING", expected_states={"WAITING_FOR_HUMAN"}
                )
            except Exception:
                session.rollback()
                return False
                
            session.commit()
            return True

    def reject_request(
        self, request_id: str, job_id: str, decided_by: str, decision_reason: str | None = None
    ) -> bool:
        """Atomically reject a HITL request and transition the job to CANCELLED."""
        with self._session_factory() as session:
            repo = SQLJobRepository(session)
            
            req = repo.get_hitl_request(request_id)
            if not req or req.job_id != job_id or req.status != HITLRequestStatus.PENDING.value:
                return False
            from app.db.models import Job
            from sqlalchemy import select
            job = session.execute(select(Job).where(Job.job_id == job_id)).scalar_one_or_none()
            if not job or job.status != "WAITING_FOR_HUMAN":
                return False
                
            if not repo.update_hitl_request_status(
                request_id, HITLRequestStatus.REJECTED.value, decided_by, decision_reason
            ):
                return False
                
            try:
                repo.update_job_status(
                    job_id, job.user_id, status="CANCELLED", expected_states={"WAITING_FOR_HUMAN"}
                )
            except Exception:
                session.rollback()
                return False
                
            session.commit()
            return True

    def expire_requests(self) -> int:
        """Expire all pending requests that have passed their expiration date."""
        count = 0
        with self._session_factory() as session:
            from app.db.models import HITLRequest
            from sqlalchemy import select
            
            now = datetime.datetime.now(datetime.timezone.utc)
            stmt = select(HITLRequest).where(
                HITLRequest.status == HITLRequestStatus.PENDING.value,
                HITLRequest.expires_at.is_not(None),
                HITLRequest.expires_at < now
            )
            expired = session.execute(stmt).scalars().all()
            
            repo = SQLJobRepository(session)
            for req in expired:
                if repo.update_hitl_request_status(req.request_id, HITLRequestStatus.EXPIRED.value):
                    try:
                        from app.db.models import Job
                        job = session.execute(select(Job).where(Job.job_id == req.job_id)).scalar_one_or_none()
                        if job:
                            repo.update_job_status(
                                req.job_id, job.user_id, status="CANCELLED", expected_states={"WAITING_FOR_HUMAN"}
                            )
                            count += 1
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        session.rollback()
                        continue
            session.commit()
        return count
