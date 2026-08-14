"""Tests for Phase 11 HITL Implementation."""

import pytest
import asyncio
from typing import Any
from uuid import uuid4

from app.agent.contracts import AgentExecutionContext
from app.agent.plan import Plan, PlanStep
from app.exceptions import AgentHITLPauseException, ToolExecutionError
from app.hitl.models import HITLPolicyDecision, HITLRequestStatus
from app.hitl.policy import HITLPolicy
from app.hitl.service import HITLService, compute_tool_fingerprint, compute_plan_fingerprint
from app.db.database import SessionLocal
from app.db.models import Job, User
from app.db.repository import SQLJobRepository
from app.mcp.policy import MCPPolicy, PolicyDecision


# Mock components
class MockBaseTool:
    def __init__(self, name="mock_tool"):
        self.name = name
    def execute(self, **kwargs):
        return "success"

class MockMCPTool(MockBaseTool):
    def __init__(self, name="mcp.test.tool", policy=None):
        super().__init__(name)
        self._client = type("MockClient", (), {"name": "test"})()
        self._policy = policy
        self._server_tool_name = "tool"


@pytest.fixture
def hitl_policy():
    return HITLPolicy(
        require_human_tools={"require_tool"},
        require_human_plans=True,
        denied_tools={"deny_tool"}
    )


@pytest.fixture
def hitl_service(hitl_policy):
    return HITLService(SessionLocal, hitl_policy)


@pytest.fixture
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def setup_job(db_session):
    repo = SQLJobRepository(db_session)
    user = repo.create_user(email=f"test_{uuid4()}@example.com")
    job_id = f"job_{uuid4().hex[:12]}"
    job = repo.create_job(job_id, user.user_id, "test goal")
    repo.update_job_status(job_id, user.user_id, "RUNNING")
    return job


def test_hitl_policy_native_tools(hitl_policy):
    allow_tool = MockBaseTool("allow_tool")
    require_tool = MockBaseTool("require_tool")
    deny_tool = MockBaseTool("deny_tool")

    assert hitl_policy.evaluate_tool(allow_tool, {}) == HITLPolicyDecision.ALLOW
    assert hitl_policy.evaluate_tool(require_tool, {}) == HITLPolicyDecision.REQUIRE_HUMAN
    assert hitl_policy.evaluate_tool(deny_tool, {}) == HITLPolicyDecision.DENY


def test_hitl_policy_mcp_tools(hitl_policy):
    class CustomMCPPolicy(MCPPolicy):
        def __init__(self, decision):
            self.decision = decision
        def evaluate_tool(self, server, tool):
            return self.decision

    allow_mcp = MockMCPTool(policy=CustomMCPPolicy(PolicyDecision.ALLOW))
    require_mcp = MockMCPTool(policy=CustomMCPPolicy(PolicyDecision.REQUIRE_HUMAN))
    deny_mcp = MockMCPTool(policy=CustomMCPPolicy(PolicyDecision.DENY))

    assert hitl_policy.evaluate_tool(allow_mcp, {}) == HITLPolicyDecision.ALLOW
    assert hitl_policy.evaluate_tool(require_mcp, {}) == HITLPolicyDecision.REQUIRE_HUMAN
    assert hitl_policy.evaluate_tool(deny_mcp, {}) == HITLPolicyDecision.DENY


def test_hitl_policy_plan(hitl_policy):
    plan = Plan(plan_id="p1", goal="test", steps={})
    assert hitl_policy.evaluate_plan(plan, None) == HITLPolicyDecision.REQUIRE_HUMAN


def test_hitl_service_tool_execution(hitl_service, setup_job):
    job = setup_job
    context = AgentExecutionContext(task="test", metadata={"job_id": job.job_id})
    tool = MockBaseTool("require_tool")
    args = {"param": "value"}

    # First execution should raise AgentHITLPauseException and create a request
    with pytest.raises(AgentHITLPauseException) as exc_info:
        hitl_service.evaluate_and_enforce_tool(tool, args, context)

    request_id = exc_info.value.request_id
    assert request_id is not None

    # Verify DB
    db = SessionLocal()
    repo = SQLJobRepository(db)
    req = repo.get_hitl_request(request_id)
    assert req is not None
    assert req.status == HITLRequestStatus.PENDING.value
    db.close()

    # Reject request
    # Let's manually transition it to WAITING_FOR_HUMAN.
    repo = SQLJobRepository(SessionLocal())
    repo.update_job_status(job.job_id, job.user_id, "WAITING_FOR_HUMAN")
    hitl_service.reject_request(request_id, job.job_id, decided_by="tester")

    req = SQLJobRepository(SessionLocal()).get_hitl_request(request_id)
    assert req.status == HITLRequestStatus.REJECTED.value

    job_after = SQLJobRepository(SessionLocal()).get_job(job.job_id, job.user_id)
    assert job_after.status == "CANCELLED"


def test_hitl_service_approve_tool_resume(hitl_service, setup_job):
    job = setup_job
    context = AgentExecutionContext(task="test", metadata={"job_id": job.job_id})
    tool = MockBaseTool("require_tool")
    args = {"param": "value"}

    with pytest.raises(AgentHITLPauseException) as exc_info:
        hitl_service.evaluate_and_enforce_tool(tool, args, context)

    request_id = exc_info.value.request_id
    repo = SQLJobRepository(SessionLocal())
    repo.update_job_status(job.job_id, job.user_id, "WAITING_FOR_HUMAN")
    
    assert hitl_service.approve_request(request_id, job.job_id, decided_by="tester")

    req = SQLJobRepository(SessionLocal()).get_hitl_request(request_id)
    assert req.status == HITLRequestStatus.APPROVED.value

    job_after = SQLJobRepository(SessionLocal()).get_job(job.job_id, job.user_id)
    assert job_after.status == "PENDING"

    # Execution should now pass without exception
    hitl_service.evaluate_and_enforce_tool(tool, args, context)


def test_hitl_service_different_args_requires_new_approval(hitl_service, setup_job):
    job = setup_job
    context = AgentExecutionContext(task="test", metadata={"job_id": job.job_id})
    tool = MockBaseTool("require_tool")
    
    # Arg 1
    with pytest.raises(AgentHITLPauseException) as exc1:
        hitl_service.evaluate_and_enforce_tool(tool, {"param": "v1"}, context)
    repo = SQLJobRepository(SessionLocal())
    repo.update_job_status(job.job_id, job.user_id, "WAITING_FOR_HUMAN")
    hitl_service.approve_request(exc1.value.request_id, job.job_id, "tester")
    
    # Arg 1 passes
    hitl_service.evaluate_and_enforce_tool(tool, {"param": "v1"}, context)

    # Arg 2 raises
    with pytest.raises(AgentHITLPauseException):
        hitl_service.evaluate_and_enforce_tool(tool, {"param": "v2"}, context)


def test_hitl_service_plan_execution(hitl_service, setup_job):
    job = setup_job
    context = AgentExecutionContext(task="test", metadata={"job_id": job.job_id})
    plan = Plan(plan_id="p1", goal="test plan", steps={})

    with pytest.raises(AgentHITLPauseException) as exc_info:
        hitl_service.evaluate_and_enforce_plan(plan, context)

    request_id = exc_info.value.request_id
    repo = SQLJobRepository(SessionLocal())
    repo.update_job_status(job.job_id, job.user_id, "WAITING_FOR_HUMAN")

    assert hitl_service.approve_request(request_id, job.job_id, decided_by="tester")

    approved_plan = hitl_service.get_approved_plan(context)
    assert approved_plan is not None
    assert approved_plan.plan_id == plan.plan_id

    # Should pass without exception now
    hitl_service.evaluate_and_enforce_plan(plan, context)


def test_hitl_concurrency(hitl_service, setup_job):
    job = setup_job
    context = AgentExecutionContext(task="test", metadata={"job_id": job.job_id})
    tool = MockBaseTool("require_tool")
    
    with pytest.raises(AgentHITLPauseException) as exc_info:
        hitl_service.evaluate_and_enforce_tool(tool, {}, context)
        
    request_id = exc_info.value.request_id
    repo = SQLJobRepository(SessionLocal())
    repo.update_job_status(job.job_id, job.user_id, "WAITING_FOR_HUMAN")
    
    # First approval should work
    assert hitl_service.approve_request(request_id, job.job_id, decided_by="user1")
    
    # Second approval should fail (return False) because status is no longer PENDING
    assert not hitl_service.approve_request(request_id, job.job_id, decided_by="user2")


def test_hitl_expiration(hitl_service, setup_job):
    job = setup_job
    context = AgentExecutionContext(task="test", metadata={"job_id": job.job_id})
    tool = MockBaseTool("require_tool")
    
    with pytest.raises(AgentHITLPauseException) as exc_info:
        hitl_service.evaluate_and_enforce_tool(tool, {}, context)
        
    request_id = exc_info.value.request_id
    repo = SQLJobRepository(SessionLocal())
    repo.update_job_status(job.job_id, job.user_id, "WAITING_FOR_HUMAN")

    # Manually expire the request
    import datetime
    db = SessionLocal()
    from app.db.models import HITLRequest
    req = db.get(HITLRequest, request_id)
    req.expires_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
    db.commit()
    db.close()

    count = hitl_service.expire_requests()
    assert count == 1

    req = SQLJobRepository(SessionLocal()).get_hitl_request(request_id)
    assert req.status == HITLRequestStatus.EXPIRED.value

    job_after = SQLJobRepository(SessionLocal()).get_job(job.job_id, job.user_id)
    assert job_after.status == "CANCELLED"
