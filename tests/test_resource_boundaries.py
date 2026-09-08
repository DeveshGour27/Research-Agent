import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main_api import app
from app.db.database import get_db, SessionLocal
from app.db.models import User, UserSession, Job, Conversation, Message, _utc_now
from app.api.auth_utils import get_password_hash, generate_verification_token, hash_verification_token
from app.services.rate_limiter import RateLimiter
from app.constants import MAX_QUERY_LENGTH
from app.config import settings

client = TestClient(app)

@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def test_user(db: Session):
    user = db.query(User).filter(User.username == "resource_test_user").first()
    if not user:
        user = User(
            username="resource_test_user",
            email="resource_test@example.com",
            password_hash=get_password_hash("ValidPass123!"),
            email_verified=True
        )
        db.add(user)
        db.commit()
    return user

@pytest.fixture
def auth_headers(test_user, db: Session):
    from app.db.models import ApiKey
    import hashlib
    raw_key = "test-api-key-resource"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    if not api_key:
        api_key = ApiKey(key_hash=key_hash, user_id=test_user.user_id, is_active=True)
        db.add(api_key)
        db.commit()
    return {"X-API-Key": raw_key}

def test_send_message_rejects_oversized_content(test_user, auth_headers):
    # Create conversation
    create_res = client.post("/api/v1/chats", headers=auth_headers)
    chat_id = create_res.json()["chat_id"]

    oversized = "a" * (MAX_QUERY_LENGTH + 1)
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": oversized}, headers=auth_headers)
    assert res.status_code == 422

def test_send_message_enforces_active_job_quota(test_user, auth_headers, db: Session):
    # Create conversation
    create_res = client.post("/api/v1/chats", headers=auth_headers)
    chat_id = create_res.json()["chat_id"]

    # Insert 3 PENDING jobs for this user
    for i in range(settings.max_concurrent_jobs_per_user):
        job = Job(job_id=f"quota-job-{i}", user_id=test_user.user_id, goal="test goal", status="PENDING")
        db.merge(job)
    db.commit()

    # Next message should be rejected with 429
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "valid message"}, headers=auth_headers)
    assert res.status_code == 429
    assert "Concurrent job limit exceeded" in res.json()["detail"]

    # Clean up jobs
    db.query(Job).filter(Job.user_id == test_user.user_id).delete()
    db.commit()

def test_password_reset_revokes_active_sessions(test_user, db: Session):
    # Create an active session
    session_id = "test-session-revocation-123"
    db.query(UserSession).filter(UserSession.session_id == session_id).delete()
    user_session = UserSession(
        session_id=session_id,
        user_id=test_user.user_id,
        expires_at=_utc_now()
    )
    db.add(user_session)
    
    # Set reset token
    raw_token, hashed_token = generate_verification_token()
    test_user.reset_token_hash = hashed_token
    import datetime
    test_user.reset_token_expires_at = _utc_now() + datetime.timedelta(hours=1)
    db.commit()

    # Call reset-password
    res = client.post("/api/v1/auth/reset-password", json={
        "email": test_user.email,
        "token": raw_token,
        "password": "NewSecretPassword123!"
    })
    assert res.status_code == 200

    # Verify session was revoked
    db.expire_all()
    updated_session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    assert updated_session.revoked_at is not None


def test_context_deadline_and_child_span_inheritance():
    import time
    from app.agent.execution_context import AgentExecutionContext, ExecutionStatus

    now = time.monotonic()
    ctx = AgentExecutionContext(task="test deadline", deadline_at=now + 10.0)
    assert not ctx.is_expired
    assert not ctx.is_timed_out

    # Child inherits deadline
    child = ctx.create_child_span(task="child task")
    assert child.deadline_at == ctx.deadline_at

    # Past deadline expires
    past_ctx = AgentExecutionContext(task="past deadline", deadline_at=now - 1.0)
    assert past_ctx.is_expired
    assert past_ctx.is_timed_out
    assert past_ctx.status == ExecutionStatus.TIMED_OUT


def test_plan_executor_cooperative_cancellation_and_timeout():
    import time
    from app.agent.execution_context import AgentExecutionContext
    from app.agent.plan import Plan, PlanStep, PlanStepStatus
    from app.agent.execution_policy import ExecutionPolicy
    from app.agent.executor import PlanExecutor
    from app.agent.routing import CapabilityRouter
    from app.agent.registry import AgentRegistry
    from app.agent.communicator import InProcessCommunicator
    from app.exceptions import AgentTimeoutError, AgentCancellationError

    registry = AgentRegistry()
    communicator = InProcessCommunicator(registry)
    executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )

    plan = Plan(goal="test cooperative exit")
    step = PlanStep(
        step_id="step1",
        description="step one",
        task_type="reasoning",
        status=PlanStepStatus.READY,
    )
    plan.steps[step.step_id] = step

    # Test Cancellation
    cancel_ctx = AgentExecutionContext(task="cancel task")
    cancel_ctx.mark_cancelled()
    with pytest.raises(AgentCancellationError):
        executor.execute(plan, cancel_ctx, ExecutionPolicy())

    # Test Timeout via expired deadline
    timeout_ctx = AgentExecutionContext(
        task="timeout task",
        deadline_at=time.monotonic() - 5.0,
    )
    with pytest.raises(AgentTimeoutError):
        executor.execute(plan, timeout_ctx, ExecutionPolicy())

