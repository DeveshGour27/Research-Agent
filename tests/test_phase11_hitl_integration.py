"""Integration tests for Phase 11 HITL Implementation."""

import pytest
import asyncio
from typing import Any
from fastapi.testclient import TestClient
from uuid import uuid4

from app.main_api import app
from app.db.database import SessionLocal
from app.hitl.service import HITLService
from app.hitl.policy import HITLPolicy
from app.db.repository import SQLJobRepository
from app.db.models import Job, User
from app.hitl.models import HITLPolicyDecision, HITLRequestStatus
from app.exceptions import AgentHITLPauseException
from app.api.auth import get_current_user

policy = HITLPolicy(require_human_plans=True)
app.state.hitl_service = HITLService(SessionLocal, policy)

client = TestClient(app)

@pytest.fixture
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def test_user(db_session):
    repo = SQLJobRepository(db_session)
    user = repo.create_user(email=f"test_{uuid4()}@example.com")
    user.user_id = f"test_user_{uuid4().hex[:8]}"
    db_session.commit()
    
    # override auth dynamically for this user
    def override_get_current_user() -> User:
        return user

    app.dependency_overrides[get_current_user] = override_get_current_user
    
    yield user
    
    # cleanup
    app.dependency_overrides.pop(get_current_user, None)

@pytest.fixture
def setup_job(db_session, test_user):
    repo = SQLJobRepository(db_session)
    job_id = f"job_{uuid4().hex[:12]}"
    job = repo.create_job(job_id, test_user.user_id, "test goal")
    repo.update_job_status(job_id, test_user.user_id, "RUNNING")
    repo.update_job_status(job_id, test_user.user_id, "WAITING_FOR_HUMAN")
    return job


def test_hitl_api_approve(setup_job):
    # Setup a request
    db = SessionLocal()
    repo = SQLJobRepository(db)
    req = repo.create_hitl_request(
        job_id=setup_job.job_id,
        run_id="test_run",
        request_type="tool",
        component_name="mock_tool",
        invocation_fingerprint="test_fingerprint",
        payload="{}"
    )
    request_id = req.request_id
    db.close()

    response = client.post(
        f"/api/v1/research/jobs/{setup_job.job_id}/hitl/{request_id}/approve"
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify status changed
    db = SessionLocal()
    repo = SQLJobRepository(db)
    req = repo.get_hitl_request(request_id)
    assert req.status == HITLRequestStatus.APPROVED.value
    db.close()


def test_hitl_api_reject(setup_job):
    # Setup a request
    db = SessionLocal()
    repo = SQLJobRepository(db)
    req = repo.create_hitl_request(
        job_id=setup_job.job_id,
        run_id="test_run",
        request_type="tool",
        component_name="mock_tool",
        invocation_fingerprint="test_fingerprint",
        payload="{}"
    )
    request_id = req.request_id
    db.close()

    response = client.post(
        f"/api/v1/research/jobs/{setup_job.job_id}/hitl/{request_id}/reject"
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify status changed
    db = SessionLocal()
    repo = SQLJobRepository(db)
    req = repo.get_hitl_request(request_id)
    assert req.status == HITLRequestStatus.REJECTED.value
    db.close()


def test_hitl_api_get_requests(setup_job):
    # Setup a request
    db = SessionLocal()
    repo = SQLJobRepository(db)
    req = repo.create_hitl_request(
        job_id=setup_job.job_id,
        run_id="test_run",
        request_type="tool",
        component_name="mock_tool",
        invocation_fingerprint="test_fingerprint",
        payload="{}"
    )
    request_id = req.request_id
    db.close()

    response = client.get(
        f"/api/v1/research/jobs/{setup_job.job_id}/hitl"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == setup_job.job_id
    assert len(data["hitl_requests"]) == 1
    assert data["hitl_requests"][0]["request_id"] == request_id
    assert data["hitl_requests"][0]["status"] == HITLRequestStatus.PENDING.value
