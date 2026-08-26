import pytest
from app.api.auth import get_api_key_hash
from fastapi.testclient import TestClient
from uuid import uuid4
import json
import asyncio

from app.main_api import app
from app.db.database import SessionLocal, Base, engine
from app.db.models import User, ApiKey, Job, JobStep, HITLRequest
from app.db.repository import SQLJobRepository
from app.hitl.service import HITLService
from app.hitl.policy import HITLPolicy

policy = HITLPolicy(require_human_plans=True)
hitl_service = HITLService(SessionLocal, policy)
app.state.hitl_service = hitl_service

# Initialize test database
Base.metadata.create_all(bind=engine)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def auth_user():
    db = SessionLocal()
    user_id = f"user_{uuid4().hex[:8]}"
    api_key = f"sk_{uuid4().hex}"
    
    user = User(user_id=user_id, email=f"{user_id}@example.com")
    key = ApiKey(key_hash=get_api_key_hash(api_key), user_id=user_id)
    db.add(user)
    db.add(key)
    db.commit()
    
    yield {"user_id": user_id, "api_key": api_key}
    
    db.query(JobStep).filter(JobStep.job_id.in_(db.query(Job.job_id).filter_by(user_id=user_id))).delete(synchronize_session=False)
    db.query(HITLRequest).filter(HITLRequest.job_id.in_(db.query(Job.job_id).filter_by(user_id=user_id))).delete(synchronize_session=False)
    db.query(Job).filter_by(user_id=user_id).delete(synchronize_session=False)
    db.query(ApiKey).filter_by(user_id=user_id).delete(synchronize_session=False)
    db.query(User).filter_by(user_id=user_id).delete(synchronize_session=False)
    db.commit()
    db.close()

@pytest.fixture
def headers(auth_user):
    return {"X-API-Key": auth_user["api_key"]}

@pytest.fixture
def other_user():
    db = SessionLocal()
    user_id = f"other_{uuid4().hex[:8]}"
    api_key = f"sk_{uuid4().hex}"
    
    user = User(user_id=user_id, email=f"{user_id}@example.com")
    key = ApiKey(key_hash=get_api_key_hash(api_key), user_id=user_id)
    db.add(user)
    db.add(key)
    db.commit()
    
    yield {"user_id": user_id, "api_key": api_key}
    db.query(Job).filter_by(user_id=user_id).delete(synchronize_session=False)
    db.query(ApiKey).filter_by(user_id=user_id).delete(synchronize_session=False)
    db.query(User).filter_by(user_id=user_id).delete(synchronize_session=False)
    db.commit()
    db.close()


def test_cors_headers(client):
    response = client.options(
        "/api/v1/research",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

def test_unauthenticated_request(client):
    response = client.post("/api/v1/research", json={"goal": "test"})
    assert response.status_code == 401

def test_create_research_job(client, headers):
    response = client.post("/api/v1/research", json={"goal": "What is AI?"}, headers=headers)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "PENDING"

def test_idor_job_access(client, headers, auth_user, other_user):
    # User 1 creates job
    response = client.post("/api/v1/research", json={"goal": "test"}, headers=headers)
    job_id = response.json()["job_id"]
    
    # User 2 tries to access
    other_headers = {"X-API-Key": other_user["api_key"]}
    res2 = client.get(f"/api/v1/jobs/{job_id}", headers=other_headers)
    assert res2.status_code == 404 # Isolated via 404
    assert "error" in res2.json()

def test_result_semantics(client, headers, auth_user):
    # Setup job
    db = SessionLocal()
    repo = SQLJobRepository(db)
    job = repo.create_job(job_id=f"job_{uuid4().hex[:12]}", user_id=auth_user["user_id"], goal="Test result")
    db.close()
    
    # Still PENDING
    res = client.get(f"/api/v1/research/{job.job_id}/result", headers=headers)
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "RESULT_UNAVAILABLE"
    
    # Complete the job (requires RUNNING first due to strict state machine)
    db = SessionLocal()
    repo = SQLJobRepository(db)
    repo.update_job_status(job.job_id, auth_user["user_id"], "RUNNING")
    repo.update_job_result(job.job_id, auth_user["user_id"], "COMPLETED", result="Final Answer")
    db.close()
    
    # Check again
    res = client.get(f"/api/v1/research/{job.job_id}/result", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["result"] == "Final Answer"

def test_events_timeline(client, headers, auth_user):
    db = SessionLocal()
    repo = SQLJobRepository(db)
    job = repo.create_job(job_id=f"job_{uuid4().hex[:12]}", user_id=auth_user["user_id"], goal="Test events")
    
    # Add step
    step = JobStep(step_id="s1", job_id=job.job_id, task_type="TOOL_CALL", status="COMPLETED")
    db.add(step)
    
    # Add HITL request
    hitl = HITLRequest(
        request_id="h1", job_id=job.job_id, run_id="r1",
        request_type="TOOL", component_name="search",
        invocation_fingerprint="f1", payload="{}"
    )
    db.add(hitl)
    db.commit()
    job_id_str = job.job_id
    db.close()

    res = client.get(f"/api/v1/jobs/{job_id_str}/events", headers=headers)
    assert res.status_code == 200
    events = res.json()["events"]
    
    assert len(events) == 3 # JOB_CREATED, TOOL_CALL, HITL_REQUESTED
    event_types = [e["event_type"] for e in events]
    assert "JOB_CREATED" in event_types
    assert "TOOL_CALL" in event_types
    assert "HITL_REQUESTED" in event_types
    
    for ev in events:
        assert ev["event_id"].startswith("job:") or ev["event_id"].startswith("jobstep:") or ev["event_id"].startswith("hitl:")
        assert ev["sequence"] is not None

def test_sse_stream_termination_and_reconnect(client, headers, auth_user):
    db = SessionLocal()
    repo = SQLJobRepository(db)
    job = repo.create_job(job_id=f"job_{uuid4().hex[:12]}", user_id=auth_user["user_id"], goal="Test SSE")
    repo.update_job_status(job.job_id, auth_user["user_id"], "RUNNING")
    repo.update_job_result(job.job_id, auth_user["user_id"], "COMPLETED", result="Done")
    db.close()
    
    # The stream should output JOB_CREATED and JOB_COMPLETED then close immediately because status is COMPLETED
    with client.stream("GET", f"/api/v1/jobs/{job.job_id}/events/stream", headers=headers) as response:
        assert response.status_code == 200
        text = response.read().decode('utf-8')
        assert "event: JOB_CREATED" in text
        assert "event: JOB_COMPLETED" in text

def test_sse_last_event_id(client, headers, auth_user):
    db = SessionLocal()
    repo = SQLJobRepository(db)
    job = repo.create_job(job_id=f"job_{uuid4().hex[:12]}", user_id=auth_user["user_id"], goal="Test SSE ID")
    repo.update_job_status(job.job_id, auth_user["user_id"], "RUNNING")
    repo.update_job_result(job.job_id, auth_user["user_id"], "COMPLETED", result="Done")
    db.close()
    
    last_id = f"job:{job.job_id}:created"
    req_headers = headers.copy()
    req_headers["Last-Event-ID"] = last_id
    
    with client.stream("GET", f"/api/v1/jobs/{job.job_id}/events/stream", headers=req_headers) as response:
        assert response.status_code == 200
        text = response.read().decode('utf-8')
        # Since last_event_id was 'created', it should only yield 'JOB_COMPLETED' (sequence > 0)
        assert "event: JOB_CREATED" not in text
        assert "event: JOB_COMPLETED" in text
