"""Tests for Phase 7.7 — Queue Observability.

Covers:
- GET /api/v1/research/jobs/{job_id} enhancements (attempt_count, timestamps)
- GET /api/v1/research/jobs/stats
- Tenant isolation
- Queue statistics accuracy
"""

import datetime
import uuid
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.models import User, Job
from app.db.repository import SQLJobRepository
from app.main_api import app
from app.api.auth import get_current_user
from app.config import settings

def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

@pytest.fixture(scope="function", autouse=True)
def _db_schema():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)

@pytest.fixture(scope="function")
def db_session(_db_schema):
    session = _TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def test_user_a(db_session):
    repo = SQLJobRepository(db_session)
    return repo.create_user(email="user_a@example.com", user_id="user_a")

@pytest.fixture(scope="function")
def test_user_b(db_session):
    repo = SQLJobRepository(db_session)
    return repo.create_user(email="user_b@example.com", user_id="user_b")

@pytest.fixture()
def client_factory(db_session):
    def _create_client(user):
        def _override_db():
            yield db_session
        def _override_user():
            return user
        
        app.dependency_overrides[get_db] = _override_db
        app.dependency_overrides[get_current_user] = _override_user
        return TestClient(app)
    
    yield _create_client
    app.dependency_overrides.clear()

def test_enhanced_job_status(client_factory, db_session, test_user_a):
    client_user_a = client_factory(test_user_a)
    # Create job directly
    now = _utc_now()
    job = Job(
        job_id="job_status_test",
        user_id=test_user_a.user_id,
        goal="status test",
        status="RUNNING",
        attempt_count=2,
        created_at=now - datetime.timedelta(seconds=100),
        started_at=now - datetime.timedelta(seconds=50),
    )
    db_session.add(job)
    db_session.commit()

    resp = client_user_a.get(f"/api/v1/research/jobs/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["attempt_count"] == 2
    assert data["created_at"] is not None
    assert data["started_at"] is not None
    assert data["completed_at"] is None
    assert data["execution_time_seconds"] is None

def test_enhanced_job_status_completed(client_factory, db_session, test_user_a):
    client_user_a = client_factory(test_user_a)
    now = _utc_now()
    started = now - datetime.timedelta(seconds=50)
    completed = now
    job = Job(
        job_id="job_status_test_comp",
        user_id=test_user_a.user_id,
        goal="status test",
        status="COMPLETED",
        attempt_count=1,
        created_at=started - datetime.timedelta(seconds=10),
        started_at=started,
        completed_at=completed,
    )
    db_session.add(job)
    db_session.commit()

    resp = client_user_a.get(f"/api/v1/research/jobs/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_time_seconds"] == 50.0
    assert data["completed_at"] is not None

def test_queue_stats_isolation(db_session, test_user_a, test_user_b):
    # Setup db override once
    app.dependency_overrides[get_db] = lambda: db_session
    
    # User A has 2 pending, 1 running
    db_session.add(Job(job_id="a1", user_id=test_user_a.user_id, goal="g", status="PENDING"))
    db_session.add(Job(job_id="a2", user_id=test_user_a.user_id, goal="g", status="PENDING"))
    db_session.add(Job(job_id="a3", user_id=test_user_a.user_id, goal="g", status="RUNNING", heartbeat_at=_utc_now()))
    
    # User B has 1 failed, 1 cancelled
    db_session.add(Job(job_id="b1", user_id=test_user_b.user_id, goal="g", status="FAILED"))
    db_session.add(Job(job_id="b2", user_id=test_user_b.user_id, goal="g", status="CANCELLED"))
    db_session.commit()

    # Query A
    app.dependency_overrides[get_current_user] = lambda: test_user_a
    with TestClient(app) as client:
        resp_a = client.get("/api/v1/research/jobs/stats")
        assert resp_a.status_code == 200
        stats_a = resp_a.json()
        assert stats_a["pending"] == 2
        assert stats_a["running"] == 1
        assert stats_a["failed"] == 0
        assert stats_a["cancelled"] == 0

    # Query B
    app.dependency_overrides[get_current_user] = lambda: test_user_b
    with TestClient(app) as client:
        resp_b = client.get("/api/v1/research/jobs/stats")
        assert resp_b.status_code == 200
        stats_b = resp_b.json()
        assert stats_b["pending"] == 0
        assert stats_b["running"] == 0
        assert stats_b["failed"] == 1
        assert stats_b["cancelled"] == 1
        
    app.dependency_overrides.clear()

def test_queue_stats_stale(client_factory, db_session, test_user_a):
    client_user_a = client_factory(test_user_a)
    stale_threshold = settings.job_stale_after_seconds
    now = _utc_now()
    
    # 1 fresh running
    db_session.add(Job(job_id="a1", user_id=test_user_a.user_id, goal="g", status="RUNNING", heartbeat_at=now))
    
    # 1 stale running
    db_session.add(Job(job_id="a2", user_id=test_user_a.user_id, goal="g", status="RUNNING", 
                       heartbeat_at=now - datetime.timedelta(seconds=stale_threshold + 5)))
                       
    # 1 exactly at cutoff (or newer, to avoid race conditions with time advancing)
    db_session.add(Job(job_id="a3", user_id=test_user_a.user_id, goal="g", status="RUNNING", 
                       heartbeat_at=now - datetime.timedelta(seconds=stale_threshold - 5)))
                       
    db_session.commit()

    resp = client_user_a.get("/api/v1/research/jobs/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert stats["running"] == 3
    # Only a2 is strictly less than cutoff
    assert stats["stale"] == 1

def test_queue_stats_unauthenticated():
    # Remove overrides for authentication
    app.dependency_overrides.pop(get_current_user, None)
    with TestClient(app) as client:
        resp = client.get("/api/v1/research/jobs/stats")
        assert resp.status_code == 401

def test_job_status_unauthenticated():
    app.dependency_overrides.pop(get_current_user, None)
    with TestClient(app) as client:
        resp = client.get("/api/v1/research/jobs/fake_job")
        assert resp.status_code == 401
