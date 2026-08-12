"""Tests for Phase 7.3 Authentication & Authorization Guard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.repository import SQLJobRepository
from app.main_api import app
from app.api.auth import generate_api_key, get_api_key_hash

# ----------------------------------------------------------------------
# Test Fixtures (In-Memory SQLite with StaticPool)
# ----------------------------------------------------------------------

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def repo(db_session):
    return SQLJobRepository(db_session)

@pytest.fixture(scope="function")
def client(db_session):
    """Test client without overrides, directly hits actual auth."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    
    from unittest.mock import patch
    with patch("app.main_api.SessionLocal", TestingSessionLocal):
        with TestClient(app) as c:
            yield c
        
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def user1(repo):
    user = repo.create_user(email="user1@example.com")
    raw_key = generate_api_key()
    repo.create_api_key(user_id=user.user_id, key_hash=get_api_key_hash(raw_key))
    return {"user": user, "raw_key": raw_key}


@pytest.fixture(scope="function")
def user2(repo):
    user = repo.create_user(email="user2@example.com")
    raw_key = generate_api_key()
    repo.create_api_key(user_id=user.user_id, key_hash=get_api_key_hash(raw_key))
    return {"user": user, "raw_key": raw_key}


# ----------------------------------------------------------------------
# Authentication Tests
# ----------------------------------------------------------------------

def test_missing_api_key_returns_401(client):
    response = client.post("/api/v1/research/jobs", json={"goal": "Test"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API Key"

def test_invalid_api_key_returns_401(client):
    headers = {"X-API-Key": "invalid_key"}
    response = client.post("/api/v1/research/jobs", json={"goal": "Test"}, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or inactive API Key"

def test_inactive_api_key_returns_401(client, repo, user1):
    # Make key inactive
    api_key_record = repo.get_api_key(get_api_key_hash(user1["raw_key"]))
    api_key_record.is_active = False
    repo.db.commit()

    headers = {"X-API-Key": user1["raw_key"]}
    response = client.post("/api/v1/research/jobs", json={"goal": "Test"}, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or inactive API Key"

def test_valid_api_key_authenticates_user(client, user1):
    headers = {"X-API-Key": user1["raw_key"]}
    response = client.post("/api/v1/research/jobs", json={"goal": "Test"}, headers=headers)
    assert response.status_code == 202
    assert "job_id" in response.json()

def test_plaintext_api_key_is_not_persisted(repo, user1):
    # Verify the hash is stored, not the plaintext
    raw_key = user1["raw_key"]
    # Check that we can't find it directly as string anywhere in ApiKey table
    from app.db.models import ApiKey
    from sqlalchemy import select
    stmt = select(ApiKey).where(ApiKey.key_hash == raw_key)
    result = repo.db.execute(stmt).scalar_one_or_none()
    assert result is None
    
    # Check that hashing it allows us to find it
    key_hash = get_api_key_hash(raw_key)
    stmt2 = select(ApiKey).where(ApiKey.key_hash == key_hash)
    result2 = repo.db.execute(stmt2).scalar_one_or_none()
    assert result2 is not None

def test_health_endpoint_remains_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "pass"}

def test_authentication_does_not_accept_user_id_from_request(client, user1, repo):
    headers = {"X-API-Key": user1["raw_key"]}
    # Try to spoof user_id in payload, even though it's not accepted by Pydantic model
    # It should still be owned by user1.
    payload = {"goal": "Spoof test", "user_id": "hacker_user"}
    response = client.post("/api/v1/research/jobs", json=payload, headers=headers)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    
    # Verify it belongs to user1, not hacker_user
    job = repo.get_job(job_id=job_id, user_id=user1["user"].user_id)
    assert job is not None
    assert job.user_id == user1["user"].user_id

def test_authentication_failure_does_not_leak_sensitive_details(client):
    headers = {"X-API-Key": "valid_looking_but_fake_key"}
    response = client.post("/api/v1/research/jobs", json={"goal": "Test"}, headers=headers)
    assert response.status_code == 401
    assert "Invalid or inactive" in response.json()["detail"]
    assert "fake_key" not in str(response.json())


# ----------------------------------------------------------------------
# Authorization / Ownership Tests
# ----------------------------------------------------------------------

def test_job_creation_uses_authenticated_user(client, user1, repo):
    headers = {"X-API-Key": user1["raw_key"]}
    response = client.post("/api/v1/research/jobs", json={"goal": "Auth Goal"}, headers=headers)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    
    job = repo.get_job(job_id, user1["user"].user_id)
    assert job is not None
    assert job.goal == "Auth Goal"

def test_user_can_access_own_job(client, user1, repo):
    # Create job in db manually for user1
    job = repo.create_job(job_id="u1_job", user_id=user1["user"].user_id, goal="My Goal")
    
    headers = {"X-API-Key": user1["raw_key"]}
    response = client.get(f"/api/v1/research/jobs/{job.job_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["job_id"] == job.job_id

def test_user_cannot_access_other_users_job(client, user1, user2, repo):
    # Job belongs to user1
    job = repo.create_job(job_id="u1_job", user_id=user1["user"].user_id, goal="My Goal")
    
    # User2 tries to access it
    headers2 = {"X-API-Key": user2["raw_key"]}
    response = client.get(f"/api/v1/research/jobs/{job.job_id}", headers=headers2)
    
    # Should get 404 to avoid enumerating jobs
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"

def test_user_can_cancel_own_job(client, user1, repo):
    job = repo.create_job(job_id="u1_job", user_id=user1["user"].user_id, goal="My Goal")
    
    headers = {"X-API-Key": user1["raw_key"]}
    response = client.post(f"/api/v1/research/jobs/{job.job_id}/cancel", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"

def test_user_cannot_cancel_other_users_job(client, user1, user2, repo):
    # Job belongs to user1
    job = repo.create_job(job_id="u1_job", user_id=user1["user"].user_id, goal="My Goal")
    
    # User2 tries to cancel it
    headers2 = {"X-API-Key": user2["raw_key"]}
    response = client.post(f"/api/v1/research/jobs/{job.job_id}/cancel", headers=headers2)
    
    # Should get 404 to avoid enumerating jobs
    assert response.status_code == 404
    
    # Verify job in DB is still PENDING
    db_job = repo.get_job(job.job_id, user1["user"].user_id)
    assert db_job.status == "PENDING"

def test_multiple_users_are_isolated(client, user1, user2, repo):
    headers1 = {"X-API-Key": user1["raw_key"]}
    headers2 = {"X-API-Key": user2["raw_key"]}
    
    # Both create jobs
    resp1 = client.post("/api/v1/research/jobs", json={"goal": "Goal 1"}, headers=headers1)
    resp2 = client.post("/api/v1/research/jobs", json={"goal": "Goal 2"}, headers=headers2)
    job_id1 = resp1.json()["job_id"]
    job_id2 = resp2.json()["job_id"]
    
    # User 1 can't see User 2's job
    assert client.get(f"/api/v1/research/jobs/{job_id2}", headers=headers1).status_code == 404
    
    # User 2 can't see User 1's job
    assert client.get(f"/api/v1/research/jobs/{job_id1}", headers=headers2).status_code == 404
