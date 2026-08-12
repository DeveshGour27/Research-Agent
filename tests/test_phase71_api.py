"""Phase 7.1 API boundary tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main_api import app
from app.api.models import (
    HealthResponse,
    ResearchJobCancelResponse,
    ResearchJobCreateResponse,
    ResearchJobStatusResponse,
)
from app.api.auth import get_current_user
from app.db.models import User

# Override authentication for Phase 7.1 tests
def override_get_current_user() -> User:
    return User(user_id="test_user_71")


from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.db.database import Base, get_db

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

@pytest.fixture(scope="module")
def client():
    from app.db.repository import SQLJobRepository
    from unittest.mock import patch
    
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        repo = SQLJobRepository(session)
        if not repo.get_user("test_user_71"):
            repo.create_user("test71@example.com", "test_user_71")
        session.commit()
    finally:
        session.close()

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = lambda: TestingSessionLocal()
    
    class DummySupervisor:
        def execute(self, req):
            import time
            time.sleep(1.0)
            return None

    with patch("app.main_api.SessionLocal", TestingSessionLocal):
        with TestClient(app) as c:
            c.app.state.job_manager._supervisor_factory = lambda: DummySupervisor()
            yield c
            
    app.dependency_overrides.clear()


def test_health_endpoint(client) -> None:
    """Verify GET /health returns HTTP 200 and 'pass' status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "pass"}
    # Schema validation
    HealthResponse(**data)


def test_create_job_request_schema(client) -> None:
    """Verify POST /api/v1/research/jobs accepts valid goal and returns HTTP 202."""
    payload = {"goal": "Analyze recent quantum computing breakthroughs"}
    response = client.post("/api/v1/research/jobs", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "PENDING"
    assert "created_at" in data
    # Schema validation
    ResearchJobCreateResponse(**data)


def test_empty_goal_rejected(client) -> None:
    """Verify POST /api/v1/research/jobs rejects empty goal string with HTTP 422."""
    payload = {"goal": ""}
    response = client.post("/api/v1/research/jobs", json=payload)
    assert response.status_code == 422


def test_whitespace_goal_rejected(client) -> None:
    """Verify POST /api/v1/research/jobs rejects whitespace-only goal string with HTTP 422."""
    payload = {"goal": "   \n\t   "}
    response = client.post("/api/v1/research/jobs", json=payload)
    assert response.status_code == 422


def test_create_job_response_schema(client) -> None:
    """Verify POST /api/v1/research/jobs response conforms strictly to ResearchJobCreateResponse."""
    payload = {"goal": "Research renewable energy advances", "chat_id": "session_123"}
    response = client.post("/api/v1/research/jobs", json=payload)
    assert response.status_code == 202
    parsed = ResearchJobCreateResponse.model_validate(response.json())
    assert parsed.job_id.startswith("job_")
    assert parsed.status in ("PENDING", "RUNNING")


def test_job_status_endpoint_contract(client) -> None:
    """Verify GET /api/v1/research/jobs/{job_id} exists and conforms to ResearchJobStatusResponse."""
    # Create job first
    create_response = client.post("/api/v1/research/jobs", json={"goal": "Test goal"})
    job_id = create_response.json()["job_id"]
    
    response = client.get(f"/api/v1/research/jobs/{job_id}")
    assert response.status_code == 200
    parsed = ResearchJobStatusResponse.model_validate(response.json())
    assert parsed.job_id == job_id
    assert parsed.status in ("PENDING", "RUNNING")
    assert parsed.result is None


def test_cancel_job_endpoint_contract(client) -> None:
    """Verify POST /api/v1/research/jobs/{job_id}/cancel exists and returns ResearchJobCancelResponse."""
    # Create job first
    create_response = client.post("/api/v1/research/jobs", json={"goal": "Test goal"})
    job_id = create_response.json()["job_id"]
    
    response = client.post(f"/api/v1/research/jobs/{job_id}/cancel")
    assert response.status_code == 200
    parsed = ResearchJobCancelResponse.model_validate(response.json())
    assert parsed.job_id == job_id
    assert parsed.status == "CANCELLED"


def test_openapi_schema(client) -> None:
    """Verify GET /openapi.json generates valid OpenAPI documentation containing required routes."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    paths = schema["paths"]
    assert "/health" in paths
    assert "/api/v1/research/jobs" in paths
    assert "/api/v1/research/jobs/{job_id}" in paths
    assert "/api/v1/research/jobs/{job_id}/cancel" in paths
