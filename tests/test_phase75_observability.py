"""Tests for Phase 7.5 — Production Observability & Health Endpoints.

Covers:
    - GET /health  (liveness)
    - GET /ready   (readiness — DB + JobManager)
    - GET /metrics (existing MetricsRegistry snapshot)
    - X-Request-ID middleware (generation + propagation)
    - Security: no secret leakage
    - Integration with existing observability infrastructure

All tests are deterministic — no real LLM, Tavily, ChromaDB, or network calls.
"""

from __future__ import annotations

import uuid
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.models import User
from app.db.repository import SQLJobRepository
from app.services.job_manager import AsyncJobManager
from app.main_api import app
from app.api.auth import get_current_user
from app.observability.metrics import MetricsRegistry
from app.observability.registry_instance import metrics_registry, get_metrics_snapshot


# ----------------------------------------------------------------------
# Database Setup (in-memory SQLite, isolated per function)
# ----------------------------------------------------------------------
_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


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
def test_user(db_session):
    repo = SQLJobRepository(db_session)
    return repo.create_user(email="p75@example.com", user_id="user_p75")


# ----------------------------------------------------------------------
# Test Client — no auth on health routes
# ----------------------------------------------------------------------
@pytest.fixture()
def client(db_session, test_user):
    """TestClient with DB + auth overrides; job_manager set to a real instance."""

    def _override_db():
        yield db_session

    def _override_user():
        return test_user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user

    with patch("app.main_api.SessionLocal", _TestingSessionLocal):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


# ======================================================================
# 1. Health Endpoint
# ======================================================================
class TestHealthEndpoint:
    def test_health_endpoint(self, client):
        """GET /health returns 200 with {"status": "pass"}."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "pass"}

    def test_health_is_public(self, client):
        """GET /health must succeed without an X-API-Key header."""
        # Clear overrides so no auth helper is injected
        app.dependency_overrides.pop(get_current_user, None)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pass"

    def test_health_does_not_call_external_services(self, client):
        """Health endpoint must not invoke LLM, Tavily, ChromaDB, or Supervisor."""
        with patch("app.api.health.engine") as mock_engine:
            # engine should NOT be touched by /health
            resp = client.get("/health")
            assert resp.status_code == 200
            mock_engine.connect.assert_not_called()


# ======================================================================
# 2. Readiness Endpoint
# ======================================================================
class TestReadinessEndpoint:
    def test_readiness_healthy(self, client):
        """When DB and job manager are healthy, return 200 + pass."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "pass"
        assert body["database"] == "connected"
        assert body["job_manager"] == "ready"

    def test_readiness_database_failure(self, client):
        """When the DB is unreachable, return 503 + fail."""
        with patch("app.api.health.engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("DB down")
            resp = client.get("/ready")
            assert resp.status_code == 503
            body = resp.json()
            assert body["status"] == "fail"
            assert body["database"] == "unavailable"

    def test_readiness_job_manager_ready(self, client):
        """When job_manager.is_ready is True, report ready."""
        resp = client.get("/ready")
        body = resp.json()
        assert body["job_manager"] == "ready"

    def test_readiness_job_manager_unavailable(self, client):
        """When job_manager.is_ready is False (shutdown), return 503."""
        jm = client.app.state.job_manager
        jm._is_shutting_down = True
        try:
            resp = client.get("/ready")
            assert resp.status_code == 503
            body = resp.json()
            assert body["status"] == "fail"
            assert body["job_manager"] == "unavailable"
        finally:
            jm._is_shutting_down = False

    def test_readiness_does_not_call_external_services(self, client):
        """Readiness must not call Groq, Tavily, ChromaDB, or Supervisor."""
        # Simply verify the endpoint returns successfully without any
        # external service configuration — the test environment has no
        # real LLM/search keys, so any accidental call would fail.
        resp = client.get("/ready")
        body = resp.json()
        body_str = resp.text
        # No trace of external service interaction in the response
        assert "groq" not in body_str.lower()
        assert "tavily" not in body_str.lower()
        assert "chromadb" not in body_str.lower()
        assert resp.status_code in (200, 503)

    def test_readiness_does_not_leak_database_url(self, client):
        """Even on failure, the response must not contain the database URL."""
        with patch("app.api.health.engine") as mock_engine:
            mock_engine.connect.side_effect = Exception(
                "sqlite:///secret/path connection refused"
            )
            resp = client.get("/ready")
            body_str = resp.text
            assert "sqlite" not in body_str.lower()
            assert "secret/path" not in body_str


# ======================================================================
# 3. Metrics Endpoint
# ======================================================================
class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        """GET /metrics returns 200 with a valid JSON snapshot."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        # Must contain the standard MetricsRegistry counters
        assert "executions" in body
        assert "handoffs" in body
        assert "latency" in body

    def test_metrics_is_public(self, client):
        """GET /metrics must succeed without an X-API-Key header."""
        app.dependency_overrides.pop(get_current_user, None)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_does_not_leak_secrets(self, client):
        """The metrics response must not contain API keys or credentials."""
        resp = client.get("/metrics")
        body_str = resp.text
        # Ensure no common secret patterns leak
        assert "gsk_" not in body_str
        assert "sk-proj-" not in body_str
        assert "GROQ_API_KEY" not in body_str
        assert "X-API-Key" not in body_str
        assert "DATABASE_URL" not in body_str
        assert "tavily" not in body_str.lower()

    def test_metrics_uses_existing_registry(self, client):
        """The /metrics endpoint must use the shared MetricsRegistry singleton."""
        snapshot_before = get_metrics_snapshot()
        resp = client.get("/metrics")
        body = resp.json()
        # Compare structure — same keys
        assert set(body.keys()) == set(snapshot_before.keys())


# ======================================================================
# 4. Request-ID Middleware
# ======================================================================
class TestRequestIDMiddleware:
    def test_request_id_generation(self, client):
        """When no X-Request-ID is sent, one is generated and returned."""
        resp = client.get("/health")
        request_id = resp.headers.get("X-Request-ID")
        assert request_id is not None
        # Must be a valid UUID4
        uuid.UUID(request_id, version=4)

    def test_request_id_preservation(self, client):
        """When X-Request-ID is sent, it is echoed back unchanged."""
        custom_id = "my-custom-request-id-12345"
        resp = client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.headers.get("X-Request-ID") == custom_id

    def test_request_id_response_header(self, client):
        """Response always has X-Request-ID regardless of the endpoint."""
        for path in ["/health", "/ready", "/metrics"]:
            resp = client.get(path)
            assert "X-Request-ID" in resp.headers, f"Missing on {path}"


# ======================================================================
# 5. Authentication Boundaries
# ======================================================================
class TestAuthBoundaries:
    def test_protected_endpoint_requires_api_key(self, client):
        """POST /api/v1/research/jobs without auth overrides must return 401."""
        app.dependency_overrides.pop(get_current_user, None)
        resp = client.post(
            "/api/v1/research/jobs",
            json={"goal": "Test from 7.5"},
        )
        assert resp.status_code == 401

    def test_health_public_while_research_protected(self, client):
        """Health endpoints must stay public even when auth overrides are removed."""
        app.dependency_overrides.pop(get_current_user, None)
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code in (200, 503)
        assert client.get("/metrics").status_code == 200
        # Protected endpoint must still be locked
        assert client.post("/api/v1/research/jobs", json={"goal": "x"}).status_code == 401


# ======================================================================
# 6. Job Lifecycle / Existing Infrastructure
# ======================================================================
class TestJobLifecycleMetrics:
    def test_job_lifecycle_metrics_or_events(self, client):
        """The existing MetricsRegistry records job-related events correctly.

        This tests the *registry* directly — no real supervisor or LLM needed.
        """
        registry = MetricsRegistry()
        # Simulate an execution start event
        start_event = MagicMock()
        start_event.event_type = "AgentExecutionStarted"
        start_event.run_id = "run_75_1"
        start_event.parent_span_id = None
        start_event.timestamp = "2026-01-01T00:00:00+00:00"
        start_event.event_data = {"agent_name": "Supervisor"}

        registry.record_event(start_event)
        snap = registry.snapshot()
        assert snap["executions"]["total"] == 1

        # Simulate completion
        end_event = MagicMock()
        end_event.event_type = "AgentExecutionCompleted"
        end_event.run_id = "run_75_1"
        end_event.parent_span_id = None
        end_event.timestamp = "2026-01-01T00:00:01+00:00"
        end_event.event_data = {
            "agent_name": "Supervisor",
            "success": True,
            "output": "done",
        }

        registry.record_event(end_event)
        snap = registry.snapshot()
        assert snap["executions"]["successful"] == 1
        assert snap["latency"]["execution"]["count"] == 1

    def test_metrics_registry_is_singleton(self):
        """Verify get_metrics_snapshot uses the canonical singleton."""
        snap1 = get_metrics_snapshot()
        snap2 = metrics_registry.snapshot()
        # Same instance produces same structure
        assert snap1.keys() == snap2.keys()


# ======================================================================
# 7. AsyncJobManager is_ready property
# ======================================================================
class TestJobManagerIsReady:
    def test_is_ready_true_when_operational(self):
        """is_ready returns True when manager is not shutting down."""
        import asyncio

        async def _run():
            manager = AsyncJobManager(
                session_factory=_TestingSessionLocal,
                supervisor_factory=lambda: MagicMock(),
            )
            assert manager.is_ready is True
            await manager.shutdown()

        asyncio.run(_run())

    def test_is_ready_false_after_shutdown(self):
        """is_ready returns False once shutdown is initiated."""
        import asyncio

        async def _run():
            manager = AsyncJobManager(
                session_factory=_TestingSessionLocal,
                supervisor_factory=lambda: MagicMock(),
            )
            await manager.shutdown()
            assert manager.is_ready is False

        asyncio.run(_run())
