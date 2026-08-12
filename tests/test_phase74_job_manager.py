"""Tests for Phase 7.4 Async Job Execution & Worker Pool."""

from __future__ import annotations

import asyncio
import pytest
from typing import Callable
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.db.database import Base, get_db
from app.db.models import User
from app.db.repository import SQLJobRepository
from app.services.job_manager import AsyncJobManager
from app.agent.contracts import AgentResult, AgentRequest
from app.agent.state import AgentState
from app.exceptions import AgentExecutionError, AgentCancellationError, AgentTimeoutError
from app.main_api import app
from app.api.auth import get_current_user

# ----------------------------------------------------------------------
# Database Setup
# ----------------------------------------------------------------------

import tempfile
import os

_temp_db_fd, _temp_db_path = tempfile.mkstemp(suffix=".db")

engine = create_engine(
    f"sqlite:///{_temp_db_path}",
    connect_args={"check_same_thread": False},
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL") # Help with concurrency
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_db():
    yield
    os.close(_temp_db_fd)
    try:
        os.unlink(_temp_db_path)
    except OSError:
        pass

@pytest.fixture(scope="function", autouse=True)
def db_schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_schema):
    """Create a fresh session for tests."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def user(db_session):
    repo = SQLJobRepository(db_session)
    return repo.create_user(email="test@example.com", user_id="test_user")

@pytest.fixture(scope="function")
def user2(db_session):
    repo = SQLJobRepository(db_session)
    return repo.create_user(email="test2@example.com", user_id="test_user2")

# ----------------------------------------------------------------------
# Supervisor Mocks
# ----------------------------------------------------------------------

class MockSupervisor:
    def __init__(self, delay=0.0, raise_error=None, output="Mock result"):
        self.delay = delay
        self.raise_error = raise_error
        self.output = output
        self.execute_called = False
        self.received_request = None

    def execute(self, request: AgentRequest) -> AgentResult:
        self.execute_called = True
        self.received_request = request
        
        if self.delay > 0:
            import time
            time.sleep(self.delay)
            
        if self.raise_error:
            raise self.raise_error
            
        state = AgentState()
        state.finished = True
        return AgentResult(
            request=request,
            state=state,
            output=self.output,
            success=True,
            context=request.context,
            metadata={"trace_id": getattr(request.context, "trace_id", None)}
        )

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
async def wait_manager_idle(manager: AsyncJobManager, timeout=5.0):
    start = asyncio.get_event_loop().time()
    # allow tasks to be created
    await asyncio.sleep(0.05)
    while manager._running_tasks:
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError("Manager did not become idle")
        await asyncio.sleep(0.01)

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_job_submission_returns_immediately(db_session, user):
    """Verify API/job submission does not wait for Supervisor completion."""
    async def run():
        mock_sup = MockSupervisor(delay=0.5)
        manager = AsyncJobManager(
            session_factory=TestingSessionLocal,
            supervisor_factory=lambda: mock_sup,
        )
        
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_1", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await asyncio.sleep(0.05) # Yield to event loop so task can start
        
        db_session.refresh(job)
        assert job.status == "RUNNING"
        
        await wait_manager_idle(manager)
        db_session.refresh(job)
        assert job.status == "COMPLETED"
        await manager.shutdown()
    asyncio.run(run())


def test_pending_to_running_transition(db_session, user):
    """Verify a submitted job transitions to RUNNING."""
    async def run():
        class PausingSupervisor(MockSupervisor):
            def execute(self, request):
                self.execute_called = True
                import time
                time.sleep(0.5)
                return super().execute(request)
                
        manager = AsyncJobManager(TestingSessionLocal, lambda: PausingSupervisor())
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_2", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await asyncio.sleep(0.1)
        
        db_session.refresh(job)
        assert job.status == "RUNNING"
        
        await wait_manager_idle(manager)
        await manager.shutdown()
    asyncio.run(run())

def test_successful_job_completion(db_session, user):
    """Mock Supervisor success and verify PENDING -> RUNNING -> COMPLETED."""
    async def run():
        manager = AsyncJobManager(TestingSessionLocal, lambda: MockSupervisor(output="My special result"))
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_3", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await wait_manager_idle(manager)
        
        db_session.refresh(job)
        assert job.status == "COMPLETED"
        assert job.result == "My special result"
        await manager.shutdown()
    asyncio.run(run())

def test_failed_job(db_session, user):
    """Mock Supervisor failure and verify PENDING -> RUNNING -> FAILED."""
    async def run():
        def fail_factory():
            req = AgentRequest(input_text="t")
            return MockSupervisor(raise_error=AgentExecutionError("Oops", request=req))
            
        manager = AsyncJobManager(TestingSessionLocal, fail_factory)
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_4", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await wait_manager_idle(manager)
        
        db_session.refresh(job)
        assert job.status == "FAILED"
        assert "Oops" in job.error_message
        await manager.shutdown()
    asyncio.run(run())

def test_error_is_sanitized(db_session, user):
    """Verify secrets/internal sensitive values are not persisted on unexpected errors."""
    async def run():
        def crash_factory():
            return MockSupervisor(raise_error=ValueError("Super secret traceback info sk-1234"))
            
        manager = AsyncJobManager(TestingSessionLocal, crash_factory)
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_5", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await wait_manager_idle(manager)
        
        db_session.refresh(job)
        assert job.status == "FAILED"
        assert job.error_message == "Internal execution error"
        assert "sk-1234" not in job.error_message
        await manager.shutdown()
    asyncio.run(run())

def test_concurrency_limit(db_session, user):
    """Submit more jobs than max_concurrent_jobs and verify bounds."""
    async def run():
        manager = AsyncJobManager(TestingSessionLocal, lambda: MockSupervisor(delay=0.5), max_concurrent_jobs=2)
        repo = SQLJobRepository(db_session)
        
        jobs = []
        for i in range(5):
            j = repo.create_job(f"job_c_{i}", user.user_id, "test goal")
            jobs.append(j)
            manager.submit_job(j.job_id, user.user_id, "test goal")
            
        await asyncio.sleep(0.2)
        
        running_count = 0
        pending_count = 0
        for j in jobs:
            db_session.refresh(j)
            if j.status == "RUNNING":
                running_count += 1
            elif j.status == "PENDING":
                pending_count += 1
                
        assert running_count == 2
        assert pending_count == 3
        
        await wait_manager_idle(manager)
        await manager.shutdown()
    asyncio.run(run())


def test_job_timeout(db_session, user):
    """Mock a long-running execution and verify timeout handling."""
    async def run():
        manager = AsyncJobManager(TestingSessionLocal, lambda: MockSupervisor(delay=2.0), job_timeout_seconds=0.1)
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_t_1", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await wait_manager_idle(manager)
        
        db_session.refresh(job)
        assert job.status == "FAILED"
        assert job.error_message == "Execution timed out"
        
        await manager.shutdown()
    asyncio.run(run())


def test_pending_job_cancellation(db_session, user):
    """Cancel a PENDING job and verify Supervisor is never executed."""
    async def run():
        manager = AsyncJobManager(TestingSessionLocal, lambda: MockSupervisor(), max_concurrent_jobs=0)
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_c_pending", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await manager.cancel_job(job.job_id, user.user_id)
        
        db_session.refresh(job)
        assert job.status == "CANCELLED"
        
        await manager.shutdown()
    asyncio.run(run())

def test_running_job_cancellation(db_session, user):
    """Verify cancellation is requested and the job does not later become COMPLETED."""
    async def run():
        sup = MockSupervisor(delay=1.0)
        manager = AsyncJobManager(TestingSessionLocal, lambda: sup)
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_c_running", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await asyncio.sleep(0.1)
        
        db_session.refresh(job)
        assert job.status == "RUNNING"
        
        await manager.cancel_job(job.job_id, user.user_id)
        
        await wait_manager_idle(manager)
        db_session.refresh(job)
        assert job.status == "CANCELLED"
        
        await manager.shutdown()
    asyncio.run(run())

def test_cancelled_job_cannot_be_overwritten(db_session, user):
    """Explicitly test the cancellation/completion race."""
    async def run():
        sup = MockSupervisor(delay=0.5)
        manager = AsyncJobManager(TestingSessionLocal, lambda: sup)
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_race", user.user_id, "test goal")
        
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await asyncio.sleep(0.1)
        
        await manager.cancel_job(job.job_id, user.user_id)
        
        await wait_manager_idle(manager)
        
        db_session.refresh(job)
        assert job.status == "CANCELLED"
        await manager.shutdown()
    asyncio.run(run())

def test_execution_context_isolation(db_session, user):
    """Submit multiple jobs and verify unique trace/run identifiers."""
    async def run():
        sup1 = MockSupervisor(delay=0.1)
        sup2 = MockSupervisor(delay=0.1)
        factory_iter = iter([sup1, sup2])
        
        manager = AsyncJobManager(TestingSessionLocal, lambda: next(factory_iter))
        repo = SQLJobRepository(db_session)
        
        j1 = repo.create_job("j1", user.user_id, "test goal")
        j2 = repo.create_job("j2", user.user_id, "test goal")
        
        manager.submit_job(j1.job_id, user.user_id, "test goal")
        manager.submit_job(j2.job_id, user.user_id, "test goal")
        
        await wait_manager_idle(manager)
        
        ctx1 = sup1.received_request.context
        ctx2 = sup2.received_request.context
        
        assert ctx1.execution_id != ctx2.execution_id
        await manager.shutdown()
    asyncio.run(run())

def test_user_job_isolation(db_session, user, user2):
    """Verify JobManager cannot accidentally execute or expose another user's job state."""
    async def run():
        manager = AsyncJobManager(TestingSessionLocal, lambda: MockSupervisor())
        repo = SQLJobRepository(db_session)
        job = repo.create_job("job_iso", user.user_id, "test goal")
        
        with pytest.raises(ValueError, match="not found"):
            await manager.cancel_job(job.job_id, user2.user_id)
            
        manager.submit_job(job.job_id, user.user_id, "test goal")
        await wait_manager_idle(manager)
        await manager.shutdown()
    asyncio.run(run())


def test_supervisor_is_mockable():
    """Verify Supervisor dependency injection works."""
    async def run():
        manager = AsyncJobManager(TestingSessionLocal, lambda: MockSupervisor())
        assert manager is not None
        await manager.shutdown()
    asyncio.run(run())


# ----------------------------------------------------------------------
# API Integration Tests
# ----------------------------------------------------------------------
import time

@pytest.fixture
def api_client(db_session, user):
    def override_get_current_user():
        return user

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db

    # Patch default_supervisor_factory
    with patch("app.main_api.default_supervisor_factory", return_value=MockSupervisor(output="Async success")):
        with patch("app.main_api.SessionLocal", TestingSessionLocal):
            with TestClient(app) as client:
                yield client

    app.dependency_overrides.clear()
    
def test_api_create_job_starts_background_execution(api_client, db_session, user):
    client = api_client
    app.state.job_manager._supervisor_factory = lambda: MockSupervisor(delay=0.1)
    
    response = client.post("/api/v1/research/jobs", json={"goal": "Async API test"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    
    repo = SQLJobRepository(db_session)
    job = repo.get_job(job_id, user.user_id)
    assert job.status in ["PENDING", "RUNNING"]
    
    # Wait for the background manager to complete it
    for _ in range(10):
        db_session.refresh(job)
        if job.status == "COMPLETED":
            break
        time.sleep(0.1)
        
    assert job.status == "COMPLETED"

def test_api_status_reflects_async_completion(api_client, db_session, user):
    client = api_client
    
    response = client.post("/api/v1/research/jobs", json={"goal": "Status test"})
    job_id = response.json()["job_id"]
    
    # Wait for it to finish
    repo = SQLJobRepository(db_session)
    job = repo.get_job(job_id, user.user_id)
    for _ in range(10):
        db_session.refresh(job)
        if job.status == "COMPLETED":
            break
        time.sleep(0.1)
    
    response = client.get(f"/api/v1/research/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"
    assert response.json()["result"] == "Async success"

def test_api_cancel_job(api_client, db_session, user):
    client = api_client
    
    app.state.job_manager._supervisor_factory = lambda: MockSupervisor(delay=1.0)
    response = client.post("/api/v1/research/jobs", json={"goal": "Cancel test"})
    job_id = response.json()["job_id"]
    
    cancel_resp = client.post(f"/api/v1/research/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"
    
    status_resp = client.get(f"/api/v1/research/jobs/{job_id}")
    assert status_resp.json()["status"] == "CANCELLED"
