"""Tests for Phase 7.8 — Hardening and Race Conditions.

Covers:
- Pending job restart recovery
- Concurrent cancellation races
- Worker cancellation races
"""

import asyncio
import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.models import User, Job
from app.db.repository import SQLJobRepository
from app.main_api import app
from app.api.auth import get_current_user
from app.services.job_manager import AsyncJobManager
from app.exceptions import InvalidStateTransitionError, AgentCancellationError
from unittest.mock import patch

def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

import tempfile
import os

_temp_db_fd, _temp_db_path = tempfile.mkstemp(suffix=".db")

_engine = create_engine(
    f"sqlite:///{_temp_db_path}",
    connect_args={"check_same_thread": False},
)

from sqlalchemy import event
@event.listens_for(_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_db():
    yield
    os.close(_temp_db_fd)
    try:
        os.unlink(_temp_db_path)
    except OSError:
        pass

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

def test_pending_job_restart_recovery(db_session, test_user_a):
    """Test that a PENDING job left in the database is recovered by the job manager."""
    async def run_test():
        # 1. Simulate a PENDING job left behind from a previous process
        db_session.add(Job(job_id="pending_job_1", user_id=test_user_a.user_id, goal="test", status="PENDING"))
        db_session.add(Job(job_id="pending_job_2", user_id=test_user_a.user_id, goal="test", status="PENDING"))
        db_session.commit()

        # 2. Create a Job Manager (simulating process start)
        class DummySupervisor:
            def execute(self, req):
                return type("Result", (), {"output": "dummy"})()

        manager = AsyncJobManager(
            session_factory=_TestingSessionLocal,
            supervisor_factory=DummySupervisor,
            job_recovery_poll_interval_seconds=1, # fast poll
        )
        
        # Wait for recovery loop to pick up the jobs
        await asyncio.sleep(1.5)
        
        try:
            repo = SQLJobRepository(db_session)
            j1 = repo.get_job("pending_job_1", test_user_a.user_id)
            j2 = repo.get_job("pending_job_2", test_user_a.user_id)
            
            # Because supervisor is dummy, it completes successfully
            assert j1.status == "COMPLETED"
            assert j2.status == "COMPLETED"
        finally:
            await manager.shutdown()
            
    asyncio.run(run_test())

def test_cancellation_race_with_completion(db_session, test_user_a):
    """Test cancellation racing with completion."""
    async def run_test():
        repo = SQLJobRepository(db_session)
        job = repo.create_job(job_id="race_job_1", user_id=test_user_a.user_id, goal="test")
        repo.claim_job("race_job_1", "worker_1")
        
        manager = AsyncJobManager(
            session_factory=_TestingSessionLocal,
            supervisor_factory=lambda: None,
        )
        
        # Simulate worker completing the job BEFORE cancel
        repo.update_job_status("race_job_1", test_user_a.user_id, status="COMPLETED")
        
        # Simulate API cancel_job() being called right after
        # Should NOT throw InvalidStateTransitionError
        await manager.cancel_job("race_job_1", test_user_a.user_id)
        
        # Verify job is still COMPLETED
        j = repo.get_job("race_job_1", test_user_a.user_id)
        assert j.status == "COMPLETED"
        
        await manager.shutdown()
        
    asyncio.run(run_test())

def test_cancellation_worker_race(db_session, test_user_a):
    """Test worker cancellation exception handling when job is already cancelled."""
    async def run_test():
        repo = SQLJobRepository(db_session)
        job = repo.create_job(job_id="race_job_2", user_id=test_user_a.user_id, goal="test")
        repo.claim_job("race_job_2", "worker_2")
        
        class DummySupervisor:
            def execute(self, req):
                raise AgentCancellationError("Cancelled by user")
                
        manager = AsyncJobManager(
            session_factory=_TestingSessionLocal,
            supervisor_factory=DummySupervisor,
            worker_id="worker_2"
        )
        
        # Simulate API already marking the job CANCELLED
        repo.update_job_status("race_job_2", test_user_a.user_id, status="CANCELLED")
        
        # Simulate the job being run (this will raise AgentCancellationError)
        await manager._run_job_with_lifecycle("race_job_2", test_user_a.user_id, "test")
        
        # Verify the job is still CANCELLED and not FAILED
        j = repo.get_job("race_job_2", test_user_a.user_id)
        assert j.status == "CANCELLED"
        
        await manager.shutdown()
        
    asyncio.run(run_test())
