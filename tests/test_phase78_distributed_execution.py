"""Tests for Phase 7.8 — Distributed Job Execution & Multi-Instance Safety."""

from __future__ import annotations

import asyncio
import time
import os
import uuid
import datetime
from typing import Generator
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import Job, User
from app.db.repository import SQLJobRepository
from app.config import settings
from app.services.job_manager import AsyncJobManager


@pytest.fixture
def sync_session_factory() -> sessionmaker:
    """Provides a fresh database session factory for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=None,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False)


@pytest.fixture
def setup_user(sync_session_factory: sessionmaker) -> tuple[sessionmaker, str]:
    with sync_session_factory() as session:
        repo = SQLJobRepository(session)
        user = repo.create_user(email="dist@test.com")
        user_id = user.user_id
    return sync_session_factory, user_id


# ----------------------------------------------------------------------
# 1. Configuration & Worker ID
# ----------------------------------------------------------------------
def test_worker_id_generation():
    """Worker ID should generate a UUID hex if not configured."""
    async def run():
        manager1 = AsyncJobManager(sessionmaker(), lambda: None)
        manager2 = AsyncJobManager(sessionmaker(), lambda: None)
        
        assert manager1.worker_id is not None
        assert manager2.worker_id is not None
        assert manager1.worker_id != manager2.worker_id
        await manager1.shutdown()
        await manager2.shutdown()
    asyncio.run(run())

def test_configured_worker_id():
    async def run():
        manager = AsyncJobManager(sessionmaker(), lambda: None, worker_id="custom_worker_123")
        assert manager.worker_id == "custom_worker_123"
        await manager.shutdown()
    asyncio.run(run())


# ----------------------------------------------------------------------
# 3. Repository Atomic Claiming & Heartbeat
# ----------------------------------------------------------------------
class TestRepositoryDistributedOps:
    def test_claim_job_success(self, setup_user):
        """claim_job transitions PENDING -> RUNNING, sets ownership, and increments attempt."""
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            job = repo.create_job(job_id, user_id, "test goal")
            assert job.status == "PENDING"
            assert job.worker_id is None
            assert job.attempt_count == 0
            
            # Action
            claimed = repo.claim_job(job_id, "worker_1")
            assert claimed is True
            
            # Verification
            job = repo.get_job(job_id, user_id)
            assert job.status == "RUNNING"
            assert job.worker_id == "worker_1"
            assert job.started_at is not None
            assert job.heartbeat_at is not None
            assert job.attempt_count == 1

    def test_claim_already_claimed_fails(self, setup_user):
        """Second claim fails gracefully."""
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "test goal")
            
            # First claim succeeds
            assert repo.claim_job(job_id, "worker_1") is True
            
            # Second claim fails
            assert repo.claim_job(job_id, "worker_2") is False
            
            # Ownership belongs to worker 1
            job = repo.get_job(job_id, user_id)
            assert job.worker_id == "worker_1"
            assert job.attempt_count == 1

    def test_heartbeat_requires_ownership(self, setup_user):
        """Heartbeat only succeeds if job is RUNNING and owned by the worker."""
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            repo.claim_job(job_id, "worker_1")
            
            # Worker 1 succeeds
            assert repo.heartbeat_job(job_id, "worker_1") is True
            
            # Worker 2 fails
            assert repo.heartbeat_job(job_id, "worker_2") is False

    def test_stale_recovery(self, setup_user):
        """Stale recovery resets only jobs older than the threshold."""
        sf, user_id = setup_user
        job_id1 = "job_stale_" + uuid.uuid4().hex[:8]
        job_id2 = "job_fresh_" + uuid.uuid4().hex[:8]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            # Create two jobs and claim them
            j1 = repo.create_job(job_id1, user_id, "goal 1")
            j2 = repo.create_job(job_id2, user_id, "goal 2")
            
            repo.claim_job(job_id1, "w1")
            repo.claim_job(job_id2, "w2")
            
            # Fake staleness for job 1 only by modifying heartbeat directly
            # This simulates a crashed worker
            import datetime
            stale_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id1)
                .values(heartbeat_at=stale_cutoff)
            )
            session.commit()
            
            # Action: Recover jobs stale for 60 seconds (max attempts: 3)
            recovered = repo.recover_stale_jobs(60, 3)
            
            assert job_id1 in recovered
            assert job_id2 not in recovered
            
            # Verification of stale job reset
            j1_updated = repo.get_job(job_id1, user_id)
            assert j1_updated.status == "PENDING"
            assert j1_updated.worker_id is None
            assert j1_updated.heartbeat_at is None
            
            # Verification of fresh job untouched
            j2_updated = repo.get_job(job_id2, user_id)
            assert j2_updated.status == "RUNNING"
            assert j2_updated.worker_id == "w2"

    def test_recovered_job_claim_increments_attempt(self, setup_user):
        """Recovered jobs can be claimed and their attempt_count increments."""
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            
            repo.claim_job(job_id, "w1") # Attempt 1
            
            import datetime
            stale_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(heartbeat_at=stale_cutoff)
            )
            session.commit()
            
            repo.recover_stale_jobs(60, 3)
            
            # New claim
            assert repo.claim_job(job_id, "w2") is True
            job = repo.get_job(job_id, user_id)
            assert job.attempt_count == 2
            assert job.worker_id == "w2"

    def test_terminal_update_ownership_enforcement(self, setup_user):
        """A worker cannot overwrite terminal state of a job it no longer owns."""
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            repo.claim_job(job_id, "worker_A")
            
            # Simulate worker_A crashing, and worker_B recovering and claiming
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120))
            )
            session.commit()
            repo.recover_stale_jobs(60, 3)
            repo.claim_job(job_id, "worker_B")
            
            # worker_A wakes up and tries to complete the job
            res = repo.update_job_result(job_id, user_id, status="COMPLETED", result="A's work", worker_id="worker_A")
            
            assert res is None # Operation failed silently
            
            # Verify B still owns it
            job = repo.get_job(job_id, user_id)
            assert job.status == "RUNNING"
            assert job.worker_id == "worker_B"
            assert job.result is None


class TestRepositoryPoisonPillRecovery:
    
    def test_recoverable_stale_job(self, setup_user):
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            repo.claim_job(job_id, "w1")
            
            # Make stale
            import datetime
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120))
            )
            session.commit()
            
            recovered = repo.recover_stale_jobs(60, 3)
            assert job_id in recovered
            
            job = repo.get_job(job_id, user_id)
            assert job.status == "PENDING"
            assert job.attempt_count == 1
            
    def test_exhausted_stale_job(self, setup_user):
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            repo.claim_job(job_id, "w1")
            
            # Exhaust attempt count
            import datetime
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(
                    heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120),
                    attempt_count=3
                )
            )
            session.commit()
            
            recovered = repo.recover_stale_jobs(60, 3)
            # Should not be returned for re-queueing
            assert job_id not in recovered
            
            job = repo.get_job(job_id, user_id)
            assert job.status == "FAILED"
            assert job.attempt_count == 3
            assert job.completed_at is not None
            assert "exceeded maximum execution attempts (3)" in (job.error_message or "")
            
    def test_exhausted_job_cannot_be_recovered_again(self, setup_user):
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            
            # Simulate already failed from exhaustion
            import datetime
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(
                    status="FAILED",
                    heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120),
                    attempt_count=3
                )
            )
            session.commit()
            
            recovered = repo.recover_stale_jobs(60, 3)
            assert job_id not in recovered
            
            job = repo.get_job(job_id, user_id)
            assert job.status == "FAILED"

    def test_poison_pill_simulation(self, setup_user):
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            
            import datetime
            def make_stale():
                session.execute(
                    Job.__table__.update()
                    .where(Job.job_id == job_id)
                    .values(heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120))
                )
                session.commit()
            
            # Attempt 1
            repo.claim_job(job_id, "w1")
            make_stale()
            recovered = repo.recover_stale_jobs(60, 3)
            assert job_id in recovered
            
            # Attempt 2
            repo.claim_job(job_id, "w2")
            make_stale()
            recovered = repo.recover_stale_jobs(60, 3)
            assert job_id in recovered
            
            # Attempt 3
            repo.claim_job(job_id, "w3")
            make_stale()
            recovered = repo.recover_stale_jobs(60, 3)
            assert job_id not in recovered # Exhausted
            
            job = repo.get_job(job_id, user_id)
            assert job.status == "FAILED"
            assert job.attempt_count == 3
            assert job.completed_at is not None

    def test_attempt_count_correctness(self, setup_user):
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            
            job = repo.get_job(job_id, user_id)
            assert job.attempt_count == 0
            
            # Claim increments
            repo.claim_job(job_id, "w1")
            job = repo.get_job(job_id, user_id)
            assert job.attempt_count == 1
            
            # Recover does not increment
            import datetime
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120))
            )
            session.commit()
            repo.recover_stale_jobs(60, 3)
            
            job = repo.get_job(job_id, user_id)
            assert job.attempt_count == 1

    @pytest.mark.skipif(
        "sqlite" in settings.database_url,
        reason="Requires real PostgreSQL connection pool for true concurrency testing."
    )
    def test_concurrent_recovery_safety(self, setup_user):
        """Verify that concurrent recovery requests against an exhausted job only process it once."""
        sf, user_id = setup_user
        job_id = "job_" + uuid.uuid4().hex[:12]
        
        with sf() as session:
            repo = SQLJobRepository(session)
            repo.create_job(job_id, user_id, "goal")
            
            import datetime
            session.execute(
                Job.__table__.update()
                .where(Job.job_id == job_id)
                .values(
                    status="RUNNING",
                    heartbeat_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120),
                    attempt_count=3
                )
            )
            session.commit()
        
        def run_recovery():
            with sf() as thread_session:
                r = SQLJobRepository(thread_session)
                return r.recover_stale_jobs(60, 3)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_recovery) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # It's an exhausted job, so no threads should return it for re-queueing
        for recovered_list in results:
            assert job_id not in recovered_list
            
        with sf() as session:
            job = SQLJobRepository(session).get_job(job_id, user_id)
            assert job.status == "FAILED"
            assert job.attempt_count == 3


# ----------------------------------------------------------------------
# 4. AsyncJobManager Lifecycle & Loops
# ----------------------------------------------------------------------
def test_job_manager_deduplication():
    """JobManager local deduplication prevents multiple identical asyncio tasks."""
    async def run():
        manager = AsyncJobManager(sessionmaker(), lambda: None)
        
        # Pre-populate the internal tracking dict
        manager._running_tasks["job_123"] = asyncio.Future()
        
        # Submitting again should be a no-op
        manager.submit_job("job_123", "usr", "goal")
        
        # Cleanup
        await manager.shutdown()
    asyncio.run(run())


# ----------------------------------------------------------------------
# 7. PostgreSQL Concurrency Integration Test
# ----------------------------------------------------------------------
def test_postgresql_concurrent_claim():
    """Validate true concurrent claim against PostgreSQL if available."""
    async def run():
        db_url = os.getenv("TEST_DATABASE_URL", settings.database_url)
        if "postgresql" not in db_url:
            pytest.skip("Skipping distributed PostgreSQL concurrency test (not using postgres).")
            
        engine = create_engine(db_url, pool_size=5, max_overflow=10)
        Base.metadata.create_all(bind=engine)
        sf = sessionmaker(bind=engine)
        
        user_id = "test_user_pg"
        job_id = "pg_job_" + uuid.uuid4().hex[:12]
        
        # 1. Setup
        with sf() as session:
            repo = SQLJobRepository(session)
            user = repo.get_user(user_id)
            if not user:
                repo.create_user(email="pg@test.com", user_id=user_id)
            repo.create_job(job_id, user_id, "goal")
            
        # 2. Concurrent claims
        async def claim_task(worker_id: str):
            def _claim():
                with sf() as session:
                    repo = SQLJobRepository(session)
                    return repo.claim_job(job_id, worker_id)
            return await asyncio.to_thread(_claim)
            
        results = await asyncio.gather(
            claim_task("worker_A"),
            claim_task("worker_B"),
            claim_task("worker_C"),
        )
        
        # 3. Verification
        # Exactly ONE worker should have succeeded
        success_count = sum(1 for r in results if r is True)
        assert success_count == 1
        
        with sf() as session:
            repo = SQLJobRepository(session)
            job = repo.get_job(job_id, user_id)
            assert job.status == "RUNNING"
            assert job.worker_id in ("worker_A", "worker_B", "worker_C")
            assert job.attempt_count == 1
            
    asyncio.run(run())
