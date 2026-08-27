import pytest
import asyncio
from unittest.mock import MagicMock, patch
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.models import Base
from app.db.repository import SQLJobRepository
from app.services.job_manager import AsyncJobManager
from app.exceptions import InvalidStateTransitionError, ToolExecutionError
from app.llm.gateway import ModelGateway, ModelRouter
from app.llm.models import ModelProfile, ModelCapability, ModelRequest, TaskType
from app.llm.openai_provider import OpenAIProvider
from app.tools.registry import ToolRegistry
from app.tools.base import BaseTool
from app.agent.supervisor import Supervisor
from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext

@pytest.fixture
def sqlite_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def test_job_recovery(sqlite_session_factory):
    async def _run():
        with sqlite_session_factory() as session:
            repo = SQLJobRepository(session)
            repo.create_job("job-1", "user-1", "Goal")
            repo.claim_job("job-1", "worker-1")
        
        manager = AsyncJobManager(
            sqlite_session_factory, 
            lambda: MagicMock(), 
            job_stale_after_seconds=0,
            job_recovery_poll_interval_seconds=1
        )
        
        with sqlite_session_factory() as session:
            repo = SQLJobRepository(session)
            recovered = repo.recover_stale_jobs(0, 3)
            assert recovered == ["job-1"]
            
            job = repo.get_job("job-1", "user-1")
            assert job.status == "PENDING"
            assert job.worker_id is None
        
        await manager.shutdown()
    asyncio.run(_run())

def test_job_retry_limit(sqlite_session_factory):
    async def _run():
        with sqlite_session_factory() as session:
            repo = SQLJobRepository(session)
            job = repo.create_job("job-2", "user-1", "Goal")
            repo.claim_job("job-2", "worker-1")
            from sqlalchemy import update
            from app.db.models import Job
            session.execute(update(Job).where(Job.job_id == "job-2").values(attempt_count=3))
            session.commit()
            
            recovered = repo.recover_stale_jobs(0, 3)
            assert recovered == []
            
            job = repo.get_job("job-2", "user-1")
            assert job.status == "FAILED"
            assert "exceeded maximum execution attempts" in job.error_message
    asyncio.run(_run())

def test_provider_fallback():
    # Setup mocks
    p1 = MagicMock()
    p1.generate.side_effect = Exception("Crash")
    p2 = MagicMock()
    p2.generate.return_value = MagicMock(provider="provider2", model="m2", input_tokens=10, output_tokens=10, total_tokens=20)
    
    router = MagicMock()
    profile2 = MagicMock(provider="provider2", model_id="m2", default_temperature=0.7, max_output_tokens=100)
    profile1 = MagicMock(provider="provider1", model_id="m1", default_temperature=0.7, max_output_tokens=100)
    router.select_profiles.return_value = [profile1, profile2]
    
    gateway = ModelGateway(router, {"provider1": p1, "provider2": p2}, max_retries=1)
    
    req = ModelRequest(messages=[], required_capabilities=set([ModelCapability.TEXT_GENERATION]), task_type=TaskType.GENERAL)
    res = gateway.generate(req)
    assert res.provider == "provider2"

def test_tool_timeout_handling():
    class SlowTool(BaseTool):
        name = "slow"
        description = "slow"
        def execute(self, **kwargs):
            return "done"
            
    registry = ToolRegistry(include_mcp=False)
    registry.register(SlowTool())
    
    from app.llm.models import ToolCall
    call = ToolCall(id="1", name="slow", arguments={})
    
    with patch("concurrent.futures.ThreadPoolExecutor.submit") as mock_submit:
        mock_future = MagicMock()
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_submit.return_value = mock_future
        
        res = registry.execute(call)
        assert res.is_error is True
        assert "timed out" in res.content

def test_job_cancellation(sqlite_session_factory):
    async def _run():
        import asyncio
        slow_supervisor = MagicMock()
        def _slow_exec(req):
            time.sleep(1.0)
            return MagicMock()
        slow_supervisor.execute = _slow_exec
        manager = AsyncJobManager(sqlite_session_factory, lambda: slow_supervisor)
        
        with sqlite_session_factory() as session:
            repo = SQLJobRepository(session)
            repo.create_job("job-cancel", "user-1", "Goal")
            
        manager.submit_job("job-cancel", "user-1", "Goal")
        await asyncio.sleep(0.1) # yield to start
        
        await manager.cancel_job("job-cancel", "user-1")
        
        with sqlite_session_factory() as session:
            repo = SQLJobRepository(session)
            job = repo.get_job("job-cancel", "user-1")
            assert job.status == "CANCELLED"
            
        await manager.shutdown()
    asyncio.run(_run())
