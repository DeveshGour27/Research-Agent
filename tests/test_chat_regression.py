import pytest
import asyncio
from httpx import AsyncClient
from app.main_api import app
from app.db.models import Job, Message
from app.db.database import get_db, SessionLocal
from app.agent.contracts import AgentResult, AgentRequest
from app.agent.state import AgentState
from app.services.job_manager import AsyncJobManager

class DummySupervisor:
    def execute(self, request: AgentRequest):
        state = AgentState(finished=True, final_answer="The capital of France is Paris.")
        return AgentResult(
            request=request,
            state=state,
            output="The capital of France is Paris.",
            success=True,
            context=request.context,
        )

def test_job_persistence_preserves_actual_output():
    async def run_test():
        db = SessionLocal()
        from app.db.repository import SQLJobRepository
        import uuid
        user_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())
        
        repo = SQLJobRepository(db)
        from app.db.models import User, Conversation
        db.add(User(user_id=user_id, username=f"test_{uuid.uuid4().hex[:8]}", email=f"test_{uuid.uuid4().hex[:8]}@example.com"))
        db.add(Conversation(chat_id=chat_id, user_id=user_id, title="Test chat"))
        db.commit()
        
        repo.create_job(job_id=job_id, user_id=user_id, goal="What is the capital of France?")
        db.add(Message(chat_id=chat_id, job_id=job_id, role="user", content="What is the capital of France?"))
        db.commit()
        
        def dummy_factory():
            return DummySupervisor()
        
        job_manager = AsyncJobManager(session_factory=SessionLocal, supervisor_factory=dummy_factory)
        job_manager.submit_job(job_id, user_id, "What is the capital of France?")
        
        await asyncio.sleep(0.5)
        await job_manager.shutdown()
        
        job = db.query(Job).filter(Job.job_id == job_id).first()
        assert job.status == "COMPLETED"
        assert job.result == "The capital of France is Paris."
        
        msg = db.query(Message).filter(Message.job_id == job_id, Message.role == "assistant").first()
        assert msg is not None
        assert msg.content == "The capital of France is Paris."
        assert msg.content != "Success"
        db.close()
    
    asyncio.run(run_test())

class DummySupervisorFailed:
    def execute(self, request: AgentRequest):
        return AgentResult(
            request=request,
            state=AgentState(finished=True),
            output=None,
            success=False,
            context=request.context,
            error=Exception("LLM Failed")
        )

def test_job_persistence_preserves_failure_state():
    async def run_test():
        db = SessionLocal()
        from app.db.repository import SQLJobRepository
        import uuid
        user_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        chat_id = str(uuid.uuid4())
        
        repo = SQLJobRepository(db)
        from app.db.models import User, Conversation
        db.add(User(user_id=user_id, username=f"test_{uuid.uuid4().hex[:8]}", email=f"test_{uuid.uuid4().hex[:8]}@example.com"))
        db.add(Conversation(chat_id=chat_id, user_id=user_id, title="Test chat"))
        db.commit()
        
        repo.create_job(job_id=job_id, user_id=user_id, goal="Fail me")
        db.add(Message(chat_id=chat_id, job_id=job_id, role="user", content="Fail me"))
        db.commit()
        
        def dummy_factory():
            return DummySupervisorFailed()
        
        job_manager = AsyncJobManager(session_factory=SessionLocal, supervisor_factory=dummy_factory)
        job_manager.submit_job(job_id, user_id, "Fail me")
        
        await asyncio.sleep(0.5)
        await job_manager.shutdown()
        
        job = db.query(Job).filter(Job.job_id == job_id).first()
        assert job.status == "FAILED"
        assert "LLM Failed" in str(job.error_message)
        
        msg = db.query(Message).filter(Message.job_id == job_id, Message.role == "assistant").first()
        assert msg is None
        db.close()
    
    asyncio.run(run_test())
