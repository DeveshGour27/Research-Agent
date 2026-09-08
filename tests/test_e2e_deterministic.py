import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from uuid import uuid4
import datetime
import tempfile
import os
import json

from app.main_api import app
from app.db.database import get_db, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import User, UserSession, Conversation, Message, Job
from app.services.job_manager import AsyncJobManager
from app.agent.contracts import AgentResult, AgentRequest

class MockSupervisor:
    def __init__(self, output="TEST_RESPONSE_123", success=True, error=None):
        self.output = output
        self.success = success
        self.error = error

    def execute(self, request: AgentRequest) -> AgentResult:
        if not self.success and self.error:
            raise Exception(self.error)
        return AgentResult(
            request=request,
            state=None,  # Not needed for test
            output=self.output,
            success=self.success,
            error=self.error,
        )

from sqlalchemy.pool import NullPool

@pytest.fixture(scope="module")
def engine():
    fd, path = tempfile.mkstemp()
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    os.close(fd)
    try:
        os.remove(path)
    except:
        pass

@pytest.fixture(scope="module")
def TestingSessionLocal(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def override_db(TestingSessionLocal):
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_user_data(db: Session, email: str, session_id: str):
    user_id = f"user_{uuid4().hex[:10]}"
    user = User(user_id=user_id, email=email, email_verified=True)
    db.add(user)
    
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    session = UserSession(session_id=session_id, user_id=user_id, expires_at=expires)
    db.add(session)
    db.commit()
    return {"user_id": user_id, "session_id": session_id}

@pytest.fixture
def test_data(engine, TestingSessionLocal):
    db = TestingSessionLocal()
    for tbl in reversed(Base.metadata.sorted_tables):
        db.execute(tbl.delete())
    db.commit()
    
    data = create_user_data(db, "a@example.com", "sess_A")
    db.close()
    return data

@pytest.fixture
def client(engine, TestingSessionLocal):
    from app.db import database as db_module
    old_env_db = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = str(engine.url)
    old_engine = db_module.engine
    old_session_local = db_module.SessionLocal
    db_module.engine = engine
    db_module.SessionLocal = TestingSessionLocal
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
            
    from app.main_api import app as fastapi_app
    fastapi_app.dependency_overrides[db_module.get_db] = override_get_db
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Use real JobManager but with MockSupervisor
    job_manager = AsyncJobManager(
        session_factory=TestingSessionLocal,
        supervisor_factory=lambda: MockSupervisor(output="TEST_RESPONSE_123"),
        job_heartbeat_interval_seconds=10,
        job_recovery_poll_interval_seconds=15,
        worker_id="test_worker"
    )
    # Override default_supervisor_factory so that when lifespan runs, it gets our mock
    import app.main_api as main_api_module
    old_main_session = main_api_module.SessionLocal
    main_api_module.SessionLocal = TestingSessionLocal
    original_factory = main_api_module.default_supervisor_factory
    main_api_module.default_supervisor_factory = lambda hitl: MockSupervisor(output="TEST_RESPONSE_123")
    
    with TestClient(fastapi_app) as c:
        # After lifespan, job_manager is created. We still need to mock submit_job for sqlite threading
        job_manager = fastapi_app.state.job_manager
        original_submit = job_manager.submit_job
        def sync_submit(job_id, user_id, goal):
            loop.run_until_complete(job_manager._run_job_with_lifecycle(job_id, user_id, goal))
        job_manager.submit_job = sync_submit
        
        yield c

    main_api_module.SessionLocal = old_main_session
    main_api_module.default_supervisor_factory = original_factory
    loop.run_until_complete(job_manager.shutdown())
    loop.close()
    fastapi_app.dependency_overrides.clear()
    if hasattr(fastapi_app.state, 'job_manager'):
        del fastapi_app.state.job_manager
    if hasattr(fastapi_app.state, 'hitl_service'):
        from app.hitl.service import HITLService
        from app.hitl.policy import HITLPolicy
        fastapi_app.state.hitl_service = HITLService(old_session_local, HITLPolicy())
    db_module.engine = old_engine
    db_module.SessionLocal = old_session_local
    if old_env_db is not None:
        os.environ["DATABASE_URL"] = old_env_db
    else:
        os.environ.pop("DATABASE_URL", None)

def test_chat_message_to_assistant_response(client, test_data):
    cookies = {"session_id": test_data["session_id"]}
    
    # 1. Create chat
    res = client.post("/api/v1/chats", cookies=cookies)
    assert res.status_code == 200
    chat_id = res.json()["chat_id"]
    
    # 2. Send message
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "What is 2+2?"}, cookies=cookies)
    assert res.status_code in (200, 202)
    job_id = res.json()["job_id"]
    
    # 3. Wait for job to complete (JobManager runs in background)
    import time
    for _ in range(50): # wait up to 5s
        res = client.get(f"/api/v1/jobs/{job_id}", cookies=cookies)
        if res.json()["status"] == "COMPLETED":
            break
        time.sleep(0.1)
    
    assert res.json()["status"] == "COMPLETED"
    
    # 4. Stream SSE and check for JOB_COMPLETED payload
    events_raw = []
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events/stream", cookies=cookies) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events_raw.append(json.loads(line[6:]))
                
    job_completed_events = [e for e in events_raw if e["event_type"] == "JOB_COMPLETED"]
    assert len(job_completed_events) == 1
    
    # Ensure payload contains output (the fix we applied)
    payload = job_completed_events[0].get("payload", {})
    assert payload.get("output") == "TEST_RESPONSE_123"
    
    # 5. Check if chat history contains TEST_RESPONSE_123
    res = client.get(f"/api/v1/chats/{chat_id}", cookies=cookies)
    assert res.status_code == 200
    chat = res.json()
    messages = chat["messages"]
    
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "TEST_RESPONSE_123"
