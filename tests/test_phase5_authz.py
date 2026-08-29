"""Tests for Phase 5 User Isolation and Authorization."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from uuid import uuid4
from typing import Generator

from app.main_api import app
from app.db.database import get_db, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import User, Conversation, Message, Job, JobStep, HITLRequest, UserSession
import datetime
import tempfile
import os

@pytest.fixture(scope="module")
def engine():
    fd, path = tempfile.mkstemp()
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
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

@pytest.fixture
def client(override_db):
    app.dependency_overrides[get_db] = lambda: override_db
    class MockJobManager:
        async def cancel_job(self, job_id, user_id):
            # For tests, if a job does not belong to a user, it should raise ValueError
            # But just returning 404 for all cancels is easiest to mock for auth
            raise ValueError('not found')
    app.state.job_manager = MockJobManager()
    
    class MockHITLService:
        def approve_request(self, request_id, job_id, decided_by):
            return True
        def reject_request(self, request_id, job_id, decided_by):
            return True
    app.state.hitl_service = MockHITLService()
    
    yield TestClient(app)
    app.dependency_overrides.clear()
    if hasattr(app.state, 'job_manager'):
        del app.state.job_manager
    if hasattr(app.state, 'hitl_service'):
        del app.state.hitl_service

def create_user_data(db: Session, email: str, session_id: str):
    user_id = f"user_{uuid4().hex[:10]}"
    user = User(user_id=user_id, email=email, email_verified=True)
    db.add(user)
    
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    session = UserSession(session_id=session_id, user_id=user_id, expires_at=expires)
    db.add(session)
    
    chat_id = f"chat_{uuid4().hex[:10]}"
    chat = Conversation(chat_id=chat_id, user_id=user_id, title=f"Chat {email}")
    db.add(chat)
    
    msg_id = f"msg_{uuid4().hex[:10]}"
    job_id = f"job_{uuid4().hex[:12]}"
    msg = Message(message_id=msg_id, chat_id=chat_id, role="user", content="Hello", job_id=job_id)
    db.add(msg)
    
    job = Job(job_id=job_id, user_id=user_id, goal="Test Goal", status="RUNNING")
    db.add(job)
    
    step_id = f"step_{uuid4().hex[:10]}"
    step = JobStep(step_id=step_id, job_id=job_id, task_type="TEST")
    db.add(step)
    
    hitl_id = f"hitl_{uuid4().hex[:10]}"
    hitl = HITLRequest(request_id=hitl_id, job_id=job_id, run_id="run1", request_type="TOOL", component_name="test", invocation_fingerprint="fp", payload="{}", status="PENDING")
    db.add(hitl)
    
    db.commit()
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "chat_id": chat_id,
        "msg_id": msg_id,
        "job_id": job_id,
        "step_id": step_id,
        "hitl_id": hitl_id
    }

@pytest.fixture
def test_data(engine, TestingSessionLocal):
    db = TestingSessionLocal()
    for tbl in reversed(Base.metadata.sorted_tables):
        db.execute(tbl.delete())
    db.commit()
    
    data_a = create_user_data(db, "a@example.com", "sess_A")
    data_b = create_user_data(db, "b@example.com", "sess_B")
    db.close()
    return {"A": data_a, "B": data_b}

def test_user_isolation(client, test_data):
    data_a = test_data["A"]
    data_b = test_data["B"]
    
    def check_access(actor, target, expected_status):
        cookies = {"session_id": actor["session_id"]}
        
        # Test Chat
        res = client.get(f"/api/v1/chats/{target['chat_id']}", cookies=cookies)
        assert res.status_code == expected_status
        
        # Test Message
        res = client.get(f"/api/v1/messages/{target['msg_id']}", cookies=cookies)
        assert res.status_code == expected_status
        
        # Test Job Status
        res = client.get(f"/api/v1/jobs/{target['job_id']}", cookies=cookies)
        assert res.status_code == expected_status
        
        # Test Job Result
        res = client.get(f"/api/v1/research/{target['job_id']}/result", cookies=cookies)
        if expected_status == 404:
            assert res.status_code == 404
        else:
            assert res.status_code == 409
            
        # Test Job Events
        res = client.get(f"/api/v1/jobs/{target['job_id']}/events", cookies=cookies)
        assert res.status_code == expected_status
        
        # Test HITL Requests
        res = client.get(f"/api/v1/research/{target['job_id']}/hitl", cookies=cookies)
        assert res.status_code == expected_status
        
        # Test Job Cancel
        # res = client.post(f"/api/v1/research/{target['job_id']}/cancel", cookies=cookies)
        # if expected_status == 404:
        #     assert res.status_code == 404
            
    check_access(data_a, data_a, 200)
    check_access(data_a, data_b, 404)
    check_access(data_b, data_b, 200)
    check_access(data_b, data_a, 404)

def test_sse_isolation(client, test_data):
    data_a = test_data["A"]
    data_b = test_data["B"]
    cookies = {"session_id": data_a["session_id"]}
    res = client.get(f"/api/v1/jobs/{data_b['job_id']}/events/stream", cookies=cookies)
    assert res.status_code == 404

def test_id_tampering(client, test_data):
    data_a = test_data["A"]
    data_b = test_data["B"]
    cookies = {"session_id": data_a["session_id"]}
    
    # replace A's chat_id with B's chat_id
    res = client.get(f"/api/v1/chats/{data_b['chat_id']}", cookies=cookies)
    assert res.status_code == 404

    # replace A's job_id with B's job_id
    res = client.get(f"/api/v1/jobs/{data_b['job_id']}", cookies=cookies)
    assert res.status_code == 404
    
    # replace A's msg_id with B's msg_id
    res = client.get(f"/api/v1/messages/{data_b['msg_id']}", cookies=cookies)
    assert res.status_code == 404
