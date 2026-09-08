import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from uuid import uuid4
import datetime
import tempfile
import os

from app.main_api import app
from app.db.database import get_db, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import User, UserSession, Conversation, Message, Job

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
            raise ValueError('not found')
        def submit_job(self, job_id, user_id, goal):
            # immediately complete job to simulate execution
            from app.db.repository import SQLJobRepository
            db = override_db
            repo = SQLJobRepository(db)
            repo.claim_job(job_id=job_id, worker_id="test")
            repo.update_job_result(job_id=job_id, user_id=user_id, status="COMPLETED", result="Fake result", worker_id="test")
            db.commit()
    app.state.job_manager = MockJobManager()
    
    yield TestClient(app)
    app.dependency_overrides.clear()
    if hasattr(app.state, 'job_manager'):
        del app.state.job_manager

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
    
    data_a = create_user_data(db, "a@example.com", "sess_A")
    data_b = create_user_data(db, "b@example.com", "sess_B")
    db.close()
    return {"A": data_a, "B": data_b}

def test_chat_creation_and_listing(client, test_data):
    # User A can create a chat
    cookies = {"session_id": test_data["A"]["session_id"]}
    res = client.post("/api/v1/chats", cookies=cookies)
    assert res.status_code == 200
    chat_a1 = res.json()["chat_id"]
    
    # User B can create a chat
    cookies_b = {"session_id": test_data["B"]["session_id"]}
    res = client.post("/api/v1/chats", cookies=cookies_b)
    assert res.status_code == 200
    chat_b1 = res.json()["chat_id"]
    
    # User A sees A's chats
    res = client.get("/api/v1/chats", cookies=cookies)
    assert res.status_code == 200
    chats = res.json()
    assert len(chats) == 1
    assert chats[0]["chat_id"] == chat_a1
    
    # User A does not see B's chats
    # (Verified by the fact len(chats) == 1 and it is chat_a1)

def test_chat_read(client, test_data):
    # A can read A's chat
    cookies = {"session_id": test_data["A"]["session_id"]}
    res = client.post("/api/v1/chats", cookies=cookies)
    chat_id = res.json()["chat_id"]
    
    res = client.get(f"/api/v1/chats/{chat_id}", cookies=cookies)
    assert res.status_code == 200
    
    # B cannot read A's chat
    cookies_b = {"session_id": test_data["B"]["session_id"]}
    res = client.get(f"/api/v1/chats/{chat_id}", cookies=cookies_b)
    assert res.status_code == 404

def test_chat_delete(client, test_data):
    cookies = {"session_id": test_data["A"]["session_id"]}
    res = client.post("/api/v1/chats", cookies=cookies)
    chat_id = res.json()["chat_id"]
    
    # B cannot delete A's chat
    cookies_b = {"session_id": test_data["B"]["session_id"]}
    res = client.delete(f"/api/v1/chats/{chat_id}", cookies=cookies_b)
    assert res.status_code == 404
    
    # A can delete A's chat
    res = client.delete(f"/api/v1/chats/{chat_id}", cookies=cookies)
    assert res.status_code == 204
    
    res = client.get(f"/api/v1/chats/{chat_id}", cookies=cookies)
    assert res.status_code == 404

def test_message_creation_and_job_association(client, test_data):
    cookies = {"session_id": test_data["A"]["session_id"]}
    res = client.post("/api/v1/chats", cookies=cookies)
    chat_id = res.json()["chat_id"]
    
    # B cannot send a message to A's chat
    cookies_b = {"session_id": test_data["B"]["session_id"]}
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "hello"}, cookies=cookies_b)
    assert res.status_code == 404
    
    # A can send a message
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "research something"}, cookies=cookies)
    assert res.status_code in (200, 202)
    data = res.json()
    assert "job_id" in data
    job_id = data["job_id"]
    
    # verify job belongs to A
    res = client.get(f"/api/v1/jobs/{job_id}", cookies=cookies)
    assert res.status_code == 200
    
    # B cannot access A's job
    res = client.get(f"/api/v1/jobs/{job_id}", cookies=cookies_b)
    assert res.status_code == 404

def test_full_flow(client, test_data):
    cookies = {"session_id": test_data["A"]["session_id"]}
    res = client.post("/api/v1/chats", cookies=cookies)
    chat_id = res.json()["chat_id"]
    
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "do a full test"}, cookies=cookies)
    assert res.status_code in (200, 202)
    job_id = res.json()["job_id"]
    
    # reload chat
    res = client.get(f"/api/v1/chats/{chat_id}", cookies=cookies)
    assert res.status_code == 200
    chat = res.json()
    assert chat["title"] == "do a full test"
    
    messages = chat["messages"]
    assert len(messages) == 2 # 1 user + 1 assistant
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "do a full test"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Fake result"
    assert messages[0]["job_id"] == job_id
    assert messages[1]["job_id"] == job_id
