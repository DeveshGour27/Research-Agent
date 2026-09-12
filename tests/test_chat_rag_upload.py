import io
import uuid
import pytest
import fitz
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main_api import app
from app.api.auth import get_current_user
from app.db.database import Base, get_db
from app.db.models import User, Conversation, Message, Job
from app.retriever import Retriever

# In-memory database for testing
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

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

TEST_USER_ID = f"rag_user_{uuid.uuid4().hex[:8]}"

def override_get_current_user():
    return User(user_id=TEST_USER_ID, username="rag_tester", email="rag_tester@example.com")

def create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    session.add(User(user_id=TEST_USER_ID, username="rag_tester", email="rag_tester@example.com"))
    session.commit()
    session.close()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield
    app.dependency_overrides.clear()

def test_upload_pdf_for_rag():
    client = TestClient(app)
    session = TestingSessionLocal()
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    session.add(Conversation(chat_id=chat_id, user_id=TEST_USER_ID, title="RAG Test Chat"))
    session.commit()
    session.close()

    pdf_content = "Superconductors exhibit zero electrical resistance and expulsion of magnetic fields."
    pdf_bytes = create_sample_pdf(pdf_content)

    # Upload PDF
    response = client.post(
        f"/api/v1/chats/{chat_id}/documents",
        files={"file": ("superconductivity.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "indexed"
    assert data["filename"] == "superconductivity.pdf"
    assert data["chunks"] >= 1
    assert data["page_count"] == 1

    # Verify retrieval
    retriever = Retriever()
    results = retriever.retrieve("zero electrical resistance", user_id=TEST_USER_ID)
    assert len(results) > 0
    matched_texts = [r.metadata.get("text", "") for r in results]
    assert any("Superconductors exhibit zero electrical resistance" in t for t in matched_texts)

def test_upload_non_pdf_fails():
    client = TestClient(app)
    session = TestingSessionLocal()
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    session.add(Conversation(chat_id=chat_id, user_id=TEST_USER_ID, title="Non PDF Test"))
    session.commit()
    session.close()

    response = client.post(
        f"/api/v1/chats/{chat_id}/documents",
        files={"file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF documents (.pdf) are supported" in response.json()["detail"]

def test_upload_to_unauthorized_chat_fails():
    client = TestClient(app)
    session = TestingSessionLocal()
    other_user_id = f"other_{uuid.uuid4().hex[:8]}"
    session.add(User(user_id=other_user_id, username="other_user"))
    other_chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    session.add(Conversation(chat_id=other_chat_id, user_id=other_user_id, title="Other's Chat"))
    session.commit()
    session.close()

    pdf_bytes = create_sample_pdf("Secret other user notes.")
    response = client.post(
        f"/api/v1/chats/{other_chat_id}/documents",
        files={"file": ("secret.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 404

def test_send_message_with_attached_documents():
    client = TestClient(app)
    session = TestingSessionLocal()
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    session.add(Conversation(chat_id=chat_id, user_id=TEST_USER_ID, title="Attached Doc Chat"))
    session.commit()
    session.close()

    response = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={
            "content": "What is the critical temperature?",
            "attached_documents": ["superconductivity.pdf"],
        },
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    job_id = data["job_id"]

    # Verify message and job in database
    session = TestingSessionLocal()
    msg = session.query(Message).filter(Message.message_id == data["message_id"]).first()
    assert msg is not None
    assert "[Attached Document: superconductivity.pdf]" in msg.content
    assert "What is the critical temperature?" in msg.content

    job = session.query(Job).filter(Job.job_id == job_id).first()
    assert job is not None
    assert "[Attached Document: superconductivity.pdf]" in job.goal
    session.close()
