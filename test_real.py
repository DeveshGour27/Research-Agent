import asyncio
import json
import httpx
import uuid
import datetime

from app.db.database import SessionLocal, engine
from app.db.models import User, UserSession, Base

Base.metadata.create_all(bind=engine)

def create_test_session():
    db = SessionLocal()
    user = db.query(User).filter(User.email == "test@example.com").first()
    if not user:
        user = User(
            user_id="test_user_id_123",
            email="test@example.com",
            username="Test User"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    session_id = uuid.uuid4().hex
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    
    sess = UserSession(
        session_id=session_id,
        user_id=user.user_id,
        expires_at=expires
    )
    db.add(sess)
    db.commit()
    db.close()
    return session_id

async def test_e2e_success():
    print("Starting e2e success test...")
    session_id = create_test_session()
    
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", cookies={"session_id": session_id}, timeout=120.0) as client:
        resp = await client.post("/api/v1/chats")
        if resp.status_code != 200:
            print("Failed to create chat:", resp.status_code, resp.text)
            return
        chat_id = resp.json()["chat_id"]
        
        # Use actual LLM for real response "What is 2+2?"
        resp = await client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "What is 2 + 2?"})
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}"
        job_id = resp.json()["job_id"]
        print(f"✅ POST returned 202, job_id={job_id}")
        
        completed_event = None
        async with client.stream("GET", f"/api/v1/jobs/{job_id}/events/stream") as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    event_type = data.get("event_type")
                    print("Received event:", event_type)
                    
                    if event_type == "HITL_REQUESTED":
                        # Auto-approve
                        event_id = data.get("event_id") # "hitl:<request_id>:requested"
                        request_id = event_id.split(":")[1]
                        print("Auto-approving HITL request:", request_id)
                        
                        # In the background
                        asyncio.create_task(client.post(f"/api/v1/research/{job_id}/hitl/{request_id}/approve"))
                        
                    elif event_type == "JOB_COMPLETED":
                        completed_event = data
                        break
                    elif event_type == "JOB_FAILED":
                        print("Job Failed:", data)
                        break

        assert completed_event is not None
        output = completed_event["payload"]["output"]
        assert "4" in output or "four" in output.lower() or "Four" in output, f"Expected answer 4, got {output}"
        print(f"✅ SSE received JOB_COMPLETED with correct answer: {output}")

async def main():
    await test_e2e_success()
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    asyncio.run(main())
