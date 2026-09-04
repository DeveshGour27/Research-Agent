import asyncio
import json
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from app.main_api import app
from app.db.database import SessionLocal, engine
from app.db.models import User, Base
from app.api.auth import get_current_user
from app.agent.contracts import AgentResult

Base.metadata.create_all(bind=engine)

def override_get_current_user():
    class MockUser:
        def __init__(self, uid):
            self.user_id = uid
    return MockUser("test_user_id_123")

app.dependency_overrides[get_current_user] = override_get_current_user

async def test_e2e_success():
    print("Starting e2e success test...")
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Mock Supervisor
        def mock_execute(self, req):
            return AgentResult(request=req, state={}, success=True, output="TEST_RESPONSE_123")
        
        with patch("app.agent.supervisor.Supervisor.execute", mock_execute):
            resp = await client.post("/api/v1/chats")
            chat_id = resp.json()["chat_id"]
            
            resp = await client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "What is 2 + 2?"})
            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}"
            job_id = resp.json()["job_id"]
            print(f"✅ POST returned 202, job_id={job_id}")
            
            db = SessionLocal()
            from app.db.models import Job, Message
            job = db.query(Job).filter(Job.job_id == job_id).first()
            assert job is not None
            print("✅ Job created in DB")
            db.close()
            
            completed_event = None
            async with client.stream("GET", f"/api/v1/jobs/{job_id}/events/stream") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("event_type") == "JOB_COMPLETED":
                            completed_event = data
                            break
                        if data.get("event_type") == "JOB_FAILED":
                            print("Job failed:", data)
                            break

            assert completed_event is not None
            assert completed_event["payload"]["output"] == "TEST_RESPONSE_123"
            print("✅ SSE received JOB_COMPLETED with TEST_RESPONSE_123")
            
            db = SessionLocal()
            messages = db.query(Message).filter(Message.chat_id == chat_id, Message.role == "assistant").all()
            assert len(messages) == 1
            assert messages[0].content == "TEST_RESPONSE_123"
            assert messages[0].job_id == job_id
            print("✅ DB Message verified: TEST_RESPONSE_123")
            db.close()

async def test_e2e_no_output():
    print("\nStarting e2e no-output test...")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        def mock_execute_fail(self, req):
            return AgentResult(request=req, state={}, success=True, output=None)
        
        with patch("app.agent.supervisor.Supervisor.execute", mock_execute_fail):
            resp = await client.post("/api/v1/chats")
            chat_id = resp.json()["chat_id"]
            
            resp = await client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "Fail this."})
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]
            
            failed_event = None
            async with client.stream("GET", f"/api/v1/jobs/{job_id}/events/stream") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("event_type") == "JOB_FAILED":
                            failed_event = data
                            break

            assert failed_event is not None
            assert failed_event["payload"]["error"] == "Agent completed without providing an output."
            print("✅ SSE received JOB_FAILED for empty output")

async def main():
    async with app.router.lifespan_context(app):
        await test_e2e_success()
        await test_e2e_no_output()
        print("ALL TESTS PASSED")

if __name__ == "__main__":
    asyncio.run(main())
