import asyncio
from app.main_api import app
import httpx
import uvicorn
import threading
import time
import json
import sqlite3
import random

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8002)

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(2)

def e2e_test():
    client = httpx.Client(base_url="http://127.0.0.1:8002")
    
    # create user
    email = f"test_{random.randint(0,100000)}@example.com"
    res = client.post("/api/v1/auth/signup", json={"email": email, "username": email, "password": "password123", "confirm_password": "password123", "name": "E2E Test"})
    
    conn = sqlite3.connect("memory/jobs.db")
    conn.execute("UPDATE users SET email_verified = 1 WHERE email = ?", (email,))
    conn.commit()
    conn.close()

    res = client.post("/api/v1/auth/login", json={"identifier": email, "password": "password123"})
    print("login", res.status_code)
    cookies = res.cookies
    
    res = client.post("/api/v1/chats", cookies=cookies)
    chat_id = res.json()["chat_id"]
    
    res = client.post(f"/api/v1/chats/{chat_id}/messages", json={"content": "What is 2+2?"}, cookies=cookies)
    job_id = res.json()["job_id"]
    print("job_id", job_id)
    
    # Wait for job to complete
    for _ in range(50):
        res = client.get(f"/api/v1/jobs/{job_id}", cookies=cookies)
        status = res.json()["status"]
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            print("Final status:", status)
            break
        time.sleep(0.2)
        
    res = client.get(f"/api/v1/jobs/{job_id}", cookies=cookies)
    print("Job status:", res.json()["status"])
    
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events/stream", cookies=cookies) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data["event_type"] == "JOB_COMPLETED":
                    print("JOB_COMPLETED payload:", data["payload"])

    res = client.get(f"/api/v1/chats/{chat_id}", cookies=cookies)
    messages = res.json()["messages"]
    print("Final messages:")
    for m in messages:
        print(f"[{m['role']}] {m['content']}")

e2e_test()
