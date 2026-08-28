import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import tempfile
import os

from app.main_api import app
from app.db.database import Base, get_db
from app.db.models import User, UserSession, _utc_now
from app.api.auth_utils import get_password_hash, hash_verification_token
from app.api.auth import get_current_web_user
from fastapi import Request, Depends

@app.get("/api/v1/test-session")
def endpoint_test_session(user: User = Depends(get_current_web_user)):
    return {"user_id": user.user_id}

@pytest.fixture
def test_db_engine():
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

@pytest.fixture
def override_get_db(test_db_engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client(override_get_db):
    app.dependency_overrides[get_db] = lambda: override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_signup_successful(client, override_get_db):
    res = client.post("/api/v1/auth/signup", json={"email": "test@example.com", "password": "password123"})
    assert res.status_code == 201
    
    # Verify DB state
    user = override_get_db.query(User).filter_by(email="test@example.com").first()
    assert user is not None
    assert user.password_hash is not None
    assert "password123" not in user.password_hash
    assert user.email_verified is False
    assert user.verification_token_hash is not None

def test_signup_duplicate_email(client):
    client.post("/api/v1/auth/signup", json={"email": "test2@example.com", "password": "password123"})
    res = client.post("/api/v1/auth/signup", json={"email": "test2@example.com", "password": "password123"})
    assert res.status_code == 400
    assert "Email already registered" in res.json()["detail"]

def test_signup_invalid_data(client):
    res = client.post("/api/v1/auth/signup", json={"email": "notanemail", "password": "pass"})
    assert res.status_code == 422 # Pydantic email validation
    
    res = client.post("/api/v1/auth/signup", json={"email": "test3@example.com", "password": "short"})
    assert res.status_code == 400
    assert "Password must be at least 8 characters" in res.json()["detail"]

def test_verification_flow(client, override_get_db):
    from app.api.auth_utils import generate_verification_token
    raw_token, hashed_token = generate_verification_token()
    
    user = User(
        email="test_verify@example.com",
        password_hash="fake",
        email_verified=False,
        verification_token_hash=hashed_token,
        verification_token_expires_at=_utc_now() + datetime.timedelta(hours=24)
    )
    override_get_db.add(user)
    override_get_db.commit()
    
    res = client.post("/api/v1/auth/verify", json={"email": "test_verify@example.com", "token": raw_token})
    assert res.status_code == 200
    
    override_get_db.refresh(user)
    assert user.email_verified is True
    assert user.verification_token_hash is None
    
    res2 = client.post("/api/v1/auth/verify", json={"email": "test_verify@example.com", "token": raw_token})
    assert res2.status_code == 200
    assert "already verified" in res2.json()["message"]

def test_verification_invalid_and_expired(client, override_get_db):
    from app.api.auth_utils import generate_verification_token
    raw_token, hashed_token = generate_verification_token()
    
    user = User(
        email="test_inv@example.com",
        password_hash="fake",
        email_verified=False,
        verification_token_hash=hashed_token,
        verification_token_expires_at=_utc_now() - datetime.timedelta(hours=1)
    )
    override_get_db.add(user)
    override_get_db.commit()
    
    res = client.post("/api/v1/auth/verify", json={"email": "test_inv@example.com", "token": raw_token})
    assert res.status_code == 400
    
    res = client.post("/api/v1/auth/verify", json={"email": "test_inv@example.com", "token": "wrongtoken"})
    assert res.status_code == 400

def test_login_flow(client, override_get_db):
    user = User(
        email="login@example.com",
        password_hash=get_password_hash("password123"),
        email_verified=True,
    )
    override_get_db.add(user)
    override_get_db.commit()
    
    res = client.post("/api/v1/auth/login", json={"email": "login@example.com", "password": "wrongpassword"})
    assert res.status_code == 401
    
    res = client.post("/api/v1/auth/login", json={"email": "login@example.com", "password": "password123"})
    assert res.status_code == 200
    cookies = res.cookies
    assert "session_id" in cookies
    
    res2 = client.get("/api/v1/test-session", cookies={"session_id": cookies["session_id"]})
    assert res2.status_code == 200
    
    res3 = client.post("/api/v1/auth/logout", cookies={"session_id": cookies["session_id"]})
    assert res3.status_code == 200, res3.json()
    
    res4 = client.get("/api/v1/test-session", cookies={"session_id": cookies["session_id"]})
    assert res4.status_code == 401

def test_unverified_login(client, override_get_db):
    user = User(
        email="unv@example.com",
        password_hash=get_password_hash("password123"),
        email_verified=False,
    )
    override_get_db.add(user)
    override_get_db.commit()
    
    res = client.post("/api/v1/auth/login", json={"email": "unv@example.com", "password": "password123"})
    assert res.status_code == 403

def test_legacy_api_key(client, override_get_db):
    from app.api.auth import get_current_user, get_api_key_hash
    from app.db.models import ApiKey
    
    user = User()
    override_get_db.add(user)
    override_get_db.commit()
    
    key_hash = get_api_key_hash("ak_test")
    api_key = ApiKey(key_hash=key_hash, user_id=user.user_id, is_active=True)
    override_get_db.add(api_key)
    override_get_db.commit()
    
    @app.get("/api/v1/test-api-key")
    def test_key(u: User = Depends(get_current_user)):
        return {"user_id": u.user_id}
        
    res = client.get("/api/v1/test-api-key", headers={"X-API-Key": "ak_test"})
    assert res.status_code == 200

def test_expired_session(client, override_get_db):
    user = User(email="exp@example.com", email_verified=True)
    override_get_db.add(user)
    override_get_db.commit()
    
    session_id = "test_expired_sess"
    user_session = UserSession(
        session_id=session_id,
        user_id=user.user_id,
        expires_at=_utc_now() - datetime.timedelta(days=1)
    )
    override_get_db.add(user_session)
    override_get_db.commit()
    
    res = client.get("/api/v1/test-session", cookies={"session_id": session_id})
    assert res.status_code == 401

