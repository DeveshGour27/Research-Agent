"""Tests for Phase 7.2 Relational Persistence and Job Repository."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError

from app.db.database import Base, get_db
from app.db.repository import SQLJobRepository
from app.main_api import app
from app.api.auth import get_current_user

# ----------------------------------------------------------------------
# Test Fixtures (In-Memory SQLite with StaticPool)
# ----------------------------------------------------------------------

# Using StaticPool ensures the same in-memory database connection is 
# shared across all sessions in a test.
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

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def repo(db_session):
    return SQLJobRepository(db_session)

@pytest.fixture(scope="function")
def client(db_session):
    """Test client with overridden dependencies."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_current_user():
        user_id = "test-user-id"
        repo = SQLJobRepository(db_session)
        user = repo.get_user(user_id)
        if not user:
            user = repo.create_user(email="test@example.com", user_id=user_id)
        return user

    app.dependency_overrides[get_db] = override_get_db
    from app.api.auth import get_current_user
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    with TestClient(app) as c:
        yield c
        
    app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# Database Initialization Tests
# ----------------------------------------------------------------------
def test_db_initialization_creates_tables(db_session):
    # 1. Verify tables exist by querying them without error
    from app.db.models import User
    users = db_session.query(User).all()
    assert users == []


# ----------------------------------------------------------------------
# Repository User & API Key Tests
# ----------------------------------------------------------------------
def test_create_user_generates_uuid(repo):
    # 2. Test user creation
    user = repo.create_user(email="test@example.com")
    assert user.user_id is not None
    assert user.email == "test@example.com"
    assert user.created_at is not None

def test_get_user_by_id(repo):
    # 3. Test retrieving user
    user1 = repo.create_user()
    user2 = repo.get_user(user1.user_id)
    assert user2 is not None
    assert user1.user_id == user2.user_id

def test_create_api_key(repo):
    # 4. Test API key creation
    user = repo.create_user()
    api_key = repo.create_api_key(user_id=user.user_id, key_hash="hash123")
    assert api_key.key_hash == "hash123"
    assert api_key.user_id == user.user_id
    assert api_key.is_active is True
    
    fetched = repo.get_api_key("hash123")
    assert fetched is not None
    assert fetched.user_id == user.user_id


# ----------------------------------------------------------------------
# Repository Job Lifecycle Tests
# ----------------------------------------------------------------------
def test_create_job_default_status(repo):
    # 5. Job defaults to PENDING
    user = repo.create_user()
    job = repo.create_job(job_id="job1", user_id=user.user_id, goal="Test Goal")
    assert job.status == "PENDING"
    assert job.goal == "Test Goal"
    assert job.completed_at is None

def test_get_job_enforces_user_ownership(repo):
    # 6. Test retrieving job by wrong user
    u1 = repo.create_user()
    u2 = repo.create_user()
    repo.create_job(job_id="job1", user_id=u1.user_id, goal="Goal")
    
    # User 2 shouldn't see User 1's job
    assert repo.get_job("job1", u2.user_id) is None
    # User 1 should see their job
    assert repo.get_job("job1", u1.user_id) is not None

def test_update_job_status_valid(repo):
    # 7. Test status update
    user = repo.create_user()
    job = repo.create_job(job_id="job1", user_id=user.user_id, goal="Goal")
    updated = repo.update_job_status("job1", user.user_id, "RUNNING")
    assert updated is not None
    assert updated.status == "RUNNING"
    assert updated.completed_at is None

def test_update_job_status_sets_completed_at_for_terminal_state(repo):
    # 8. Test completed_at set on CANCELLED
    user = repo.create_user()
    repo.create_job(job_id="job1", user_id=user.user_id, goal="Goal")
    updated = repo.update_job_status("job1", user.user_id, "CANCELLED")
    assert updated.status == "CANCELLED"
    assert updated.completed_at is not None

def test_update_job_result(repo):
    # 9. Test full result update
    user = repo.create_user()
    repo.create_job(job_id="job1", user_id=user.user_id, goal="Goal")
    
    # Must transition to RUNNING before COMPLETED per Phase 7.6 state machine rules
    repo.update_job_status("job1", user.user_id, "RUNNING")
    
    updated = repo.update_job_result(
        "job1", user.user_id, "COMPLETED", result="Success!", error_message=None
    )
    assert updated.status == "COMPLETED"
    assert updated.result == "Success!"
    assert updated.completed_at is not None


# ----------------------------------------------------------------------
# Repository Job Step Tests
# ----------------------------------------------------------------------
def test_create_job_step(repo):
    # 10. Test step creation
    user = repo.create_user()
    job = repo.create_job(job_id="job1", user_id=user.user_id, goal="Goal")
    step = repo.create_job_step("step1", "job1", "SEARCH")
    assert step.task_type == "SEARCH"
    assert step.status == "PENDING"

def test_update_job_step(repo):
    # 11. Test step update
    user = repo.create_user()
    repo.create_job(job_id="job1", user_id=user.user_id, goal="Goal")
    repo.create_job_step("step1", "job1", "SEARCH")
    updated = repo.update_job_step("step1", "job1", "COMPLETED", output="Found stuff", execution_time_seconds=1.5)
    assert updated.status == "COMPLETED"
    assert updated.output == "Found stuff"
    assert updated.execution_time_seconds == 1.5

def test_get_job_steps_ordered_by_created_at(repo):
    # 12. Test step ordering
    user = repo.create_user()
    repo.create_job(job_id="job1", user_id=user.user_id, goal="Goal")
    repo.create_job_step("step1", "job1", "TASK_A")
    repo.create_job_step("step2", "job1", "TASK_B")
    
    steps = repo.get_job_steps("job1", user.user_id)
    assert len(steps) == 2
    assert steps[0].task_type == "TASK_A"
    assert steps[1].task_type == "TASK_B"


# ----------------------------------------------------------------------
# Foreign Key Tests
# ----------------------------------------------------------------------
def test_foreign_key_constraint_job_requires_user(db_session, repo):
    # 13. Test database enforces integrity constraints
    with pytest.raises(IntegrityError):
        repo.create_job(job_id="job1", user_id="non-existent-user", goal="Goal")


# ----------------------------------------------------------------------
# API Integration Tests
# ----------------------------------------------------------------------
def test_api_create_job_persists_in_db(client, repo):
    # 14. API POST correctly writes to DB
    response = client.post("/api/v1/research/jobs", json={"goal": "Test database persistence"})
    assert response.status_code == 202
    data = response.json()
    job_id = data["job_id"]
    
    # Verify in DB
    job = repo.get_job(job_id, "test-user-id")
    assert job is not None
    assert job.goal == "Test database persistence"
    assert job.status == "PENDING"

def test_api_get_job_returns_persisted_data(client, repo):
    # 15. API GET correctly reads from DB
    user = repo.create_user(user_id="test-user-id")
    job = repo.create_job(job_id="test-job-999", user_id=user.user_id, goal="Pre-existing goal")
    repo.update_job_status(job.job_id, user.user_id, "RUNNING")
    
    response = client.get(f"/api/v1/research/jobs/{job.job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-999"
    assert data["status"] == "RUNNING"
