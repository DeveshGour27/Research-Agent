"""Database initialization and session management."""

from __future__ import annotations

from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

import os

# Engine setup
_connect_args = {}
if settings.database_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
    # Extract path and ensure directory exists if it's a file
    db_path = settings.database_url.replace("sqlite:///", "")
    if db_path and db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

engine_kwargs: dict = {
    "connect_args": _connect_args,
}

if settings.database_url.startswith("postgresql"):
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(
    settings.database_url,
    **engine_kwargs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ensure tables are created for local development/testing without Alembic
# In production, Alembic migrations must be run before startup
from app.db.models import User, ApiKey, Job, JobStep, HITLRequest, Conversation, Message, OAuthIdentity, UserSession
from app.config import Environment

if settings.environment != Environment.PRODUCTION:
    pass # Base.metadata.create_all(bind=engine)
