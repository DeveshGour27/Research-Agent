"""SQLAlchemy ORM models for the Research Agent API."""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.database import Base


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _uuid_str() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=_uuid_str)
    email = Column(String, unique=True, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    key_hash = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    user = relationship("User", back_populates="api_keys")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="uq_user_idempotency"),
    )

    job_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    goal = Column(Text, nullable=False)
    status = Column(String, default="PENDING", nullable=False, index=True)
    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    trace_id = Column(String, nullable=True, index=True)
    idempotency_key = Column(String, nullable=True, index=True)
    worker_id = Column(String, nullable=True, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    attempt_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="jobs")
    steps = relationship("JobStep", back_populates="job", cascade="all, delete-orphan", order_by="JobStep.created_at")


class JobStep(Base):
    __tablename__ = "job_steps"

    step_id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.job_id"), nullable=False, index=True)
    task_type = Column(String, nullable=False)
    status = Column(String, default="PENDING", nullable=False)
    output = Column(Text, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    job = relationship("Job", back_populates="steps")
