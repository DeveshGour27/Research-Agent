"""Data models for HITL operations."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel


class HITLRequestType(str, Enum):
    TOOL = "TOOL"
    PLAN = "PLAN"


class HITLRequestStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class HITLPolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    DENY = "DENY"


class HITLRequestResponse(BaseModel):
    """Pydantic model for API responses representing a HITL request."""
    request_id: str
    job_id: str
    run_id: str
    request_type: str
    component_name: str
    status: str
    payload: Any
    created_at: str
    expires_at: str | None = None
    decided_at: str | None = None
    decided_by: str | None = None
    decision_reason: str | None = None

