"""Authentication and Authorization Guard for Phase 7.3."""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.db.repository import SQLJobRepository

# Define the API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key() -> str:
    """Generate a cryptographically secure random API key.
    
    This secret should only be returned to the client once at creation time,
    and should never be logged or stored in plaintext.
    """
    return f"ak_{secrets.token_urlsafe(32)}"


def get_api_key_hash(raw_key: str) -> str:
    """Deterministically hash the raw API key for secure storage and lookup.
    
    SHA-256 is appropriate here because the input is a high-entropy,
    cryptographically random secret, making brute-force impossible.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_current_user(
    api_key: str | None = Security(api_key_header),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency to authenticate the user via X-API-Key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )

    # Hash the provided key deterministically to avoid logging/comparing plaintext
    key_hash = get_api_key_hash(api_key)

    repo = SQLJobRepository(db)
    api_key_record = repo.get_api_key(key_hash)

    # Validate key existence and status without revealing if it exists but is inactive
    if not api_key_record or not api_key_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key",
        )

    # Validate associated user exists
    user = api_key_record.user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key: User not found",
        )

    return user
