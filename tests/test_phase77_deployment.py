"""Tests for Phase 7.7 — Production Deployment, Configuration & Containerization."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.config import Settings, Environment
from app.main_api import app


# ----------------------------------------------------------------------
# 1-7. Configuration Hardening
# ----------------------------------------------------------------------
class TestConfiguration:
    def test_production_missing_secret_fails(self):
        """Missing GROQ_API_KEY in production mode must fail validation."""
        with pytest.raises(ValidationError) as exc:
            Settings(
                environment="production",
                groq_api_key="",
                database_url="postgresql+psycopg://user:pass@localhost:5432/db",
            )
        assert "GROQ_API_KEY must be set in production environment" in str(exc.value)

    def test_production_sqlite_fails(self):
        """SQLite is not allowed in production mode."""
        with pytest.raises(ValidationError) as exc:
            Settings(
                environment="production",
                groq_api_key="valid-key",
                database_url="sqlite:///./memory/jobs.db",
            )
        assert "DATABASE_URL must not be SQLite in production environment" in str(exc.value)

    def test_production_valid_config_passes(self):
        """Valid production configuration passes validation."""
        s = Settings(
            environment="production",
            groq_api_key="valid-key",
            database_url="postgresql+psycopg://user:pass@localhost:5432/db",
        )
        assert s.environment == Environment.PRODUCTION

    def test_sqlite_remains_supported_in_development(self):
        """SQLite remains supported for development."""
        s = Settings(
            environment="development",
            database_url="sqlite:///./memory/jobs.db",
        )
        assert s.environment == Environment.DEVELOPMENT
        assert "sqlite" in s.database_url

    def test_secrets_not_exposed_in_validation_errors(self):
        """Secret values should not be leaked in validation errors if a different field fails."""
        with pytest.raises(ValidationError) as exc:
            Settings(
                environment="development",
                groq_api_key="SUPER_SECRET_KEY_123",
                database_url="sqlite:///./memory/jobs.db",
                llm_temperature=3.0,  # invalid
            )
        assert "SUPER_SECRET_KEY_123" not in str(exc.value)

    def test_env_example_contains_placeholders(self):
        """.env.example must not contain real api keys."""
        root = Path(__file__).parent.parent
        env_example = root / ".env.example"
        assert env_example.exists()
        content = env_example.read_text()
        assert "your-groq-api-key-here" in content
        assert "gsk_" not in content
        assert "sk-" not in content


# ----------------------------------------------------------------------
# 8-16. Docker & Containerization
# ----------------------------------------------------------------------
class TestContainerization:
    @pytest.fixture
    def root(self):
        return Path(__file__).parent.parent

    def test_docker_files_exist(self, root):
        assert (root / "Dockerfile").exists()
        assert (root / "docker-compose.yml").exists()

    def test_dockerfile_security_and_best_practices(self, root):
        content = (root / "Dockerfile").read_text()
        
        # 9. No plaintext API keys
        assert "GROQ_API_KEY=" not in content
        assert "gsk_" not in content
        
        # 10. Does not copy .env
        assert "COPY .env" not in content
        
        # 11. Uses non-root user
        assert "USER research_agent" in content
        
        # 12. Does not enable reload mode (uvicorn --reload)
        assert "--reload" not in content
        assert "CMD [\"uvicorn\", \"app.main_api:app\"" in content

    def test_docker_compose_properties(self, root):
        content = (root / "docker-compose.yml").read_text()
        
        # 13. Contains PostgreSQL
        assert "postgres:" in content
        assert "image: postgres:" in content
        
        # 14. Exposes application correctly
        assert "\"8000:8000\"" in content
        
        # 15. PostgreSQL credentials are environment-driven
        assert "${POSTGRES_USER:-postgres}" in content
        assert "${POSTGRES_PASSWORD:-postgres_password}" in content
        
        # 16. Healthcheck targets /health
        assert "curl" in content
        assert "/health" in content


# ----------------------------------------------------------------------
# 17-20. Architecture Intactness (Existing behavior)
# ----------------------------------------------------------------------
class TestArchitectureBoundaries:
    def test_readiness_behavior_compatible_with_startup(self):
        """Readiness endpoint returns 503 if DB down, 200 if up."""
        client = TestClient(app)
        # Using in-memory sqlite defaults, this should pass if JobManager is ready
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert resp.json()["database"] == "connected"

    def test_existing_request_id_middleware_intact(self):
        """RequestIDMiddleware continues to function."""
        client = TestClient(app)
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_existing_authentication_intact(self):
        """Authentication continues to function and protect routes."""
        client = TestClient(app)
        resp = client.post("/api/v1/research/jobs", json={"goal": "test"})
        assert resp.status_code == 401
