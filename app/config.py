"""
Configuration module for the Production AI Research Agent.

Responsibilities:
    - Load environment variables from .env file
    - Validate all configuration values at startup
    - Expose a single, typed `settings` object for the rest of the application

Design Decision:
    All configuration is centralised here. No other module should read
    environment variables directly. This makes it trivial to swap sources
    (e.g. secrets managers) in the future without touching business code.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Allowed runtime environments."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Allowed log-level strings (maps to Python's logging constants)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings.

    All values are loaded from environment variables (or a .env file).
    Pydantic validates every field at import time, so misconfigured
    environments are caught before the agent starts running.
    """

    # ------------------------------------------------------------------ #
    # LLM / API
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Must be set in production.",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Default OpenAI chat model to use.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for RAG.",
    )
    llm_provider: str = Field(
        default="openai",
        min_length=1,
        description="Configured LLM provider identifier.",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for chat generation.",
    )
    llm_max_output_tokens: int = Field(
        default=1_024,
        gt=0,
        description="Maximum number of tokens generated for one response.",
    )
    llm_request_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Timeout for a single LLM API request in seconds.",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Runtime environment (development | production | testing).",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level emitted by the application.",
    )

    # ------------------------------------------------------------------ #
    # Vector database
    # ------------------------------------------------------------------ #
    chroma_db_dir: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB persists its data.",
    )
    chroma_collection_name: str = Field(
        default="research_agent",
        description="ChromaDB collection used for the knowledge base.",
    )

    # ------------------------------------------------------------------ #
    # RAG tuning
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Target token count for each document chunk.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap in tokens between adjacent chunks.",
    )
    top_k_retrieval: int = Field(
        default=5,
        gt=0,
        description="Number of chunks returned by the retriever.",
    )

    # ------------------------------------------------------------------ #
    # Resilience
    # ------------------------------------------------------------------ #
    max_retries: int = Field(
        default=3,
        gt=0,
        description="Maximum retry attempts for tool execution and API calls.",
    )

    # ------------------------------------------------------------------ #
    # Pydantic-settings meta
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Accept extra keys in .env without raising an error, so that
        # operators can add their own variables without breaking the app.
        extra="ignore",
        # Validate default values as well so we catch bad Enum defaults.
        validate_default=True,
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(
        cls, overlap: int, info: object
    ) -> int:
        """Ensure chunk_overlap < chunk_size to prevent degenerate splits."""
        # `info.data` holds already-validated sibling fields.
        data = getattr(info, "data", {})
        chunk_size = data.get("chunk_size", 1000)
        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )
        return overlap


# Single application-wide instance.
# Imported by all modules that need configuration.
settings = Settings()
