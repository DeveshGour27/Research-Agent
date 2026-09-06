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

from pydantic import Field, field_validator, model_validator
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
    # Authentication & Email
    # ------------------------------------------------------------------ #
    session_secret: str = Field(default="super-secret-default-key", description="Secret key for signing sessions")
    smtp_host: str = Field(default="localhost", description="SMTP server host")
    smtp_port: int = Field(default=1025, description="SMTP server port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")
    email_from: str = Field(default="noreply@researchagent.local", description="Sender email address")
    app_base_url: str = Field(default="http://localhost:3000", description="Frontend base URL for links")
    
    # ------------------------------------------------------------------ #
    # Google OAuth
    # ------------------------------------------------------------------ #
    google_client_id: str = Field(default="", description="Google OAuth Client ID")
    google_client_secret: str = Field(default="", description="Google OAuth Client Secret")
    google_redirect_uri: str = Field(default="http://localhost:8000/api/v1/auth/google/callback", description="Google OAuth callback URI")

    # ------------------------------------------------------------------ #
    # LLM / API
    # ------------------------------------------------------------------ #
    groq_api_key: str = Field(
        default="",
        description="Groq API key. Must be set in production.",
    )
    llm_model: str = Field(
        default="qwen/qwen3.6-27b",
        min_length=1,
        description="Primary chat model used by the configured LLM provider.",
    )
    llm_fallback_model: str = Field(
        default="openai/gpt-oss-20b",
        min_length=1,
        description="Fallback chat model used after a transient provider failure.",
    )
    llm_provider: str = Field(
        default="groq",
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
        default=4_096,
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
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description="List of allowed origins for CORS. Used by the API boundary.",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level emitted by the application.",
    )
    database_url: str = Field(
        default="sqlite:///./memory/jobs.db",
        description="Database connection string for relational persistence.",
    )
    max_concurrent_jobs: int = Field(
        default=10,
        gt=0,
        description="Maximum number of concurrently running background jobs.",
    )
    job_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="Execution timeout in seconds for a single research job.",
    )
    worker_id: str | None = Field(
        default=None,
        description="Unique identifier for this application instance. If not provided, a UUID will be generated.",
    )
    job_heartbeat_interval_seconds: int = Field(
        default=10,
        gt=0,
        description="Interval in seconds for a running job to update its heartbeat.",
    )
    job_stale_after_seconds: int = Field(
        default=60,
        gt=0,
        description="Threshold in seconds before a job without a heartbeat is considered stale.",
    )
    job_recovery_poll_interval_seconds: int = Field(
        default=15,
        gt=0,
        description="Interval in seconds to poll for stale running jobs to recover.",
    )
    job_max_attempts: int = Field(
        default=3,
        gt=0,
        description="Maximum number of execution attempts for a job.",
    )

    api_rate_limit_requests: int = Field(
        default=60,
        gt=0,
        description="Maximum API requests per user within the rate-limit window.",
    )
    api_rate_limit_window_seconds: int = Field(
        default=60,
        gt=0,
        description="Rate-limit sliding window duration in seconds.",
    )

    # ------------------------------------------------------------------ #
    # Multi-Model Gateway (Phase 15)
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(default="", description="API key for OpenAI provider.")
    llm_gateway_profiles_json: str = Field(default="", description="JSON string of profiles")

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
        default=50,
        gt=0,
        description="Number of candidates returned by each retrieval branch (vector and lexical).",
    )
    top_k_rerank: int = Field(
        default=50,
        gt=0,
        description="Maximum number of candidates passed to the reranker.",
    )
    top_k_final: int = Field(
        default=10,
        gt=0,
        description="Final number of items returned to the user.",
    )
    # Feature flags and weights for the RAG pipeline
    rag_enabled: bool = Field(
        default=True, description="Enable the RAG pipeline features."
    )
    bm25_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for BM25 in hybrid scoring"
    )
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for vector similarity in hybrid scoring"
    )
    reranking_enabled: bool = Field(
        default=True, description="Enable optional reranking after retrieval"
    )
    # Reranker selection: 'neural' | 'lexical' | 'noop'
    reranker_type: str = Field(default="neural", description="Reranker type to use")
    rewriting_enabled: bool = Field(
        default=True, description="Enable query rewriting before retrieval"
    )
    decomposition_enabled: bool = Field(
        default=True, description="Enable query decomposition into subqueries"
    )
    max_retrieval_iterations: int = Field(
        default=3, gt=0, description="Maximum iterations for the agentic retrieval loop"
    )
    max_retrieval_retries: int = Field(
        default=1, ge=0, description="Maximum number of times to retry retrieval when evidence is insufficient"
    )
    max_reflection_attempts: int = Field(
        default=1, ge=0, description="Maximum number of times to invoke reflection per step"
    )
    reflection_enabled: bool = Field(
        default=True, description="Whether to enable pre-generation reflection gate"
    )
    
    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    max_plan_steps: int = Field(
        default=10,
        gt=0,
        description="Maximum number of steps permitted in an LLM-generated plan.",
    )
    # Embedding provider selection
    embedding_provider: str = Field(
        default="sentence_transformers", description="Embedding provider: sentence_transformers|hashing|mock"
    )
    embedding_dim: int = Field(default=128, gt=1, description="Embedding dimension for hashing provider")
    # Vector store backend
    vector_store: str = Field(default="chroma", description="Vector store backend: chroma|inmemory")

    # ------------------------------------------------------------------ #
    # Resilience
    # ------------------------------------------------------------------ #
    max_retries: int = Field(
        default=3,
        gt=0,
        description="Maximum retry attempts for tool execution and API calls.",
    )

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    mcp_servers: str = Field(
        default="{}",
        description="JSON string representing a dictionary of MCP server configurations.",
    )
    web_search_provider: str = Field(
        default="searxng",
        description="Provider used for the WebSearchTool (e.g., searxng).",
    )
    searxng_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for the local SearXNG instance.",
    )
    web_search_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Timeout for a single web search API request in seconds.",
    )
    web_search_max_results: int = Field(
        default=2,
        gt=0,
        description="Maximum number of search results to return.",
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
    @field_validator("llm_model", "llm_fallback_model")
    @classmethod
    def model_name_must_not_be_blank(cls, model_name: str) -> str:
        """Reject model names that contain only whitespace."""
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("LLM model names must not be blank.")
        return normalized_model_name

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

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Strictly enforce that production environments have required secrets."""
        if self.environment == Environment.PRODUCTION:
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY must be set in production environment.")
            if self.database_url.startswith("sqlite"):
                raise ValueError("DATABASE_URL must not be SQLite in production environment.")
        return self

    @model_validator(mode="after")
    def validate_top_k_hierarchy(self) -> "Settings":
        """Ensure retrieval flow candidates decrease strictly."""
        if not (self.top_k_retrieval >= self.top_k_rerank >= self.top_k_final > 0):
            raise ValueError(
                f"Invalid top-k hierarchy: retrieval ({self.top_k_retrieval}) "
                f">= rerank ({self.top_k_rerank}) "
                f">= final ({self.top_k_final}) > 0 must hold."
            )
        return self


# Single application-wide instance.
# Imported by all modules that need configuration.
settings = Settings()
