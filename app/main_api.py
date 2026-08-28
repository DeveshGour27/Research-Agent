"""FastAPI application entrypoint for Phase 7.1 service boundary."""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.api.health import health_router
from app.db.database import Base, engine, SessionLocal
from app.services.job_manager import AsyncJobManager
from app.middleware.request_id import RequestIDMiddleware
from app.config import settings
from app.services.rate_limiter import RateLimiter
from app.hitl.service import HITLService
from app.hitl.policy import HITLPolicy

def default_supervisor_factory(hitl_service: HITLService | None = None):
    # In a real deployment, this builds the full Phase 6 Supervisor.
    # For Phase 7.4, this is overridden in tests.
    from app.agent.supervisor import Supervisor
    from app.agent.registry import AgentRegistry
    return Supervisor(registry=AgentRegistry(), hitl_service=hitl_service)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup (dev only)
    from app.config import Environment, settings
    if settings.environment != Environment.PRODUCTION:
        Base.metadata.create_all(bind=engine)
    
    # Initialize HITL
    policy = HITLPolicy(
        require_human_tools={"safe_search", "database.write"}, # Example policy config
        require_human_plans=True
    )
    hitl_service = HITLService(session_factory=SessionLocal, policy=policy)
    app.state.hitl_service = hitl_service

    # Initialize AsyncJobManager
    job_manager = AsyncJobManager(
        session_factory=SessionLocal,
        supervisor_factory=lambda: default_supervisor_factory(hitl_service),
        max_concurrent_jobs=settings.max_concurrent_jobs,
        job_timeout_seconds=settings.job_timeout_seconds,
        worker_id=settings.worker_id,
        job_heartbeat_interval_seconds=settings.job_heartbeat_interval_seconds,
        job_stale_after_seconds=settings.job_stale_after_seconds,
        job_recovery_poll_interval_seconds=settings.job_recovery_poll_interval_seconds,
        hitl_service=hitl_service,
    )
    app.state.job_manager = job_manager
    
    # Initialize per-user rate limiter
    rate_limiter = RateLimiter(
        max_requests=settings.api_rate_limit_requests,
        window_seconds=settings.api_rate_limit_window_seconds,
    )
    app.state.rate_limiter = rate_limiter
    
    yield
    
    # Shutdown AsyncJobManager gracefully
    await job_manager.shutdown()

app = FastAPI(
    title="Production AI Research Agent API",
    version="1.0.0",
    description="Production-oriented HTTP service boundary for the AI Research Agent system.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Include routers — health_router is public, api_router contains protected routes
from app.api.auth_routes import router as auth_router
app.include_router(health_router)
app.include_router(api_router)
app.include_router(auth_router)

# Add request-ID propagation middleware
app.add_middleware(RequestIDMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_app() -> FastAPI:
    """Factory function for creating the FastAPI application instance."""
    return app

