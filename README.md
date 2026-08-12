# Production AI Research Agent

Version: 1.0
Status: Draft

## Overview
The Production AI Research Agent is an autonomous AI system capable of planning, reasoning, retrieving knowledge, using external tools, and producing high-quality research reports with verifiable citations.

## Deployment & Configuration

### Local Development Setup
The application defaults to `ENVIRONMENT=development` and uses SQLite for persistence out of the box. No configuration is necessary to get started locally.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
uvicorn app.main_api:app --reload
```

### Environment Variables
For production, you must set configuration securely via environment variables. See `.env.example` for a complete list of placeholders.

Key variables include:
- `ENVIRONMENT=production`
- `DATABASE_URL`: Your PostgreSQL connection string.
- `GROQ_API_KEY`: Required in production for the LLM pipeline.
- `WEB_SEARCH_API_KEY`: Required for the search tool.

> [!WARNING]
> Never commit your `.env` file to version control. Production secrets must be provided securely to the container runtime.

### Database Configuration
- **SQLite**: Supported only in `development` mode (`ENVIRONMENT=development`). The schema will automatically initialize on startup if it doesn't exist.
- **PostgreSQL**: Required in `production` mode.

### Running Migrations
The project uses **Alembic** to manage database schema migrations.

To initialize or upgrade a database to the latest schema:
```bash
alembic upgrade head
```

To downgrade a migration (if supported):
```bash
alembic downgrade -1
```

### Building the Docker Image
The application includes a production-oriented `Dockerfile` that uses a secure, minimal Python environment with a non-root user.

```bash
docker build -t research-agent .
```

### Running Docker Compose
A local production-like environment is provided via `docker-compose.yml`, which spins up a PostgreSQL database and the API container.

```bash
docker compose up -d
```
The compose file automatically manages:
1. PostgreSQL persistent volumes.
2. PostgreSQL healthchecks.
3. Running `alembic upgrade head` before starting the API.
4. Starting the API on port `8000`.

### Endpoints
The following operational endpoints are available:
- **Health**: `GET /health` - Lightweight liveness probe.
- **Readiness**: `GET /ready` - Deep check verifying PostgreSQL connectivity and `AsyncJobManager` initialization.
- **Metrics**: `GET /metrics` - Operational metrics snapshot.

### Graceful Shutdown
The API correctly handles `SIGTERM` and `SIGINT` signals. Upon termination, it stops accepting new requests and signals the `AsyncJobManager` to perform a graceful shutdown, ensuring running tasks are either safely aborted or completed.

### Architectural Constraints
> [!IMPORTANT]
> **Single Process Limitation**: Currently, the system relies on an in-memory `AsyncJobManager` to orchestrate multi-agent execution. Due to this architectural constraint, **only ONE application process** (one Uvicorn worker) is officially supported. Deploying multiple processes without an external broker (e.g., Redis/Celery) will result in inconsistent job states and execution tracking. Future architectural phases will address horizontal scaling.
