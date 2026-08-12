# Use a minimal Python base image
FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r research_agent && useradd -r -g research_agent research_agent

# Set working directory
WORKDIR /app

# Install dependencies (do this before copying app code to cache layers)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and Alembic migrations
COPY app /app/app
COPY alembic /app/alembic
COPY alembic.ini /app/

# Create memory directory for SQLite fallback and chroma_db directory
RUN mkdir -p /app/memory /app/chroma_db && chown -R research_agent:research_agent /app

# Switch to non-root user
USER research_agent

# Expose port
EXPOSE 8000

# Start FastAPI application with Uvicorn (exec form)
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
