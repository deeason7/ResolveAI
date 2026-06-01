FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps needed by some Python packages (e.g. psycopg2 build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Deps layer — cached unless pyproject.toml changes
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# App code
COPY . .

# Non-root user
RUN useradd --create-home appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -sf http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
