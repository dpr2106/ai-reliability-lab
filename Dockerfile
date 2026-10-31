# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
# Test Stage: includes quality tooling, test runner, and development dependencies
# -----------------------------------------------------------------------------
FROM base AS test

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[dev]"

COPY reliability_lab/ reliability_lab/
COPY tests/ tests/
COPY data/ data/
COPY docs/ docs/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

CMD ["pytest", "-v"]

# -----------------------------------------------------------------------------
# Runtime Stage: minimal production image running as non-root user
# -----------------------------------------------------------------------------
FROM base AS runtime

# Create unprivileged system group and user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Prepare persistent data volume directory
RUN mkdir -p /data && chown -R appuser:appgroup /data /app

# Install production runtime dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[web,llm]"

# Copy source code, sample benchmark data, and utility scripts
COPY reliability_lab/ reliability_lab/
COPY data/ data/
COPY scripts/ scripts/

RUN pip install --no-cache-dir --no-deps . && \
    chmod +x scripts/*.sh 2>/dev/null || true && \
    chown -R appuser:appgroup /app

USER appuser

ENV RELIABILITY_LAB_DB=/data/experiments.sqlite3

EXPOSE 8000

CMD ["uvicorn", "reliability_lab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

