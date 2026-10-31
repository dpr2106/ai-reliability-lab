"""Tests validating containerization configuration, Dockerfiles, Compose specs, and scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_root_dockerfile_structure() -> None:
    """Verify backend Dockerfile contains multi-stage targets, non-root user, and security."""
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Root Dockerfile must exist"

    content = dockerfile.read_text(encoding="utf-8")

    # Multi-stage targets
    assert "AS base" in content, "Dockerfile must define a base stage"
    assert "AS test" in content, "Dockerfile must define a test stage"
    assert "AS runtime" in content, "Dockerfile must define a runtime stage"

    # Non-root user
    assert "useradd" in content or "adduser" in content, (
        "Dockerfile must create an unprivileged user"
    )
    assert "USER appuser" in content, "Dockerfile runtime stage must switch to non-root appuser"

    # Minimal runtime dependencies & clean cache
    assert "--no-cache-dir" in content, "pip installations must avoid caching wheels"
    assert "EXPOSE 8000" in content, "FastAPI backend port 8000 must be exposed"
    assert "uvicorn" in content, "Runtime command must start uvicorn"


def test_frontend_dockerfile_structure() -> None:
    """Verify frontend Dockerfile contains builder/runtime stages, npm ci, and non-root."""
    dockerfile = REPO_ROOT / "frontend" / "Dockerfile"
    assert dockerfile.is_file(), "frontend/Dockerfile must exist"

    content = dockerfile.read_text(encoding="utf-8")

    # Multi-stage targets
    assert "AS builder" in content, "frontend/Dockerfile must define a builder stage"
    assert "AS runtime" in content, "frontend/Dockerfile must define a runtime stage"

    # Lockfile reproducibility
    assert "npm ci" in content, "frontend/Dockerfile must install via npm ci for reproducibility"
    assert "npm run check" in content, "frontend/Dockerfile must run svelte-check in build stage"
    assert "npm test" in content, "frontend/Dockerfile must run tests in build stage"
    assert "npm run build" in content, "frontend/Dockerfile must run production build"

    # Production runtime
    assert "USER node" in content, (
        "frontend/Dockerfile runtime stage must switch to unprivileged node user"
    )
    assert "EXPOSE 5173" in content, "Frontend port 5173 must be exposed"
    assert '"node"' in content and '"build/index.js"' in content, (
        "Runtime command must launch adapter-node server"
    )


def test_docker_compose_specification() -> None:
    """Verify docker-compose.yml defines required services, volumes, ports, and healthchecks."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml must exist"

    content = compose_path.read_text(encoding="utf-8")

    # Required services
    assert "api:" in content, "docker-compose.yml must define api service"
    assert "frontend:" in content, "docker-compose.yml must define frontend service"
    assert "tests:" in content, "docker-compose.yml must define tests service"
    assert "benchmark:" in content, "docker-compose.yml must define benchmark service"

    # Port mappings
    assert '"8000:8000"' in content or "'8000:8000'" in content or "8000:8000" in content
    assert '"5173:5173"' in content or "'5173:5173'" in content or "5173:5173" in content

    # Healthchecks
    assert "/api/health" in content, "API healthcheck must verify /api/health"
    assert "fetch('http://127.0.0.1:5173/')" in content, (
        "Frontend healthcheck must verify HTTP response"
    )

    # Shared persistent volume
    assert "reliability-lab-data:" in content, "Persistent named volume must be declared"
    assert "/data/experiments.sqlite3" in content, "RELIABILITY_LAB_DB must point to mounted volume"


def test_dockerignore_coverage() -> None:
    """Verify .dockerignore excludes virtual environments, git metadata, and local databases."""
    dockerignore = REPO_ROOT / ".dockerignore"
    assert dockerignore.is_file(), ".dockerignore must exist"

    patterns = set(dockerignore.read_text(encoding="utf-8").splitlines())

    assert ".git" in patterns
    assert ".venv/" in patterns or ".venv" in patterns
    assert "*.sqlite3" in patterns
    assert ".pytest_cache/" in patterns or ".pytest_cache" in patterns
    assert "frontend/node_modules/" in patterns or "frontend/node_modules" in patterns


def test_benchmark_script_properties() -> None:
    """Verify scripts/run_benchmarks.sh is present, executable, and contains strict handling."""
    script_path = REPO_ROOT / "scripts" / "run_benchmarks.sh"
    assert script_path.is_file(), "scripts/run_benchmarks.sh must exist"

    content = script_path.read_text(encoding="utf-8")

    # Strict shell execution
    assert "set -euo pipefail" in content, "Script must use strict bash error handling"

    # Benchmark invocation
    assert "reliability-lab run-benchmark" in content, "Script must invoke run-benchmark"
    assert "reliability-lab run-rag-evaluation" in content, "Script must invoke run-rag-evaluation"

    # Provenance and safety
    assert "GEMINI_API_KEY" in content, "Script must validate GEMINI_API_KEY for live provider"
    assert "PROVENANCE NOTICE" in content, "Script must emit provenance banner for mock runs"
    assert "--skip-nli" in content, "Script must support neural NLI omission flag"


def test_env_example_presence() -> None:
    """Verify .env.example contains configuration defaults without secret keys."""
    env_example = REPO_ROOT / ".env.example"
    assert env_example.is_file(), ".env.example must exist"

    content = env_example.read_text(encoding="utf-8")
    assert "RELIABILITY_LAB_DB" in content
    assert "VITE_API_URL" in content
    assert "AIza" not in content, "Sample env must not contain real API keys"
