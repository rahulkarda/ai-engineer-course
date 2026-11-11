# Phase 4 · Week 08 · Day 5 — Dockerize + Config + Security

> Objective: Package the Document Intelligence Platform into a production-ready container with security middleware, PII scrubbing, and secret management.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

A FastAPI service that works locally is not production-ready until it can be:

1. Packaged reproducibly (Docker multi-stage build)
2. Composed with its dependencies (vector store, cache)
3. Configured without secrets touching source code (pydantic-settings + env vars)
4. Hardened against common LLM-specific attack vectors (prompt injection, PII leakage)

Security in LLM applications has three distinct layers: transport security (API keys, Bearer tokens), data security (PII scrubbing before logging, before storage), and model security (input validation against prompt injection). All three must be addressed.

## How it works internally

**Multi-stage Docker build**: Stage 1 (builder) installs all build-time deps into a venv. Stage 2 (runtime) copies only the venv and app source — no compiler toolchain, no cached pip artifacts. The final image is smaller and has a reduced attack surface.

**pydantic-settings**: Reads env vars (and optionally `.env` files in dev) at startup, validates types and required fields. If `ANTHROPIC_API_KEY` is missing, the app fails fast at boot rather than at first request. The `Settings` object is instantiated once as a module-level singleton via `lru_cache` so it is not re-read on every request.

**Starlette middleware chain**: Each `BaseHTTPMiddleware` subclass wraps the ASGI app. The chain is: `SecurityMiddleware` (auth check) → `PIIScrubber` (log sanitization) → your route handler. Middleware sees the raw `Request` and can short-circuit by returning a `Response` before calling `call_next`.

**Prompt injection defense**: The most effective lightweight defense is allow-listing structure rather than deny-listing strings. Validate that user input does not exceed a maximum length, does not contain known jailbreak scaffolding (role-switching phrases, delimiter injection), and is coerced to plain text before it reaches the model context.

**Docker Compose networking**: Services in the same Compose project share a user-defined bridge network. The FastAPI app reaches ChromaDB at `http://chromadb:8000` and Redis at `redis://redis:6379` using service names as hostnames — these never need to be exposed to the host.

## Worked example

**Multi-stage Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# ── builder stage ──────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Create a venv inside /build/venv so we can copy it cleanly
RUN python -m venv /build/venv && \
    /build/venv/bin/pip install --upgrade pip && \
    /build/venv/bin/pip install -e ".[all]"

# ── runtime stage ──────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Non-root user for security
RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --gid 1001 --no-create-home --disabled-password appuser

WORKDIR /app

# Copy only the venv from builder (no build tools)
COPY --from=builder /build/venv /app/venv

# Copy application source
COPY src/ ./src/
COPY pyproject.toml ./

# Make venv the active Python
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

**docker-compose.yml**

```yaml
version: "3.9"

services:
  app:
    build:
      context: .
      target: runtime
    ports:
      - "8080:8080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - API_SECRET_KEY=${API_SECRET_KEY}
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - ENVIRONMENT=${ENVIRONMENT:-production}
    depends_on:
      chromadb:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - platform_net
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:0.5.0
    environment:
      - CHROMA_SERVER_AUTH_CREDENTIALS=${CHROMA_AUTH_TOKEN}
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenConfigServerAuthProvider
    volumes:
      - chroma_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - platform_net

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - platform_net

volumes:
  chroma_data:
  redis_data:

networks:
  platform_net:
    driver: bridge
```

**src/config.py — pydantic-settings**

```python
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    anthropic_api_key: SecretStr = Field(...)
    anthropic_model: str = Field(default="claude-haiku-5-5")

    # Auth
    api_secret_key: SecretStr = Field(...)

    # Vector store
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_auth_token: SecretStr | None = Field(default=None)

    # Cache
    redis_url: str = Field(default="redis://localhost:6379/0")

    # App
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    max_input_length: int = Field(default=4096)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Fails fast if required vars are missing."""
    settings = Settings()  # type: ignore[call-arg]
    logger.info("Settings loaded for environment=%s", settings.environment)
    return settings
```

**src/middleware/security.py — SecurityMiddleware + PIIScrubber**

```python
from __future__ import annotations

import logging
import os
import re
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import get_settings

logger = logging.getLogger(__name__)

# PII patterns — extend as needed
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
]

# Prompt injection patterns — conservative blocklist
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.IGNORECASE),
    re.compile(r"<\s*\|?\s*(?:system|human|assistant)\s*\|?\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>"),
    re.compile(r"####\s*(?:new|override)\s+instructions", re.IGNORECASE),
]


class PIIScrubber:
    """Scrub PII from strings before logging."""

    @staticmethod
    def scrub(text: str) -> str:
        for pattern, replacement in _PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforce Bearer token auth on all routes except /health."""

    def __init__(self, app, exclude_paths: list[str] | None = None) -> None:
        super().__init__(app)
        self._exclude = set(exclude_paths or ["/health", "/docs", "/openapi.json"])
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._exclude:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Missing or invalid Authorization header"}, status_code=401)

        token = auth.removeprefix("Bearer ").strip()
        expected = self._settings.api_secret_key.get_secret_value()
        if token != expected:
            logger.warning("auth_failure path=%s", request.url.path)
            return JSONResponse({"detail": "Invalid API key"}, status_code=403)

        return await call_next(request)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Block prompt injection attempts and enforce max input length."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self._settings = get_settings()
        self._scrubber = PIIScrubber()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only inspect JSON POST/PUT bodies
        if request.method in ("POST", "PUT") and "application/json" in request.headers.get(
            "content-type", ""
        ):
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="replace")

            if len(body_text) > self._settings.max_input_length * 4:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)

            for pattern in _INJECTION_PATTERNS:
                if pattern.search(body_text):
                    logger.warning(
                        "injection_attempt path=%s body_preview=%s",
                        request.url.path,
                        self._scrubber.scrub(body_text[:120]),
                    )
                    return JSONResponse(
                        {"detail": "Input contains disallowed content"}, status_code=422
                    )

        return await call_next(request)
```

**src/main.py — wiring middleware**

```python
from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from src.config import get_settings
from src.middleware.security import InputValidationMiddleware, SecurityMiddleware

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Document Intelligence Platform", version="1.0.0")

# Middleware is applied in reverse registration order (last registered = outermost)
app.add_middleware(InputValidationMiddleware)
app.add_middleware(SecurityMiddleware)


@app.get("/health")
async def health() -> dict:
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "ok", "environment": settings.environment}
```

**tests/test_security.py**

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.middleware.security import PIIScrubber


class TestPIIScrubber:
    def test_email_scrubbed(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("Contact john.doe@example.com for details")
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result

    def test_phone_scrubbed(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("Call us at 555-867-5309")
        assert "[PHONE]" in result

    def test_no_pii_unchanged(self):
        scrubber = PIIScrubber()
        text = "The chunking strategy uses a 512-token window"
        assert scrubber.scrub(text) == text

    def test_multiple_pii_types(self):
        scrubber = PIIScrubber()
        result = scrubber.scrub("Email: user@test.org, SSN: 123-45-6789")
        assert "[EMAIL]" in result
        assert "[SSN]" in result
```

## Exercises

1. Add a `RateLimitMiddleware` that allows at most 60 requests per minute per API key. Use Redis (inject via `get_settings().redis_url`) to store sliding-window counters. Return HTTP 429 with a `Retry-After` header when the limit is exceeded.

2. The `PIIScrubber` only handles the patterns shown above. Add patterns for: UK national insurance numbers (format `AB 12 34 56 C`), IPv4 addresses, and AWS access key IDs (format `AKIA...`). Write a parametrized pytest test that covers each new pattern.

3. The `/health` endpoint returns `{"status": "ok"}` unconditionally. Extend it to also check ChromaDB and Redis connectivity (attempt a lightweight ping to each). Return HTTP 200 only if both downstream services respond; return HTTP 503 with `{"status": "degraded", "checks": {...}}` if any fail. Write tests using `unittest.mock.patch`.

(Solutions: ../solutions/week-08/day-5.md)

## Self-check

- [ ] I can explain why a multi-stage Dockerfile reduces the attack surface compared to a single-stage build
- [ ] I can explain why `pydantic-settings` with `SecretStr` is safer than reading env vars directly with `os.getenv` at call sites
- [ ] I can implement Bearer token middleware that short-circuits before the route handler
- [ ] I can implement a PIIScrubber and explain where in the request/response lifecycle it should run
- [ ] I can describe three categories of prompt injection and a mitigation for each

## Supplementary links

- [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- [pydantic-settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Starlette middleware](https://www.starlette.io/middleware/)
- [OWASP LLM Top 10 — Prompt Injection (LLM01)](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [Docker Compose environment variable substitution](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)
