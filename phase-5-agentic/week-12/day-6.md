# Phase 5 · Week 12 · Day 6 — Dockerization + FastAPI Main App + Full Integration

> Objective: Package the agent framework as a portable Docker service with FastAPI as the API layer.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

## Concept

After building all subsystems, the final integration step is packaging them into a deployable service. This requires two things:

1. **FastAPI as the API layer** — exposes HTTP endpoints for the agent, health checks, metrics, and HITL operations. FastAPI's lifespan pattern replaces global state with request-scoped or app-scoped instances.
2. **Docker as the portability layer** — a multi-stage build produces a minimal runtime image. Docker Compose wires the app with optional infrastructure services (Redis, a database) and injects secrets from the host environment.

**Dependency injection via lifespan:** FastAPI's `lifespan` context manager runs on startup and shutdown. All shared objects (registry, memory store, audit logger, metrics collector, Anthropic client) are created once and attached to `app.state`. Route handlers access them via `app.state.*`. This avoids module-level globals that break test isolation.

**Multi-stage Docker build:** The builder stage installs all dependencies (including dev tools). The runtime stage copies only the installed packages and application source — no build tooling, no cache directories, smaller attack surface. A dedicated non-root user (`appuser`) runs the process.

**Secret injection:** `docker-compose.yml` uses `${ANTHROPIC_API_KEY}` syntax. This reads the value from the host shell environment at `docker compose up` time. The value never appears in the image or compose file. The app reads it via `os.getenv("ANTHROPIC_API_KEY")` through `pydantic-settings`.

## How it works internally

FastAPI startup sequence:

1. `lifespan` fires — creates `AgentSettings` (reads env vars), instantiates all subsystems
2. Each subsystem is stored on `app.state`
3. Route handlers are called per-request — they read subsystems from `app.state`, no module-level state
4. On shutdown, `lifespan` resumes after the `yield` for cleanup

Docker multi-stage build flow:

```
docker build .
  → builder stage: python:3.12-slim, install deps into site-packages
  → runtime stage: python:3.12-slim (fresh), copy site-packages + app source
  → CMD: uvicorn finops_agent.api.main:app --host 0.0.0.0 --port 8000
```

The runtime image never ran `pip install` — packages are copied as files. This means no `pip`, `setuptools`, or build metadata in production.

Compose services start in dependency order. The health check on `finops-agent` polls `GET /health` every 30 seconds — container orchestrators (ECS, Kubernetes) use this signal for rolling deploys and restart policies.

## Worked example

Main FastAPI app:

```python
# src/finops_agent/api/main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

from finops_agent.config import AgentSettings
from finops_agent.core.registry import ToolRegistry, ToolSchema
from finops_agent.memory.store import SQLiteMemoryStore
from finops_agent.audit.logger import AuditLogger
from finops_agent.hitl.gate import HumanApprovalGate
from finops_agent.observability.metrics import MetricsCollector
from finops_agent.tools.finance import retrieve_invoice, validate_payment, check_budget
from finops_agent.core.sanitizer import sanitize_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AgentSettings()
    registry = ToolRegistry()
    # Register finance tools
    registry.register(ToolSchema(name="retrieve_invoice", description="Fetch invoice by ID", parameters={"invoice_id": "str"}, risk_level="low"), retrieve_invoice)
    registry.register(ToolSchema(name="validate_payment", description="Validate payment amount", parameters={"invoice_id": "str", "amount": "float"}, risk_level="low"), validate_payment)
    registry.register(ToolSchema(name="check_budget", description="Check department budget", parameters={"department": "str", "amount": "float"}, risk_level="medium"), check_budget)
    memory = SQLiteMemoryStore(db_path=os.getenv("MEMORY_DB_PATH", "memory.db"))
    audit = AuditLogger(log_path=os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl"))
    gate = HumanApprovalGate(auto_approve_levels={"low"})
    metrics = MetricsCollector()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    app.state.settings = settings
    app.state.registry = registry
    app.state.memory = memory
    app.state.audit = audit
    app.state.gate = gate
    app.state.metrics = metrics
    app.state.client = client
    logger.info("FinOps Agent API started")
    yield
    logger.info("FinOps Agent API shutdown")


app = FastAPI(title="FinOps Agent Orchestration Framework", version="1.0.0", lifespan=lifespan)


class AgentRequest(BaseModel):
    session_id: str
    task: str

class AgentResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[str] = []
    pending_approvals: list[str] = []


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(req: AgentRequest) -> AgentResponse:
    audit: AuditLogger = app.state.audit
    gate: HumanApprovalGate = app.state.gate
    metrics: MetricsCollector = app.state.metrics

    clean_task = sanitize_input(req.task)
    audit.log_event("agent_request", session_id=req.session_id, task=clean_task)
    span = metrics.start_span("api.agent_run", session_id=req.session_id)

    # Simplified: direct tool dispatch without LLM loop for determinism in tests
    tool_calls: list[str] = []
    answer = f"Processed task for session {req.session_id}: {clean_task}"

    span.finish()
    metrics.increment("api_requests_total")
    audit.log_event("agent_response", session_id=req.session_id, answer=answer)
    return AgentResponse(session_id=req.session_id, answer=answer, tool_calls=tool_calls)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "finops-agent"}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    return app.state.metrics.get_counters()
```

Dockerfile (multi-stage):

```dockerfile
# Dockerfile
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir build pip-tools && \
    pip install --no-cache-dir -e .

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN useradd --no-create-home --shell /bin/false appuser
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY src/ ./src/
COPY pyproject.toml .
RUN mkdir -p logs && chown appuser:appuser logs
USER appuser
EXPOSE 8000
CMD ["uvicorn", "finops_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

docker-compose.yml:

```yaml
# docker-compose.yml
version: "3.9"
services:
  finops-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      MEMORY_DB_PATH: /data/memory.db
      AUDIT_LOG_PATH: /data/logs/audit.jsonl
    volumes:
      - agent-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  agent-data:
```

pyproject.toml:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "finops-agent"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.30.0",
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/finops_agent"]
```

## Exercises

1. Add a `POST /agent/invoke-tool` endpoint that takes `tool_name: str` and `parameters: dict`, runs the tool through `ToolRegistry`, and returns the result. Include a HITL gate check — if the tool's `risk_level` is `"critical"`, return a `202 Accepted` with a pending approval ID instead of calling the tool directly.
2. Modify the Dockerfile to run `python -m pytest tests/unit` during the builder stage after installing dependencies. The build should fail if any tests fail.
3. Add a `depends_on` condition to the `finops-agent` service in `docker-compose.yml` that waits for the Redis service to pass a health check before starting. Add a `healthcheck` block to the `redis` service using `redis-cli ping`.
4. Write a `pytest` integration test using `httpx.AsyncClient` with `ASGITransport` that: (a) calls `GET /health` and asserts `status_code == 200` and `{"status": "ok"}` in the response, (b) calls `GET /metrics` and asserts `status_code == 200` and the response is a dict.

## Self-check

- [ ] I can build a multi-stage Dockerfile for a FastAPI Python service
- [ ] I can wire all subsystems via FastAPI lifespan without global singletons
- [ ] I understand how docker-compose passes secrets via `${VAR}` without hardcoding
- [ ] I can write integration tests using httpx `AsyncClient` against an ASGI app
