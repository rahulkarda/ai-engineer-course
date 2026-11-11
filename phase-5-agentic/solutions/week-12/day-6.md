# Week 12 · Day 6 Solutions — Dockerization + FastAPI

## Exercise 1 — POST /agent/invoke-tool endpoint

```python
# src/finops_agent/api/main.py — addition
from fastapi import Request
from pydantic import BaseModel
from typing import Any

class InvokeToolRequest(BaseModel):
    tool_name: str
    parameters: dict[str, Any]
    session_id: str = "anonymous"

@app.post("/agent/invoke-tool")
def invoke_tool(req: InvokeToolRequest, request: Request) -> dict:
    registry = request.app.state.registry
    gate: HumanApprovalGate = request.app.state.gate
    audit: AuditLogger = request.app.state.audit

    entry = registry.get(req.tool_name)
    if not entry:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")

    schema, fn = entry

    # HITL gate check for high-risk tools
    if schema.risk_level in ("high", "critical"):
        from finops_agent.hitl.gate import ApprovalRequest
        approval_req = ApprovalRequest(
            session_id=req.session_id,
            tool_name=req.tool_name,
            parameters=req.parameters,
            risk_level=schema.risk_level,
            reason=f"Direct tool invocation via API",
        )
        result = gate.request_approval(approval_req)
        if result.status.value == "pending":
            audit.log_event("tool_invoke_pending", session_id=req.session_id, tool_name=req.tool_name)
            return {"status": "pending", "approval_id": result.request_id, "message": "Awaiting human review"}

    audit.log_event("tool_invoke", session_id=req.session_id, tool_name=req.tool_name, risk_level=schema.risk_level)
    try:
        result = fn(**req.parameters)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        return {"status": "ok", "tool_name": req.tool_name, "result": result}
    except Exception as e:
        logger.error("Tool %s invocation failed: %s", req.tool_name, e)
        return {"status": "error", "tool_name": req.tool_name, "error": str(e)}
```

## Exercise 2 — pytest step in Dockerfile builder stage

```dockerfile
# Dockerfile — updated builder stage
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml .
COPY src/ ./src/
COPY tests/ ./tests/
RUN pip install --no-cache-dir -e ".[dev]" && \
    python -m pytest tests/unit -x --tb=short
```

If tests fail, `docker build` exits non-zero and the image is not created.

## Exercise 3 — depends_on with Redis health condition

```yaml
# docker-compose.yml — updated
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
    depends_on:
      redis:
        condition: service_healthy
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
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  agent-data:
```

## Exercise 4 — httpx AsyncClient integration tests

```python
# tests/integration/test_api.py
import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.anyio
async def test_health_endpoint():
    from finops_agent.api.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "finops-agent"

@pytest.mark.anyio
async def test_metrics_endpoint():
    from finops_agent.api.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Make one request to increment counter
        await client.post("/agent/run", json={"session_id": "s1", "task": "retrieve invoice"})
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # api_requests_total should be >= 1 after the first /agent/run call
    assert data.get("api_requests_total", 0) >= 1
```

Note: Use `pytest-anyio` or `pytest-asyncio` with `asyncio_mode = "auto"` in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```
