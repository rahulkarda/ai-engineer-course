# Phase 6 · Week 12 · Day 3 — Tool Abuse, Excessive Agency & Tool Sandboxing

> Objective: Understand how agents can be weaponized through their tools, and implement a sandboxed FastAPI tool server with least-privilege design and audit logging.  ·  Time: ~1 hr  ·  Prereqs: [[week-12/day-2]]

---

## Concept

A tool-calling agent's blast radius is determined by what its tools can do. Grant an agent a file-write tool with no path restrictions and a crafted prompt injection can overwrite `/etc/passwd`. Grant it an HTTP-fetch tool with no URL allowlist and it becomes an SSRF proxy. The tool interface is the actuator — it is where intent becomes effect.

**Excessive agency** (OWASP LLM08) is the root cause: agents are given broader permissions than any single task requires, then errors or attacks in prompt handling cause those permissions to be misused.

**Tool abuse** is the exploitation path: an attacker shapes tool parameters via prompt injection or crafted input to make a legitimate tool do something unintended.

---

## How it works internally

### Excessive agency failure modes

Three sources of excess permissions in practice:

1. **Scope creep**: developer adds tools for convenience; none are removed; agent accumulates permissions over time.
2. **Implicit trust of LLM output**: tool parameters constructed directly from LLM response without validation; LLM is not a sanitizer.
3. **Overly broad tool definitions**: a `run_command(cmd: str)` tool that accepts arbitrary shell commands will eventually be exploited.

### Tool abuse attack classes

| Attack | Mechanism | Example |
|--------|-----------|---------|
| Path traversal | `../` sequences in file path parameter | `read_file(path="../../etc/shadow")` |
| SSRF | Internal URL in HTTP fetch tool | `fetch_url(url="http://169.254.169.254/latest/meta-data/")` |
| Command injection | Shell metacharacters in tool param | `run_shell(cmd="ls; cat /etc/passwd")` |
| Parameter pollution | Extra fields ignored by schema but consumed by downstream | `create_user(role="admin")` when role is not expected |
| Resource exhaustion | Recursive or unbounded operations | `list_directory(path="/", recursive=True)` |

### Sandboxing principles

**Allowlist over blocklist.** Define exactly what is permitted; reject everything else. Blocklists always have gaps.

**Validate at the boundary.** Validate every tool call parameter *before* execution — not inside the tool implementation. The gate is before the door.

**Least-privilege credentials.** Tool calls use credentials scoped to the minimum necessary. A read-only query does not use a write-capable DB connection.

**Execution limits.** Timeouts, CPU limits, and output size caps prevent resource abuse.

**Human-in-the-loop gates.** Irreversible or high-impact actions pause for human approval. The pause is part of the tool contract, not an afterthought.

**Audit every call.** Emit a structured log entry for every tool call regardless of outcome. Audit logs are the primary detection signal.

---

## Worked example

A FastAPI tool server with sandboxed execution, parameter validation, and structured audit logging. Two tools: `read_file` and `fetch_url`, each with strict allowlists and validation.

```python
# tool_server.py
# Sandboxed FastAPI tool server with least-privilege design and audit logging.
# Usage: uvicorn tool_server:app --host 0.0.0.0 --port 8000
# Requires: pip install fastapi uvicorn pydantic httpx

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Sandboxed Tool Server", version="1.0.0")

# ── Security configuration ─────────────────────────────────────────────────────

# Allowlist of directories the read_file tool may access.
# Real deployment: load from environment or secrets manager.
ALLOWED_READ_DIRS: list[Path] = [
    Path("/tmp/agent-workspace").resolve(),
    Path("/var/agent/data").resolve(),
]

# Allowlist of URL prefixes the fetch_url tool may contact.
ALLOWED_URL_PREFIXES: list[str] = [
    "https://api.github.com/",
    "https://api.openweathermap.org/",
]

# Blocklist of internal/cloud-metadata address ranges — belt-and-suspenders
BLOCKED_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"https?://169\.254\.", re.IGNORECASE),        # AWS/GCP metadata
    re.compile(r"https?://10\.\d+\.\d+\.\d+", re.IGNORECASE),  # RFC-1918 10/8
    re.compile(r"https?://192\.168\.", re.IGNORECASE),          # RFC-1918 192.168/16
    re.compile(r"https?://172\.(1[6-9]|2\d|3[01])\.", re.IGNORECASE),  # RFC-1918 172.16/12
    re.compile(r"https?://localhost", re.IGNORECASE),
    re.compile(r"https?://127\.", re.IGNORECASE),
]

MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
HTTP_TIMEOUT_SECONDS = 10.0
MAX_HTTP_RESPONSE_BYTES = 512 * 1024   # 512 KB

# ── Audit logging ──────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    audit_id: str
    timestamp: str
    tool_name: str
    caller_ip: str
    parameters: dict[str, Any]   # redacted of any secret values
    outcome: str                  # "success" | "blocked" | "error"
    reason: str = ""
    duration_ms: float = 0.0


def emit_audit(entry: AuditEntry) -> None:
    """
    Write a structured audit log entry.
    In production: forward to append-only SIEM (Splunk, Datadog, CloudWatch Logs).
    """
    logger.info(
        "AUDIT tool=%s caller=%s outcome=%s reason=%s params=%s",
        entry.tool_name,
        entry.caller_ip,
        entry.outcome,
        entry.reason or "-",
        entry.parameters,
    )


# ── Tool request/response schemas ─────────────────────────────────────────────

class ReadFileRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def path_no_traversal(cls, v: str) -> str:
        # Reject obvious traversal sequences before path resolution
        if ".." in v or v.startswith("/") and not any(
            v.startswith(str(d)) for d in ALLOWED_READ_DIRS
        ):
            # Note: full validation happens after resolution in the endpoint
            pass
        return v


class ReadFileResponse(BaseModel):
    path: str
    content: str
    size_bytes: int


class FetchUrlRequest(BaseModel):
    url: str
    method: str = "GET"
    headers: dict[str, str] = {}

    @field_validator("method")
    @classmethod
    def method_allowlist(cls, v: str) -> str:
        if v.upper() not in ("GET", "HEAD"):
            raise ValueError(f"method '{v}' not permitted; only GET and HEAD allowed")
        return v.upper()

    @model_validator(mode="after")
    def strip_dangerous_headers(self) -> "FetchUrlRequest":
        # Prevent header injection or credential forwarding
        dangerous = {"authorization", "cookie", "x-api-key", "x-auth-token"}
        self.headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in dangerous
        }
        return self


class FetchUrlResponse(BaseModel):
    url: str
    status_code: int
    body_preview: str      # truncated to MAX_HTTP_RESPONSE_BYTES
    content_type: str


# ── Validation helpers ─────────────────────────────────────────────────────────

def validate_file_path(raw_path: str) -> Path:
    """
    Resolve path and verify it falls within an allowed directory.
    Raises HTTPException if validation fails.
    """
    try:
        resolved = Path(raw_path).resolve()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid path: {exc}",
        )

    for allowed_dir in ALLOWED_READ_DIRS:
        try:
            resolved.relative_to(allowed_dir)
            return resolved
        except ValueError:
            continue

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"Path '{resolved}' is outside allowed directories. "
            f"Permitted roots: {[str(d) for d in ALLOWED_READ_DIRS]}"
        ),
    )


def validate_url(url: str) -> None:
    """
    Validate URL against allowlist prefixes and blocklist patterns.
    Raises HTTPException if validation fails.
    """
    for pattern in BLOCKED_URL_PATTERNS:
        if pattern.search(url):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"URL '{url}' targets a blocked address range (SSRF protection).",
            )

    if not any(url.startswith(prefix) for prefix in ALLOWED_URL_PREFIXES):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"URL '{url}' is not in the allowed prefix list. "
                f"Permitted prefixes: {ALLOWED_URL_PREFIXES}"
            ),
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/tools/read_file", response_model=ReadFileResponse)
async def read_file(req_body: ReadFileRequest, request: Request) -> ReadFileResponse:
    """Read a file from the sandboxed workspace."""
    audit_id = str(uuid.uuid4())
    caller_ip = request.client.host if request.client else "unknown"
    start = datetime.now(timezone.utc)

    try:
        resolved = validate_file_path(req_body.path)

        if not resolved.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {resolved}",
            )
        if not resolved.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a regular file: {resolved}",
            )

        size = resolved.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size {size} bytes exceeds limit {MAX_FILE_SIZE_BYTES}.",
            )

        content = resolved.read_text(encoding="utf-8", errors="replace")
        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        emit_audit(AuditEntry(
            audit_id=audit_id,
            timestamp=start.isoformat(),
            tool_name="read_file",
            caller_ip=caller_ip,
            parameters={"path": str(resolved)},
            outcome="success",
            duration_ms=duration_ms,
        ))

        return ReadFileResponse(path=str(resolved), content=content, size_bytes=size)

    except HTTPException as exc:
        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        emit_audit(AuditEntry(
            audit_id=audit_id,
            timestamp=start.isoformat(),
            tool_name="read_file",
            caller_ip=caller_ip,
            parameters={"path": req_body.path},
            outcome="blocked",
            reason=str(exc.detail),
            duration_ms=duration_ms,
        ))
        raise


@app.post("/tools/fetch_url", response_model=FetchUrlResponse)
async def fetch_url(req_body: FetchUrlRequest, request: Request) -> FetchUrlResponse:
    """Fetch a URL from the approved external service list."""
    audit_id = str(uuid.uuid4())
    caller_ip = request.client.host if request.client else "unknown"
    start = datetime.now(timezone.utc)

    try:
        validate_url(req_body.url)

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.request(
                method=req_body.method,
                url=req_body.url,
                headers={**req_body.headers, "User-Agent": "AgentToolServer/1.0"},
                follow_redirects=False,  # do not follow redirects — could bypass URL validation
            )

        body = response.text[:MAX_HTTP_RESPONSE_BYTES]
        content_type = response.headers.get("content-type", "unknown")
        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        emit_audit(AuditEntry(
            audit_id=audit_id,
            timestamp=start.isoformat(),
            tool_name="fetch_url",
            caller_ip=caller_ip,
            parameters={"url": req_body.url, "method": req_body.method},
            outcome="success",
            duration_ms=duration_ms,
        ))

        return FetchUrlResponse(
            url=req_body.url,
            status_code=response.status_code,
            body_preview=body,
            content_type=content_type,
        )

    except HTTPException as exc:
        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        emit_audit(AuditEntry(
            audit_id=audit_id,
            timestamp=start.isoformat(),
            tool_name="fetch_url",
            caller_ip=caller_ip,
            parameters={"url": req_body.url},
            outcome="blocked",
            reason=str(exc.detail),
            duration_ms=duration_ms,
        ))
        raise
    except httpx.TimeoutException:
        emit_audit(AuditEntry(
            audit_id=audit_id,
            timestamp=start.isoformat(),
            tool_name="fetch_url",
            caller_ip=caller_ip,
            parameters={"url": req_body.url},
            outcome="error",
            reason="timeout",
        ))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request to '{req_body.url}' timed out after {HTTP_TIMEOUT_SECONDS}s.",
        )


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

**Testing the sandboxing (without a running server):**

```python
# test_tool_server.py
# Unit tests for path and URL validation logic.
# Usage: pytest test_tool_server.py -v
# Requires: pip install pytest fastapi httpx

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the validation functions directly — no network needed.
from tool_server import validate_file_path, validate_url, app

client = TestClient(app)


class TestPathValidation:
    def test_path_traversal_blocked(self) -> None:
        with pytest.raises(Exception) as exc_info:
            validate_file_path("../../etc/passwd")
        assert "403" in str(exc_info.value.status_code) or "outside allowed" in str(exc_info.value.detail)

    def test_absolute_outside_allowed_blocked(self) -> None:
        with pytest.raises(Exception) as exc_info:
            validate_file_path("/etc/shadow")
        assert exc_info.value.status_code == 403

    def test_allowed_path_resolves(self, tmp_path: Path) -> None:
        # Temporarily add tmp_path to allowed dirs for this test
        from tool_server import ALLOWED_READ_DIRS
        ALLOWED_READ_DIRS.append(tmp_path.resolve())
        test_file = tmp_path / "safe.txt"
        test_file.write_text("hello")
        resolved = validate_file_path(str(test_file))
        assert resolved == test_file.resolve()
        ALLOWED_READ_DIRS.remove(tmp_path.resolve())


class TestUrlValidation:
    def test_ssrf_metadata_blocked(self) -> None:
        with pytest.raises(Exception) as exc_info:
            validate_url("http://169.254.169.254/latest/meta-data/")
        assert exc_info.value.status_code == 403

    def test_internal_ip_blocked(self) -> None:
        with pytest.raises(Exception) as exc_info:
            validate_url("http://10.0.0.1/admin")
        assert exc_info.value.status_code == 403

    def test_not_in_allowlist_blocked(self) -> None:
        with pytest.raises(Exception) as exc_info:
            validate_url("https://evil.com/payload")
        assert exc_info.value.status_code == 403

    def test_allowed_url_passes(self) -> None:
        # Should not raise
        validate_url("https://api.github.com/users/octocat")


class TestAuditLogging:
    def test_blocked_request_emits_audit(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.INFO):
            response = client.post(
                "/tools/read_file",
                json={"path": "/etc/passwd"},
            )
        assert response.status_code == 403
        assert any("AUDIT" in r.message and "blocked" in r.message for r in caplog.records)
```

---

## Key design decisions

**Why `follow_redirects=False`?** A URL that passes allowlist checks could redirect to an internal address. Disabling redirect following closes this bypass.

**Why validate *before* the tool runs, not inside it?** Validation at the gate is centralized and testable. Validation inside each tool is easy to miss when adding new tools. The `validate_file_path` and `validate_url` functions are the single source of truth.

**Why emit audit logs on *blocked* requests too?** Blocked attempts are the primary intrusion signal. A spike in blocked `read_file` calls targeting `/etc/` is a detection event. Logging only successful calls misses it.

---

## Exercises

1. **Tool design review**: You are given a tool with signature `run_sql(query: str) -> list[dict]`. It runs against a read-write production database. List every security problem with this design. Then redesign it: restrict to SELECT only, use a read-only DB connection string, enforce a maximum result row count, and add parameter binding (no string interpolation). Implement the redesigned tool function.

2. **Human-in-the-loop gate**: Add a new endpoint `POST /tools/delete_file` that does NOT immediately delete the file. Instead, it creates a pending action record (in memory is fine), logs an audit entry with `outcome="pending_approval"`, and returns the action ID. Add a second endpoint `POST /tools/confirm_action/{action_id}` that a human operator calls to execute the deletion. Implement both endpoints with the same path validation from the worked example.

3. **Scope analysis**: You have an agent that helps users with three tasks: (a) search the company knowledge base, (b) read the user's own calendar events, (c) draft (but not send) email replies. List the exact set of tool permissions this agent needs. Then list five tools that a naïve implementation might accidentally include. For each excess tool, name the worst-case attack if an injection exploited it.

(Solutions: ../solutions/week-12/day-3.md)

---

## Self-check

- [ ] Can name three tool abuse attack classes and give a concrete parameter example for each.
- [ ] Can explain why allowlists are preferred over blocklists for tool sandboxing.
- [ ] Can implement path traversal validation using `Path.resolve()` and `relative_to()`.
- [ ] Can implement SSRF protection using both URL prefix allowlists and IP range blocklists.
- [ ] Can explain why audit logs should include blocked requests, not just successful ones.
- [ ] Can describe the human-in-the-loop gate pattern and name two scenarios where it is required.

---

## Supplementary links

- [OWASP LLM07 — Insecure Plugin Design](https://owasp.org/www-project-top-10-for-large-language-model-applications/LLM07_2025_Insecure_Plugin_Design)
- [OWASP LLM08 — Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/LLM08_2025_Excessive_Agency)
- [OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)
- [FastAPI Security docs](https://fastapi.tiangolo.com/tutorial/security/)
- [Anthropic — Tool use safety](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/implement-tool-use)
