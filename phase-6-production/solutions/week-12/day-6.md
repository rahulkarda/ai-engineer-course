# Week 12 · Day 6 — Solutions: Saturday Deep Practice (Harden + Operate)

This file contains reference implementations for each of the 8 steps in the Saturday deep-practice session.

---

## Step 1 Reference: STRIDE Threat Model as Pydantic Code

```python
"""
solutions/week-12/day6_threat_model.py
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class STRIDECategory(str, Enum):
    S = "Spoofing"; T = "Tampering"; R = "Repudiation"
    I = "Information Disclosure"; D = "Denial of Service"; E = "Elevation of Privilege"


class Threat(BaseModel):
    id: str
    category: STRIDECategory
    component: str
    description: str
    mitigation: str
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)

    @property
    def risk_score(self) -> int:
        return self.likelihood * self.impact


class ThreatModel(BaseModel):
    system: str
    threats: list[Threat]

    def high_priority(self) -> list[Threat]:
        return [t for t in self.threats if t.risk_score >= 12]


PLATFORM_THREAT_MODEL = ThreatModel(
    system="Hardened Agent Platform",
    threats=[
        Threat(id="S-01", category=STRIDECategory.S, component="Agent identity",
               description="Rogue agent impersonates trusted summarizer",
               mitigation="Sign agent messages with HMAC; verify at planner",
               likelihood=3, impact=4),
        Threat(id="T-01", category=STRIDECategory.T, component="Document retrieval",
               description="Indirect prompt injection via retrieved document",
               mitigation="Document boundary wrapping + injection classifier",
               likelihood=4, impact=5),
        Threat(id="T-02", category=STRIDECategory.T, component="Tool parameters",
               description="Path traversal via malicious file path argument",
               mitigation="Allowlist-based path validation in ToolSandbox",
               likelihood=3, impact=5),
        Threat(id="R-01", category=STRIDECategory.R, component="Audit log",
               description="No tamper-evident record of agent actions",
               mitigation="Append-only structured log with request IDs",
               likelihood=3, impact=3),
        Threat(id="I-01", category=STRIDECategory.I, component="LLM context",
               description="PII leaks from retrieved documents into output",
               mitigation="PIIScrubber on all LLM outputs",
               likelihood=4, impact=3),
        Threat(id="D-01", category=STRIDECategory.D, component="Planner",
               description="Unbounded tool-call loop causes DoS",
               mitigation="Max-steps limit + per-user rate limit",
               likelihood=3, impact=4),
        Threat(id="E-01", category=STRIDECategory.E, component="Code execution tool",
               description="Agent runs arbitrary OS commands",
               mitigation="Sandboxed execution environment; no network access",
               likelihood=3, impact=5),
    ],
)
```

---

## Step 2 Reference: Injection Defense Pipeline

```python
"""
solutions/week-12/day6_injection_defense.py

Complete two-stage injection defense: regex pre-filter + boundary wrapping.
"""
from __future__ import annotations
import re
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions?"),
    re.compile(r"(?i)new\s+system\s+instruction\s*:"),
    re.compile(r"(?i)you\s+are\s+now\s+.{3,50}"),
    re.compile(r"(?i)from\s+now\s+on\s+you\s+(must|will|should)"),
    re.compile(r"(?i)act\s+as\s+a?\s*.{3,40}"),
    re.compile(r"(?i)disregard\s+(your\s+)?(previous|system)\s+(instructions?|prompt)"),
    re.compile(r"(?i)</?(instructions?|system|prompt)>"),
    re.compile(r"(?i)jailbreak|DAN\s+mode"),
    re.compile(r"(?i)repeat\s+(back\s+)?verbatim"),
]


class InputValidationResult(BaseModel):
    original: str
    cleaned: str
    injection_detected: bool
    matched_pattern_count: int


def validate_user_input(text: str) -> InputValidationResult:
    matched = sum(1 for p in _INJECTION_PATTERNS if p.search(text))
    is_injection = matched > 0
    if is_injection:
        logger.warning("Direct injection detected. pattern_matches=%d", matched)
    return InputValidationResult(
        original=text,
        cleaned=text,  # for direct input, we block rather than clean
        injection_detected=is_injection,
        matched_pattern_count=matched,
    )


def wrap_document_content(url: str, content: str) -> str:
    """Wrap retrieved content so the LLM treats it as data, not instructions."""
    safe = re.sub("|".join(p.pattern for p in _INJECTION_PATTERNS), "[FILTERED]", content)
    return f'<document source="{url}">\n{safe}\n</document>'
```

---

## Step 3 Reference: Tool Sandboxing with Allowlist

```python
"""
solutions/week-12/day6_tool_sandbox.py

Minimal ToolSandbox for the platform integration test.
"""
from __future__ import annotations
import logging
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SandboxViolation(Exception):
    pass


class ToolSandbox(BaseModel):
    allowed_path_prefixes: list[str] = Field(default_factory=list)
    allowed_url_prefixes: list[str] = Field(default_factory=list)
    max_query_length: int = 1000

    def validate_file_path(self, path: str) -> Path:
        p = Path(path).resolve()
        if not any(str(p).startswith(prefix) for prefix in self.allowed_path_prefixes):
            logger.warning("Path violation blocked: %s", path)
            raise SandboxViolation(f"Path '{p}' outside allowed prefixes")
        return p

    def validate_url(self, url: str) -> str:
        if not any(url.startswith(prefix) for prefix in self.allowed_url_prefixes):
            logger.warning("URL violation blocked: %s", url)
            raise SandboxViolation(f"URL '{url}' outside allowed prefixes")
        return url

    def validate_search_query(self, query: str) -> str:
        if len(query) > self.max_query_length:
            raise SandboxViolation(f"Query length {len(query)} exceeds limit {self.max_query_length}")
        return query
```

---

## Step 4 Reference: PII Guardrail Layer

(Full implementation in day-4 solutions. Integration version shown here.)

```python
"""
solutions/week-12/day6_guardrail_integration.py

Integrate PIIScrubber as a middleware around any LLM call function.
"""
from __future__ import annotations
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"(?<!\d)\d{3}[-\s]\d{2}[-\s]\d{4}(?!\d)"),  # SSN
    re.compile(r"(?<!\d)(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"),  # phone
]


def scrub_pii(text: str) -> tuple[str, int]:
    """Returns (scrubbed_text, count_of_replacements)."""
    count = 0
    for pat in _PII_PATTERNS:
        new_text, n = pat.subn("[REDACTED]", text)
        count += n
        text = new_text
    return text, count


def guardrail(fn: Callable[[str], str]) -> Callable[[str], str]:
    """Decorator: scrub PII from input and output of any text→text function."""
    def wrapper(prompt: str) -> str:
        clean_in, in_count = scrub_pii(prompt)
        if in_count:
            logger.warning("PII scrubbed from input: %d items", in_count)
        raw_output = fn(clean_in)
        clean_out, out_count = scrub_pii(raw_output)
        if out_count:
            logger.warning("PII scrubbed from output: %d items", out_count)
        return clean_out
    return wrapper
```

---

## Steps 5–8 Reference: Full Integration

```python
"""
solutions/week-12/day6_hardened_platform.py

Full hardened agent platform integrating:
- STRIDE threat model
- Injection defense
- Tool sandbox
- PII guardrail
- Prometheus-style metrics
- FastAPI endpoints
- Chaos drill
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from day6_injection_defense import validate_user_input, wrap_document_content
from day6_tool_sandbox import ToolSandbox, SandboxViolation
from day6_guardrail_integration import scrub_pii
from day6_threat_model import PLATFORM_THREAT_MODEL

logger = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────────────────────

_counters: dict[str, int] = defaultdict(int)
_latencies: list[float] = []


def inc(metric: str, n: int = 1) -> None:
    _counters[metric] += n


def record_latency(ms: float) -> None:
    _latencies.append(ms)
    if len(_latencies) > 10_000:
        _latencies.pop(0)


def get_slo_status() -> dict:
    if not _latencies:
        return {"status": "no_data"}
    total = _counters["requests.total"]
    failures = _counters["requests.failed"]
    success_rate = 1.0 if total == 0 else (total - failures) / total
    sorted_l = sorted(_latencies)
    p99 = sorted_l[int(len(sorted_l) * 0.99)] if sorted_l else 0.0
    return {
        "success_rate": success_rate,
        "p99_latency_ms": p99,
        "slo_success_met": success_rate >= 0.99,
        "slo_p99_met": p99 <= 2000.0,
        "error_budget_remaining": max(0.0, 1.0 - (1 - success_rate) / 0.01),
    }


# ── Sandbox config ────────────────────────────────────────────────────────────

SANDBOX = ToolSandbox(
    allowed_path_prefixes=["/tmp/agent-data"],
    allowed_url_prefixes=["https://api.example.com/"],
)

# ── FastAPI app ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Hardened agent platform starting")
    yield
    logger.info("Hardened agent platform shutting down")


app = FastAPI(title="Hardened Agent Platform", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: str


class ChatResponse(BaseModel):
    reply: str
    security_events: list[str]


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest) -> ChatResponse:
    inc("requests.total")
    start = time.monotonic()
    events: list[str] = []

    try:
        # 1. Injection check
        validation = validate_user_input(req.message)
        if validation.injection_detected:
            inc("security.injections_blocked")
            events.append("injection_blocked")
            raise HTTPException(
                status_code=400,
                detail={"error": "prompt_injection_detected"},
            )

        # 2. PII scrub input
        clean_msg, pii_count = scrub_pii(req.message)
        if pii_count:
            events.append(f"pii_scrubbed_input:{pii_count}")

        # 3. Simulated LLM call (mock)
        await asyncio.sleep(0.01)
        raw_reply = f"Processed: {clean_msg[:50]}"

        # 4. PII scrub output
        clean_reply, out_pii = scrub_pii(raw_reply)
        if out_pii:
            events.append(f"pii_scrubbed_output:{out_pii}")

        duration_ms = (time.monotonic() - start) * 1000
        record_latency(duration_ms)
        return ChatResponse(reply=clean_reply, security_events=events)

    except HTTPException:
        inc("requests.failed")
        duration_ms = (time.monotonic() - start) * 1000
        record_latency(duration_ms)
        raise
    except Exception as exc:
        inc("requests.failed")
        duration_ms = (time.monotonic() - start) * 1000
        record_latency(duration_ms)
        logger.exception("Unhandled error in agent_chat")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
async def metrics() -> dict:
    return dict(_counters)


@app.get("/slo")
async def slo() -> dict:
    return get_slo_status()


@app.get("/threat-model")
async def threat_model() -> dict:
    return PLATFORM_THREAT_MODEL.model_dump()


# ── Chaos drill ───────────────────────────────────────────────────────────────

async def run_chaos_drill(n_requests: int = 100, failure_rate: float = 0.15) -> dict:
    """Simulate n_requests with injected failures and report outcomes."""
    successes = 0
    failures: dict[str, int] = defaultdict(int)

    for _ in range(n_requests):
        r = random.random()
        if r < failure_rate * 0.33:
            failures["llm_timeout"] += 1
        elif r < failure_rate * 0.66:
            failures["tool_server_error"] += 1
        elif r < failure_rate:
            failures["dependency_crash"] += 1
        else:
            successes += 1

    return {
        "total": n_requests,
        "successes": successes,
        "failures": dict(failures),
        "success_rate": successes / n_requests,
        "slo_met": (successes / n_requests) >= 0.99,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Step 8 Reference: Incident Runbook Template

```markdown
# Incident Runbook: Agent Platform

## SEV1 — Agent Platform Completely Down

### Detection
- Alert: `AgentSuccessRate < 0.50` for 2 minutes
- Symptoms: All `/agent/chat` requests return 500

### Triage
1. Check pod health: `kubectl get pods -l app=agent`
2. Check recent deployments: `kubectl rollout history deployment/agent`
3. Check LLM API status: https://status.anthropic.com

### Mitigation
1. If bad deploy: `kubectl rollout undo deployment/agent`
2. If LLM API down: Enable mock/fallback mode via feature flag
3. If database issue: Restore from last snapshot

### Rollback steps
1. `kubectl rollout undo deployment/agent --to-revision=N`
2. Verify: `kubectl rollout status deployment/agent`
3. Confirm SLO recovery in Grafana dashboard

### Communication template
**[SEV1] Agent Platform Down**
Impact: 100% of agent requests failing
Current status: Investigating
Action: [FILL IN]
ETA: [FILL IN]
Next update: in 15 minutes
```
