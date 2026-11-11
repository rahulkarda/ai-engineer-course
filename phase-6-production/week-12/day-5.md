# Phase 6 · Week 12 · Day 5 — Operate Under Failure: Chaos Drills, Runbooks, SLOs

> Objective: Inject failures into your agent platform, define SLOs, and write an incident runbook that can guide triage in production.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Production AI systems fail in novel ways: LLMs time out mid-stream, tool servers return corrupt JSON, dependency chains stall silently. Traditional "error rate" SLOs miss these. You need failure-mode-aware SLOs and runbooks written before the incident, not during it.

### Chaos engineering for AI systems

Chaos engineering is controlled, hypothesis-driven failure injection. The loop:

```
Hypothesis → Inject → Observe → Measure → Fix (or accept)
```

Failure modes worth injecting for an agent system:

| Failure mode | Injection technique | What to observe |
|---|---|---|
| LLM timeout | Wrap client; randomly raise `asyncio.TimeoutError` | Retry logic, fallback model, circuit breaker |
| Tool returns 500 | Patch tool function; return `ToolError` | Error propagation, partial plan recovery |
| Tool returns malformed JSON | Inject corrupt payload | Pydantic validation path, fallback |
| Dependency crash (DB, cache) | Raise `ConnectionError` on call | Circuit breaker open, graceful degradation |
| Memory/state corruption | Inject stale checkpoint | Idempotency, re-execution safety |
| Slow LLM (high latency) | `asyncio.sleep(N)` before response | p99 latency SLO, timeout budget |

### SLOs for agent systems

An **SLI** (Service Level Indicator) is a measured fact. An **SLO** (Service Level Objective) is the target. An **SLA** (Service Level Agreement) is the contractual consequence of missing the SLO.

Examples:

| SLI | SLO | Measurement window |
|---|---|---|
| Request success rate | ≥ 99.0% | 7-day rolling |
| p99 end-to-end latency | ≤ 15 s | 24-hour rolling |
| Tool call success rate | ≥ 99.5% | 7-day rolling |
| LLM retry rate | ≤ 5% of requests | 1-hour rolling |
| Mean turns per task | ≤ 8 | 7-day rolling |

SLO budget math: if your SLO is 99.0% over 30 days, you have 0.01 × 30 × 24 × 60 = **432 minutes** of error budget. Spend it deliberately on deployments; protect it from unplanned incidents.

### Rollback strategies

| Strategy | How it works | When to use |
|---|---|---|
| Blue/green | Keep old deployment live; switch traffic | Zero-downtime, instant rollback |
| Feature flags | Gate new behaviour behind a flag; flip to off | Gradual rollout, instant disable |
| Model version pin | Lock `model=claude-sonnet-5-5` in config; don't auto-update | Model regression |
| Canary | Route 5% of traffic to new version; promote on SLO pass | High-risk changes |

For model version pinning: always specify the exact model ID in your config (`claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`). Never rely on alias pointers like `claude-latest` in production — they change.

## How it works internally

### Error budget burn rate alert

An alert fires when your error budget is burning faster than sustainable:

```
burn_rate = (1 - current_success_rate) / (1 - SLO_target)
```

If `burn_rate > 14.4`, you will exhaust the 30-day budget in 50 hours. Page on burn rate, not on raw error count — it tells you *how fast* you're heading toward an SLO breach.

### Why circuit breakers alone aren't enough

Circuit breakers protect against cascading failure but don't measure SLO compliance. A circuit breaker in the HALF_OPEN state may allow requests through at a trickle that still burns your error budget. Always couple circuit breakers with SLO tracking.

## Worked example: chaos test harness

```python
"""
week_12/chaos_harness.py

Chaos test harness: inject random LLM failures and measure SLO compliance.
Env: ANTHROPIC_API_KEY (not used during chaos runs — LLM is mocked)
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, patch

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── SLO configuration ────────────────────────────────────────────────────────

@dataclass
class SLOConfig:
    success_rate_target: float = 0.99       # 99.0% success rate
    p99_latency_target_s: float = 15.0      # 15 s p99 end-to-end
    tool_success_rate_target: float = 0.995 # 99.5% tool success rate
    window_seconds: float = 300.0           # 5-minute rolling window (demo)


# ── Metrics collector ────────────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    started_at: float
    ended_at: float | None = None
    success: bool = False
    error_type: str | None = None

    @property
    def latency_s(self) -> float:
        if self.ended_at is None:
            return 0.0
        return self.ended_at - self.started_at


@dataclass
class SLOTracker:
    config: SLOConfig = field(default_factory=SLOConfig)
    _requests: list[RequestMetrics] = field(default_factory=list)

    def record(self, m: RequestMetrics) -> None:
        self._requests.append(m)
        self._evict_old()

    def _evict_old(self) -> None:
        cutoff = time.monotonic() - self.config.window_seconds
        self._requests = [r for r in self._requests if r.started_at >= cutoff]

    def success_rate(self) -> float:
        if not self._requests:
            return 1.0
        return sum(1 for r in self._requests if r.success) / len(self._requests)

    def p99_latency(self) -> float:
        latencies = sorted(r.latency_s for r in self._requests if r.ended_at)
        if not latencies:
            return 0.0
        idx = max(0, int(len(latencies) * 0.99) - 1)
        return latencies[idx]

    def slo_status(self) -> dict[str, Any]:
        sr = self.success_rate()
        p99 = self.p99_latency()
        return {
            "total_requests": len(self._requests),
            "success_rate": round(sr, 4),
            "success_rate_ok": sr >= self.config.success_rate_target,
            "p99_latency_s": round(p99, 3),
            "p99_latency_ok": p99 <= self.config.p99_latency_target_s,
            "all_slos_met": (
                sr >= self.config.success_rate_target
                and p99 <= self.config.p99_latency_target_s
            ),
        }


# ── Failure injection ────────────────────────────────────────────────────────

class FailureMode(Enum):
    NONE = "none"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    TOOL_FAILURE = "tool_failure"
    SLOW_RESPONSE = "slow_response"


@dataclass
class ChaosConfig:
    failure_rate: float = 0.2           # 20% of requests injected with a failure
    failure_modes: list[FailureMode] = field(default_factory=lambda: [
        FailureMode.LLM_TIMEOUT,
        FailureMode.LLM_RATE_LIMIT,
        FailureMode.SLOW_RESPONSE,
    ])
    slow_response_s: float = 6.0        # injected latency for SLOW_RESPONSE
    timeout_budget_s: float = 10.0      # agent-side timeout budget


def pick_failure(cfg: ChaosConfig) -> FailureMode:
    """Randomly choose a failure mode based on configured rate."""
    if random.random() < cfg.failure_rate:
        return random.choice(cfg.failure_modes)
    return FailureMode.NONE


# ── Agent under test ─────────────────────────────────────────────────────────

async def call_llm(prompt: str, client: anthropic.AsyncAnthropic) -> str:
    """Thin wrapper around the LLM call — injectable for chaos."""
    message = await client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text  # type: ignore[index]


async def run_agent_request(
    prompt: str,
    client: anthropic.AsyncAnthropic,
    chaos: ChaosConfig,
    tracker: SLOTracker,
) -> str | None:
    """Run one agent request with chaos injection and SLO tracking."""
    m = RequestMetrics(started_at=time.monotonic())
    failure = pick_failure(chaos)
    log.debug("Injecting failure mode: %s", failure.value)

    try:
        if failure == FailureMode.LLM_TIMEOUT:
            raise asyncio.TimeoutError("chaos: simulated LLM timeout")

        if failure == FailureMode.LLM_RATE_LIMIT:
            raise anthropic.RateLimitError(  # type: ignore[call-arg]
                "chaos: simulated rate limit",
                response=None,  # type: ignore[arg-type]
                body=None,
            )

        if failure == FailureMode.SLOW_RESPONSE:
            await asyncio.sleep(chaos.slow_response_s)

        result = await asyncio.wait_for(
            call_llm(prompt, client),
            timeout=chaos.timeout_budget_s,
        )
        m.success = True
        return result

    except asyncio.TimeoutError:
        m.error_type = "timeout"
        log.warning("Request timed out after %.1f s", chaos.timeout_budget_s)
        return None

    except anthropic.RateLimitError:
        m.error_type = "rate_limit"
        log.warning("Rate limited by LLM API")
        return None

    except Exception as exc:
        m.error_type = type(exc).__name__
        log.exception("Unexpected error: %s", exc)
        return None

    finally:
        m.ended_at = time.monotonic()
        tracker.record(m)


# ── Chaos drill ──────────────────────────────────────────────────────────────

async def run_chaos_drill(
    n_requests: int = 50,
    failure_rate: float = 0.25,
) -> dict[str, Any]:
    """
    Run a chaos drill: fire n_requests with injected failures.
    LLM calls are mocked — no real API key needed.
    Returns SLO status at the end.
    """
    slo = SLOConfig()
    tracker = SLOTracker(config=slo)
    chaos = ChaosConfig(failure_rate=failure_rate)

    mock_response = AsyncMock()
    mock_response.content = [AsyncMock(text="mock agent response")]

    with patch.object(
        anthropic.AsyncAnthropic,
        "messages",
        new_callable=lambda: property(lambda self: AsyncMock(
            create=AsyncMock(return_value=mock_response)
        )),
    ):
        client = anthropic.AsyncAnthropic(api_key="mock-key-for-chaos-drill")
        tasks = [
            run_agent_request(f"prompt-{i}", client, chaos, tracker)
            for i in range(n_requests)
        ]
        await asyncio.gather(*tasks)

    status = tracker.slo_status()
    log.info("=== Chaos Drill Results ===")
    log.info("Total requests:     %d", status["total_requests"])
    log.info("Success rate:       %.1f%%  (SLO: %.1f%%)  %s",
             status["success_rate"] * 100,
             slo.success_rate_target * 100,
             "PASS" if status["success_rate_ok"] else "FAIL")
    log.info("p99 latency:        %.2f s  (SLO: %.1f s)  %s",
             status["p99_latency_s"],
             slo.p99_latency_target_s,
             "PASS" if status["p99_latency_ok"] else "FAIL")
    log.info("All SLOs met:       %s", status["all_slos_met"])
    return status


if __name__ == "__main__":
    asyncio.run(run_chaos_drill(n_requests=50, failure_rate=0.25))
```

Expected output (approximate — randomised):
```
INFO  Total requests:     50
INFO  Success rate:       76.0%  (SLO: 99.0%)  FAIL
INFO  p99 latency:        6.01 s  (SLO: 15.0 s)  PASS
INFO  All SLOs met:       False
```

The drill surfaces that a 25% failure rate obliterates the 99% success SLO immediately. Real mitigation: retries with backoff + circuit breaker reduces observable failure rate.

## Incident runbook template

An incident runbook is a living document. Write it when calm; use it when paging:

```markdown
# Runbook: LLM Timeout Spike

## Severity triggers
- P1: success rate < 95% for > 5 minutes
- P2: success rate 95–99% for > 15 minutes

## Symptoms
- Error budget burn rate > 14.4
- `error_type=timeout` in logs at elevated rate
- p99 latency approaching SLO boundary

## Detection
1. Check SLO dashboard: success rate + burn rate
2. `grep 'error_type.*timeout' /var/log/app.log | tail -50`
3. Check Anthropic status page: https://status.anthropic.com

## Triage (5 min)
- Is it a single model? Try switching to fallback model in feature flag
- Is it all requests or a subset (specific tool, user, region)?
- Has a deploy happened in the last 30 min? (`git log --oneline -5`)

## Mitigation options (ordered by blast radius)
1. Flip feature flag `use_fallback_model=true` → routes to haiku-5-5 (lower latency)
2. Reduce `max_tokens` in model config (decreases TTFT)
3. Increase `timeout_budget_s` if LLM is slow but succeeding
4. Roll back to previous container image if correlated with deploy

## Rollback procedure
```bash
# Blue/green: swap traffic back to stable target group
aws elbv2 modify-listener --listener-arn $LISTENER_ARN \
  --default-actions Type=forward,TargetGroupArn=$STABLE_TG_ARN

# Or via feature flag (instant, no deploy)
curl -X POST https://flags.internal/api/features/use_fallback_model \
  -d '{"enabled": true}'
```

## Escalation
- 15 min no improvement → page on-call senior engineer
- SLO breach confirmed → open incident in PagerDuty, notify stakeholders

## Post-incident
- Write blameless post-mortem within 48 hr
- Add failing scenario to chaos harness
- Update SLO alert thresholds if needed
```

## Exercises

1. Extend `ChaosConfig` to support `FailureMode.TOOL_FAILURE`. Add a `run_tool()` async function that can be injected with this failure. Measure how tool failure rate affects overall request success rate.

2. Implement the **error budget burn rate** formula in `SLOTracker`. Add a method `burn_rate() -> float` and a method `minutes_until_budget_exhausted() -> float | None`. Log a warning when burn rate exceeds 14.4.

3. Add a `FallbackChaosConfig` that switches to `claude-haiku-5-5` instead of failing on `LLM_TIMEOUT`. Rerun the chaos drill. What happens to the success rate? What is the latency trade-off?

4. Sketch a blue/green rollback for your agent API using feature flags. What does the feature flag control (model ID, tool set, prompt version)? Write a `FeatureFlagClient` stub with `get(flag_name: str) -> bool` backed by an environment variable.

(Solutions: ../solutions/week-12/day-5.md)

## Self-check

- [ ] I can define SLI, SLO, and SLA and give a concrete agent example of each.
- [ ] I can calculate error budget minutes from an SLO target and window length.
- [ ] I can explain burn rate and why it's a better alert signal than raw error count.
- [ ] I can write a chaos test that injects LLM timeouts and measures SLO compliance.
- [ ] I can write the first three steps of an incident runbook from memory.
- [ ] I can describe blue/green and feature-flag rollback and when to use each.

---

## Supplementary links

- [Google SRE Book — SLOs](https://sre.google/sre-book/service-level-objectives/)
- [Principles of Chaos Engineering](https://principlesofchaos.org/)
- [Anthropic Status Page](https://status.anthropic.com)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
