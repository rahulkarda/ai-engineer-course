# Week 12 · Day 5 — Solutions: Chaos Drill, Incident Runbook & SLO Monitoring

## Exercise 1

`ChaosHarness` that injects failures with configurable probability and tracks SLO metrics.

```python
"""
solutions/week-12/chaos_harness.py

ChaosHarness — inject random failures into an agent call path
and measure SLO compliance.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailureMode(str, Enum):
    LLM_TIMEOUT = "llm_timeout"
    TOOL_SERVER_ERROR = "tool_server_error"
    DEPENDENCY_CRASH = "dependency_crash"
    RATE_LIMIT = "rate_limit"


class ChaosConfig(BaseModel):
    enabled: bool = True
    failure_modes: dict[FailureMode, float] = Field(
        default_factory=lambda: {
            FailureMode.LLM_TIMEOUT: 0.05,         # 5% of calls
            FailureMode.TOOL_SERVER_ERROR: 0.03,   # 3%
            FailureMode.DEPENDENCY_CRASH: 0.02,    # 2%
            FailureMode.RATE_LIMIT: 0.04,          # 4%
        }
    )


class CallRecord(BaseModel):
    timestamp: float
    duration_ms: float
    success: bool
    failure_mode: FailureMode | None = None


class SLOMetrics(BaseModel):
    total_calls: int
    successful_calls: int
    success_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    slo_success_target: float = 0.99
    slo_p99_target_ms: float = 2000.0

    @property
    def slo_success_met(self) -> bool:
        return self.success_rate >= self.slo_success_target

    @property
    def slo_latency_met(self) -> bool:
        return self.p99_ms <= self.slo_p99_target_ms

    @property
    def error_budget_remaining(self) -> float:
        """Fraction of error budget remaining (1.0 = all remaining, 0.0 = exhausted)."""
        allowed_failure_rate = 1.0 - self.slo_success_target
        actual_failure_rate = 1.0 - self.success_rate
        if allowed_failure_rate == 0:
            return 0.0
        return max(0.0, 1.0 - actual_failure_rate / allowed_failure_rate)


@dataclass
class ChaosHarness:
    config: ChaosConfig = field(default_factory=ChaosConfig)
    _records: list[CallRecord] = field(default_factory=list)

    def _should_inject(self, mode: FailureMode) -> bool:
        if not self.config.enabled:
            return False
        prob = self.config.failure_modes.get(mode, 0.0)
        return random.random() < prob

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        timeout_ms: float = 5000.0,
        **kwargs: Any,
    ) -> T:
        """
        Wrap an async function call with chaos injection.
        Measures latency and records outcome.
        """
        start = time.monotonic()
        failure: FailureMode | None = None

        try:
            # Check each failure mode in order
            if self._should_inject(FailureMode.LLM_TIMEOUT):
                failure = FailureMode.LLM_TIMEOUT
                logger.warning("[CHAOS] Injecting LLM timeout")
                raise asyncio.TimeoutError("Chaos-injected LLM timeout")

            if self._should_inject(FailureMode.TOOL_SERVER_ERROR):
                failure = FailureMode.TOOL_SERVER_ERROR
                logger.warning("[CHAOS] Injecting tool server error")
                raise RuntimeError("Chaos-injected tool server 500")

            if self._should_inject(FailureMode.DEPENDENCY_CRASH):
                failure = FailureMode.DEPENDENCY_CRASH
                logger.warning("[CHAOS] Injecting dependency crash")
                raise ConnectionError("Chaos-injected dependency crash")

            if self._should_inject(FailureMode.RATE_LIMIT):
                failure = FailureMode.RATE_LIMIT
                logger.warning("[CHAOS] Injecting rate limit error")
                raise RuntimeError("Chaos-injected rate limit 429")

            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_ms / 1000)
            duration_ms = (time.monotonic() - start) * 1000
            self._records.append(CallRecord(
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=True,
            ))
            return result

        except (asyncio.TimeoutError, RuntimeError, ConnectionError):
            duration_ms = (time.monotonic() - start) * 1000
            self._records.append(CallRecord(
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=False,
                failure_mode=failure,
            ))
            raise

    def slo_metrics(self) -> SLOMetrics:
        """Compute current SLO metrics from recorded calls."""
        if not self._records:
            return SLOMetrics(
                total_calls=0, successful_calls=0, success_rate=1.0,
                p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
            )
        durations = sorted(r.duration_ms for r in self._records)
        n = len(durations)
        successful = sum(1 for r in self._records if r.success)

        def percentile(p: float) -> float:
            idx = max(0, int(n * p / 100) - 1)
            return durations[idx]

        return SLOMetrics(
            total_calls=n,
            successful_calls=successful,
            success_rate=successful / n,
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
        )
```

---

## Exercise 2

Incident runbook as structured Pydantic models.

```python
"""
solutions/week-12/incident_runbook.py

Incident runbook modelled as Pydantic v2 data structures.
Serializes to JSON for storage in runbook repos.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    SEV1 = "SEV1"   # production down, all users affected
    SEV2 = "SEV2"   # degraded, major impact
    SEV3 = "SEV3"   # minor degradation
    SEV4 = "SEV4"   # investigation only


class RunbookStep(BaseModel):
    step_number: int
    title: str
    action: str
    expected_outcome: str
    escalate_if: str | None = None
    rollback_step: bool = False


class DetectionCriteria(BaseModel):
    alert_name: str
    threshold: str
    monitoring_link: str


class RunbookEntry(BaseModel):
    incident_name: str
    severity: Severity
    description: str
    detection: list[DetectionCriteria]
    triage_steps: list[RunbookStep]
    mitigation_steps: list[RunbookStep]
    rollback_steps: list[RunbookStep]
    communication_template: str
    owner: str
    last_tested: str | None = None


# ── Example: LLM API quota exhaustion ────────────────────────────────────────

LLM_QUOTA_RUNBOOK = RunbookEntry(
    incident_name="LLM API Quota Exhaustion",
    severity=Severity.SEV2,
    description="Agent requests to the Anthropic API are failing with 429/rate-limit errors, causing agent tasks to fail.",
    detection=[
        DetectionCriteria(
            alert_name="HighLLMErrorRate",
            threshold="LLM error rate > 5% over 5 minutes",
            monitoring_link="https://grafana.internal/d/llm-errors",
        ),
    ],
    triage_steps=[
        RunbookStep(
            step_number=1,
            title="Confirm the error type",
            action="Run: kubectl logs -l app=agent --since=5m | grep 'rate_limit\\|429'",
            expected_outcome="See rate-limit errors in logs",
            escalate_if="Errors are 500s, not 429 — escalate to Anthropic support",
        ),
        RunbookStep(
            step_number=2,
            title="Check current quota usage",
            action="Check Anthropic console dashboard for token/request usage",
            expected_outcome="Usage near or at tier limit",
        ),
    ],
    mitigation_steps=[
        RunbookStep(
            step_number=1,
            title="Enable fallback model",
            action="Set env var FALLBACK_MODEL=claude-haiku-5-5 and restart agent pods",
            expected_outcome="Agent resumes with cheaper/faster model; error rate drops",
        ),
        RunbookStep(
            step_number=2,
            title="Increase retry jitter",
            action="Update RETRY_MAX_JITTER_MS=5000 in configmap and hot-reload",
            expected_outcome="Retry storms dissipate",
        ),
    ],
    rollback_steps=[
        RunbookStep(
            step_number=1,
            title="Restore original model",
            action="Unset FALLBACK_MODEL env var; redeploy pods",
            expected_outcome="Agent returns to primary model",
            rollback_step=True,
        ),
    ],
    communication_template=(
        "**Incident:** LLM API Quota Exhaustion\n"
        "**Severity:** SEV2\n"
        "**Impact:** Agent tasks are failing; users see errors\n"
        "**Current action:** Switched to fallback model\n"
        "**ETA to resolution:** Under investigation"
    ),
    owner="platform-team",
    last_tested="2026-10-01",
)


if __name__ == "__main__":
    print(json.dumps(LLM_QUOTA_RUNBOOK.model_dump(), indent=2))
```

---

## Exercise 3

SLO monitor that tracks latency percentiles, success rate, and triggers alerts.

```python
"""
solutions/week-12/slo_monitor.py

SLO monitor — tracks real-time latency and success rate metrics,
triggers alert callbacks when thresholds are breached.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SLOConfig(BaseModel):
    window_seconds: int = 300           # 5-minute rolling window
    success_rate_slo: float = 0.995     # 99.5%
    p99_latency_slo_ms: float = 2000.0  # 2 s
    tool_success_rate_slo: float = 0.99 # 99%


@dataclass
class _Sample:
    timestamp: float
    duration_ms: float
    success: bool
    is_tool_call: bool


@dataclass
class SLOMonitor:
    config: SLOConfig = field(default_factory=SLOConfig)
    _samples: deque[_Sample] = field(default_factory=lambda: deque(maxlen=10_000))
    _alert_handlers: list[Callable[[str, float, float], None]] = field(default_factory=list)

    def record(
        self,
        duration_ms: float,
        success: bool,
        is_tool_call: bool = False,
    ) -> None:
        self._samples.append(_Sample(
            timestamp=time.time(),
            duration_ms=duration_ms,
            success=success,
            is_tool_call=is_tool_call,
        ))
        self._check_slos()

    def add_alert_handler(self, handler: Callable[[str, float, float], None]) -> None:
        """handler(slo_name, current_value, threshold)"""
        self._alert_handlers.append(handler)

    def _window_samples(self) -> list[_Sample]:
        cutoff = time.time() - self.config.window_seconds
        return [s for s in self._samples if s.timestamp >= cutoff]

    def _check_slos(self) -> None:
        samples = self._window_samples()
        if len(samples) < 10:
            return  # not enough data

        # Success rate
        success_rate = sum(1 for s in samples if s.success) / len(samples)
        if success_rate < self.config.success_rate_slo:
            self._fire_alert("success_rate_slo", success_rate, self.config.success_rate_slo)

        # p99 latency
        durations = sorted(s.duration_ms for s in samples)
        p99 = durations[int(len(durations) * 0.99)]
        if p99 > self.config.p99_latency_slo_ms:
            self._fire_alert("p99_latency_slo_ms", p99, self.config.p99_latency_slo_ms)

        # Tool success rate
        tool_samples = [s for s in samples if s.is_tool_call]
        if tool_samples:
            tool_success = sum(1 for s in tool_samples if s.success) / len(tool_samples)
            if tool_success < self.config.tool_success_rate_slo:
                self._fire_alert("tool_success_rate_slo", tool_success, self.config.tool_success_rate_slo)

    def _fire_alert(self, slo_name: str, current: float, threshold: float) -> None:
        logger.warning(
            "SLO BREACH: %s current=%.4f threshold=%.4f",
            slo_name, current, threshold,
        )
        for handler in self._alert_handlers:
            handler(slo_name, current, threshold)
```

Test suite:

```python
"""tests/test_slo_monitor.py"""
import pytest
from slo_monitor import SLOMonitor, SLOConfig


def test_alert_fires_on_high_failure_rate() -> None:
    alerts: list[tuple[str, float, float]] = []
    monitor = SLOMonitor(config=SLOConfig(success_rate_slo=0.9))
    monitor.add_alert_handler(lambda name, val, threshold: alerts.append((name, val, threshold)))

    for _ in range(100):
        monitor.record(100.0, success=False)
    monitor.record(100.0, success=True)

    assert any(a[0] == "success_rate_slo" for a in alerts)


def test_no_alert_when_slo_met() -> None:
    alerts: list[tuple] = []
    monitor = SLOMonitor(config=SLOConfig(success_rate_slo=0.9))
    monitor.add_alert_handler(lambda *a: alerts.append(a))

    for _ in range(100):
        monitor.record(100.0, success=True)

    slo_alerts = [a for a in alerts if a[0] == "success_rate_slo"]
    assert not slo_alerts
```
