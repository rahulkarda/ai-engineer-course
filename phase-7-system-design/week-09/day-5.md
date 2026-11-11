# Phase 7 · Week 09 · Day 5 — Incident Response Under Pressure & Root-Cause Analysis

> Objective: Execute a structured incident response workflow and produce a rigorous root-cause analysis that prevents recurrence.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Incident response is a *process*, not just debugging. The process exists because under pressure humans make systematic errors: they skip verification steps, tunnel on the first hypothesis, and take irreversible actions without a rollback plan.

### The incident lifecycle

```
DETECT → TRIAGE → MITIGATE → INVESTIGATE → RESOLVE → POST-MORTEM
```

| Phase | Goal | Key action |
|-------|------|-----------|
| Detect | Know it happened | Alerting thresholds, SLO burn |
| Triage | Assess severity | Customer impact? Blast radius? |
| Mitigate | Stop the bleeding | Feature flag off, rollback, traffic shift |
| Investigate | Find root cause | Logs, traces, metrics — the five-step method |
| Resolve | Confirm fix in production | Monitor error rate back to baseline |
| Post-mortem | Prevent recurrence | 5-Whys → action items |

### Severity tiers

| Sev | Description | Response SLA |
|-----|-------------|-------------|
| SEV-1 | Service down / data loss | All hands, 5 min page |
| SEV-2 | Major feature broken, degraded perf | On-call + lead, 30 min |
| SEV-3 | Minor feature, workaround exists | Next business day |
| SEV-4 | Cosmetic / logging gap | Backlog |

### Root-cause analysis: 5-Whys

The 5-Whys technique drives from symptom to systemic cause by asking "why" recursively:

```
Symptom:  API returning 504s
Why 1:    Database queries timing out
Why 2:    Query plan chose full table scan
Why 3:    Statistics were stale (last ANALYZE: 6 months ago)
Why 4:    ANALYZE is not in autovacuum configuration
Why 5:    No DBA review when table grew from 10K to 10M rows
Root:     No capacity/schema review process for tables crossing growth thresholds
```

The root cause is always *a process or system gap*, not a person.

## How it works internally

### Mitigation vs. fix

Mitigation stops customer impact; it does not fix the root cause. Common mitigations:
- **Rollback**: revert to last known good deploy (`git revert` + deploy)
- **Feature flag**: disable the offending code path without a deploy
- **Traffic shift**: route to a healthy replica, region, or cached response
- **Rate limiting**: shed load to prevent total collapse

The mitigation buys time for a proper investigation. Never skip mitigation to investigate first — customers are impacted.

### Timeline reconstruction

During post-mortem you reconstruct the exact timeline from:
1. Alert fire timestamps (monitoring system)
2. Deployment logs (CI/CD system)
3. Correlated log/trace queries (e.g., Loki/Grafana, Datadog)
4. Communication timestamps (Slack, PagerDuty)

The timeline often reveals *precursor events* — warnings logged hours before the alert fired.

## Worked example

```python
"""
day5_incident_response.py — incident lifecycle simulation + 5-Whys RCA

No external dependencies. Run: python day5_incident_response.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("incident")


# ─────────────────────────────────────────────────────────────────────────────
# Incident data model
# ─────────────────────────────────────────────────────────────────────────────

class Severity(Enum):
    SEV1 = 1
    SEV2 = 2
    SEV3 = 3
    SEV4 = 4


class IncidentStatus(Enum):
    DETECTING = auto()
    TRIAGING = auto()
    MITIGATING = auto()
    INVESTIGATING = auto()
    RESOLVED = auto()
    POST_MORTEM = auto()


@dataclass
class TimelineEvent:
    ts: datetime
    actor: str
    action: str
    details: str = ""


@dataclass
class Incident:
    id: str
    title: str
    severity: Severity
    started_at: datetime
    status: IncidentStatus = IncidentStatus.DETECTING
    timeline: list[TimelineEvent] = field(default_factory=list)
    mitigated_at: datetime | None = None
    resolved_at: datetime | None = None
    root_cause: str = ""
    action_items: list[str] = field(default_factory=list)

    def add_event(self, actor: str, action: str, details: str = "") -> None:
        event = TimelineEvent(
            ts=datetime.now(timezone.utc),
            actor=actor,
            action=action,
            details=details,
        )
        self.timeline.append(event)
        log.info("[%s] %s: %s %s", self.id, actor, action, f"— {details}" if details else "")

    def time_to_mitigate(self) -> timedelta | None:
        if self.mitigated_at:
            return self.mitigated_at - self.started_at
        return None

    def time_to_resolve(self) -> timedelta | None:
        if self.resolved_at:
            return self.resolved_at - self.started_at
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.name,
            "status": self.status.name,
            "root_cause": self.root_cause,
            "ttm_seconds": self.time_to_mitigate().total_seconds() if self.time_to_mitigate() else None,
            "ttr_seconds": self.time_to_resolve().total_seconds() if self.time_to_resolve() else None,
            "action_items": self.action_items,
            "timeline_events": len(self.timeline),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5-Whys engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhyChain:
    symptom: str
    whys: list[str] = field(default_factory=list)

    def add_why(self, why: str) -> "WhyChain":
        self.whys.append(why)
        return self

    def root_cause(self) -> str:
        return self.whys[-1] if self.whys else self.symptom

    def display(self) -> None:
        print(f"\n5-Whys Analysis")
        print(f"Symptom: {self.symptom}")
        for i, why in enumerate(self.whys, 1):
            indent = "  " * i
            print(f"{indent}Why {i}: {why}")
        print(f"\nRoot cause: {self.root_cause()}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulate a full incident lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def simulate_incident() -> Incident:
    """Simulate the INC-001 database timeout incident."""

    now = datetime.now(timezone.utc)
    incident = Incident(
        id="INC-001",
        title="API 504 errors — database query timeout",
        severity=Severity.SEV2,
        started_at=now - timedelta(minutes=45),  # started 45 min ago
    )

    # ── DETECT ──
    incident.status = IncidentStatus.DETECTING
    incident.add_event(
        "PagerDuty", "alert fired",
        "error_rate > 5% for 5 min on /api/orders",
    )

    # ── TRIAGE ──
    incident.status = IncidentStatus.TRIAGING
    incident.add_event(
        "on-call-engineer", "triage started",
        "checking dashboards — 504s on /api/orders, ~15% of requests",
    )
    incident.add_event(
        "on-call-engineer", "severity confirmed",
        "SEV-2: partial outage, orders feature unavailable, checkout unaffected",
    )

    # ── MITIGATE ──
    incident.status = IncidentStatus.MITIGATING
    incident.add_event(
        "on-call-engineer", "mitigation: feature flag off",
        "disabled 'new-order-history-v2' flag — routes to cached stub",
    )
    incident.mitigated_at = datetime.now(timezone.utc)
    incident.add_event(
        "on-call-engineer", "error rate confirmed down",
        "error_rate dropped from 15% to 0.1% — mitigation effective",
    )

    # ── INVESTIGATE ──
    incident.status = IncidentStatus.INVESTIGATING
    incident.add_event(
        "on-call-engineer", "log correlation",
        "trace_id=abc123: db span = 28 000 ms, rows_examined=9 800 000",
    )
    incident.add_event(
        "on-call-engineer", "EXPLAIN ANALYZE run",
        "Query 'SELECT * FROM orders WHERE customer_id=? ORDER BY created_at' — full table scan",
    )
    incident.add_event(
        "DBA", "root cause found",
        "Index on (customer_id, created_at) was dropped in migration 20241101_cleanup",
    )

    # ── RESOLVE ──
    incident.status = IncidentStatus.RESOLVED
    incident.add_event(
        "DBA", "fix deployed",
        "CREATE INDEX CONCURRENTLY idx_orders_customer_date ON orders(customer_id, created_at DESC)",
    )
    incident.add_event(
        "on-call-engineer", "feature flag re-enabled",
        "error_rate: 0.08%, p99 latency: 42 ms — incident resolved",
    )
    incident.resolved_at = datetime.now(timezone.utc)
    incident.root_cause = (
        "Index (customer_id, created_at) dropped by migration 20241101_cleanup "
        "without detecting downstream query dependency"
    )

    # ── ACTION ITEMS ──
    incident.status = IncidentStatus.POST_MORTEM
    incident.action_items = [
        "Add index-dependency check to migration linter (blocks merge if index dropped without query audit)",
        "Add EXPLAIN ANALYZE to CI query benchmarks for critical paths",
        "Set PagerDuty p99 latency alert at 2 s (currently no latency alert)",
        "Schedule quarterly DBA review of autovacuum/ANALYZE settings",
    ]

    return incident


def run_five_whys() -> None:
    chain = WhyChain("API returning 504 errors on /api/orders")
    chain.add_why("Database queries timing out (28 s vs 50 ms baseline)")
    chain.add_why("Query planner chose sequential scan over index scan")
    chain.add_why("Index on (customer_id, created_at) does not exist")
    chain.add_why("Index was dropped in migration 20241101_cleanup")
    chain.add_why("Migration linter has no rule preventing index drops without dependency check")
    chain.display()


if __name__ == "__main__":
    incident = simulate_incident()
    print("\n--- Incident Summary ---")
    print(json.dumps(incident.summary(), indent=2))
    run_five_whys()
```

## Exercises

1. Add a `time_to_detect(self) -> timedelta` method to `Incident` that measures from `started_at` to the first "alert fired" timeline event. Note: you need to store `started_at` as the actual fault start, not the alert time.

2. Write a `PostMortem` dataclass with fields: `incident_id`, `timeline`, `root_cause`, `contributing_factors: list[str]`, `action_items: list[ActionItem]` where `ActionItem` has `description`, `owner`, `due_date`, `status`. Add a `to_markdown() -> str` method. Write a test that renders a post-mortem and checks it contains all action item descriptions.

3. Implement a `SLOBurnRateAlert` that takes `error_budget_pct` (e.g., 0.1 for 10%) and `window_minutes`. Given a list of `(timestamp, is_error)` tuples, return True if the burn rate exceeds `2×` the expected steady-state rate. Test with: all-ok traffic (False), 50% errors for 10 minutes (True), 5% errors (False).

(Solutions: ../solutions/week-09/day-5.md)

## Self-check

- [ ] I can recite the six incident lifecycle phases and their goals
- [ ] I know the difference between mitigation and fix
- [ ] I can conduct a 5-Whys analysis that reaches a systemic root cause (not a person)
- [ ] I can implement an `Incident` data model with timeline tracking
- [ ] I can explain what TTM (time-to-mitigate) and TTR (time-to-resolve) measure

## Supplementary links

- [Google SRE — Incident Management](https://sre.google/sre-book/managing-incidents/)
- [PagerDuty Incident Response Guide](https://response.pagerduty.com/)
- [Atlassian post-mortem template](https://www.atlassian.com/incident-management/postmortem/templates)
