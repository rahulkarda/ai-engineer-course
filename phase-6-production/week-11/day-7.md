# Phase 6 · Week 11 · Day 7 — Incident Response, Post-Mortems, and Dashboards

> Objective: Model the incident lifecycle with Pydantic, generate a blameless post-mortem document, and produce a Grafana dashboard JSON spec programmatically.  ·  Time: ~3 hr project + 1 hr assessment prep  ·  Prereqs: [[day-4]], [[day-5]], [[day-6]]

---

## Concept

### Incident response lifecycle

An incident is any unplanned event that degrades service quality or reliability. The response lifecycle has five phases:

```
Detect → Triage → Mitigate → Resolve → Review
```

| Phase | Goal | Output |
|---|---|---|
| **Detect** | Know something is wrong before users report it | Alert fires, on-call paged |
| **Triage** | Understand scope and severity; decide impact | Severity level assigned, incident channel opened |
| **Mitigate** | Stop the bleeding — reduce user impact fast | Rollback, traffic shift, feature flag disabled |
| **Resolve** | Fix the root cause | Root fix deployed, monitoring confirms recovery |
| **Review** | Learn from it; prevent recurrence | Post-mortem published, action items tracked |

**Time to detect** and **time to mitigate** are the two metrics that matter most. Resolving root cause can take days; stopping user impact must happen in minutes.

### Severity levels

| Sev | Criteria | Response |
|---|---|---|
| SEV-1 | Total outage, data loss risk | Immediate all-hands, C-level notification |
| SEV-2 | Major feature degraded, > 20% users affected | On-call + team lead, within 15 min |
| SEV-3 | Minor degradation, workaround exists | Next business day |
| SEV-4 | Cosmetic / logging issues | Backlog |

### Blameless post-mortem culture

Traditional incident reviews ask *who* made the mistake. Blameless post-mortems ask *what conditions made the mistake possible* and *how to change the system* so it cannot happen again.

Principles:
- Engineers acted reasonably with the information they had at the time.
- The system, not the individual, failed.
- Shame prevents honest disclosure — blameless reviews surface more facts.
- Action items go on the system: monitoring, tests, documentation, automated checks.

### 5 Whys technique

Repeatedly ask "Why did this happen?" until you reach a root cause that is addressable:

```
1. Why did users see errors?        → LLM gateway returned 503s
2. Why did it return 503s?          → Context window exceeded, no graceful handling
3. Why was context window exceeded? → Conversation history not trimmed
4. Why was history not trimmed?     → Trimming logic had an off-by-one bug
5. Why wasn't the bug caught?       → No integration test for maximum context length
Root cause: missing integration test → Action: add context-length integration tests
```

### Dashboards with Grafana

Grafana is the de facto standard dashboard platform for observability. Key concepts:

- **Data source**: where metrics live (Prometheus, Loki for logs, Tempo for traces, etc.)
- **Panel**: one visualization — graph, stat, table, gauge, heatmap.
- **Dashboard**: a collection of panels with a shared time range and variables.
- **Variables**: template variables (e.g., `$service`) let one dashboard cover multiple services.
- **Annotations**: vertical lines on panels marking events (e.g., deployments, incidents).

### RED dashboard pattern

The RED method (Weaveworks) defines three panels for any service:

| Letter | Metric | Prometheus query example |
|---|---|---|
| **R** Rate | Requests per second | `rate(http_requests_total[5m])` |
| **E** Errors | Error rate (% of requests) | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])` |
| **D** Duration | Latency distribution | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` |

For LLM services, add:
- Token throughput (tokens/sec)
- Cost per minute (USD)
- Context window saturation (%)

### Dashboard as code

Storing dashboards as JSON or Grafonnet (Jsonnet-based DSL) in version control means:
- Dashboards are reviewed, versioned, and testable.
- Reproduced across environments without clicking.
- Diffable — you see exactly what changed in a PR.

Grafana's HTTP API accepts JSON directly. The structure is:

```json
{
  "title": "LLM Gateway RED",
  "uid": "llm-gateway-red",
  "panels": [...],
  "time": {"from": "now-1h", "to": "now"},
  "refresh": "30s"
}
```

---

## How it works internally

1. **Post-mortem as data**: represent the post-mortem as a typed Pydantic model. This enables validation (required fields, severity enum), storage, and templated rendering.
2. **Markdown generation**: walk the model fields and emit structured Markdown following a standard template. Consistent templates make post-mortems scannable and comparable over time.
3. **Dashboard as data**: represent the Grafana dashboard structure as nested Pydantic models. `model_dump(mode="json")` produces the JSON spec that Grafana's API accepts.
4. **Programmatic panel generation**: define panels as typed objects, then serialise. This is the foundation of "dashboard-as-code" without external tools.

---

## Worked example

Install dependencies:

```bash
uv add pydantic
```

```python
# incident_tools.py
"""
Incident response tools:
- PostMortem Pydantic v2 model with markdown generation
- GrafanaDashboardSpec with Grafana JSON export
- Both are fully typed with Python 3.12+ syntax
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── 1. Enums ───────────────────────────────────────────────────────────────────

class Severity(IntEnum):
    SEV1 = 1  # Total outage
    SEV2 = 2  # Major degradation
    SEV3 = 3  # Minor degradation
    SEV4 = 4  # Cosmetic / non-functional


class IncidentStatus(StrEnum):
    OPEN = "open"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


# ── 2. Post-mortem Pydantic models ─────────────────────────────────────────────

class TimelineEntry(BaseModel):
    """One event in the incident timeline."""

    timestamp: datetime
    actor: str = Field(description="Person or system that performed the action")
    action: str = Field(description="What happened or was done")
    impact: str | None = Field(default=None, description="Effect of this action on the incident")


class ActionItem(BaseModel):
    """A follow-up task arising from the post-mortem."""

    title: str
    owner: str
    due_date: datetime
    priority: Literal["P1", "P2", "P3"] = "P2"
    status: Literal["open", "in_progress", "done"] = "open"
    jira_ticket: str | None = None


class ImpactEstimate(BaseModel):
    """Quantified impact of the incident."""

    affected_users: int = Field(ge=0)
    duration_minutes: int = Field(ge=0)
    error_budget_consumed_pct: float = Field(ge=0.0, le=100.0)
    estimated_revenue_impact_usd: float | None = None
    services_affected: list[str] = Field(default_factory=list)


class PostMortem(BaseModel):
    """
    Structured blameless post-mortem record.

    Fields map to the standard sections of a post-mortem document.
    Populate all fields; generate_markdown() produces the final document.
    """

    incident_id: str = Field(description="Unique identifier, e.g. INC-2024-001")
    title: str
    severity: Severity
    status: IncidentStatus = IncidentStatus.CLOSED
    detected_at: datetime
    mitigated_at: datetime | None = None
    resolved_at: datetime | None = None

    summary: str = Field(
        description="One-paragraph summary: what happened, what was the impact, how it was resolved"
    )
    impact: ImpactEstimate
    timeline: list[TimelineEntry] = Field(
        min_length=1,
        description="Chronological sequence of events from detection to resolution",
    )
    root_causes: list[str] = Field(
        min_length=1,
        description="Root causes identified via 5 Whys or similar analysis",
    )
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Conditions that made the incident worse or harder to detect",
    )
    action_items: list[ActionItem] = Field(
        min_length=1,
        description="Concrete follow-up tasks to prevent recurrence",
    )
    what_went_well: list[str] = Field(
        default_factory=list,
        description="Blameless recognition of effective responses",
    )
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Key insights for future incidents",
    )

    @field_validator("mitigated_at")
    @classmethod
    def mitigated_after_detected(cls, v: datetime | None, info: Any) -> datetime | None:
        if v is not None and "detected_at" in info.data:
            if v < info.data["detected_at"]:
                raise ValueError("mitigated_at must be after detected_at")
        return v

    @property
    def time_to_detect_minutes(self) -> float:
        """Minutes from incident start to detection (approximated as 0 here — detection = start)."""
        return 0.0

    @property
    def time_to_mitigate_minutes(self) -> float | None:
        if self.mitigated_at is None:
            return None
        return round((self.mitigated_at - self.detected_at).total_seconds() / 60, 1)

    @property
    def time_to_resolve_minutes(self) -> float | None:
        if self.resolved_at is None:
            return None
        return round((self.resolved_at - self.detected_at).total_seconds() / 60, 1)


def generate_post_mortem_markdown(pm: PostMortem) -> str:
    """
    Render a PostMortem model as a structured Markdown document.

    Args:
        pm: Populated PostMortem model.

    Returns:
        Markdown string ready to be written to a file or posted to a wiki.
    """
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}\n")

    def section(title: str, items: list[str]) -> None:
        h(2, title)
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    # Header
    h(1, f"Post-Mortem: {pm.title}")
    lines.append(f"**Incident ID**: {pm.incident_id}  ")
    lines.append(f"**Severity**: SEV-{pm.severity.value}  ")
    lines.append(f"**Status**: {pm.status}  ")
    lines.append(f"**Detected**: {pm.detected_at.strftime('%Y-%m-%d %H:%M UTC')}  ")
    if pm.mitigated_at:
        lines.append(f"**Mitigated**: {pm.mitigated_at.strftime('%Y-%m-%d %H:%M UTC')} ({pm.time_to_mitigate_minutes} min)  ")
    if pm.resolved_at:
        lines.append(f"**Resolved**: {pm.resolved_at.strftime('%Y-%m-%d %H:%M UTC')} ({pm.time_to_resolve_minutes} min)  ")
    lines.append("")

    # Summary
    h(2, "Summary")
    lines.append(pm.summary)
    lines.append("")

    # Impact
    h(2, "Impact")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Affected users | {pm.impact.affected_users:,} |")
    lines.append(f"| Duration | {pm.impact.duration_minutes} minutes |")
    lines.append(f"| Error budget consumed | {pm.impact.error_budget_consumed_pct:.1f}% |")
    if pm.impact.estimated_revenue_impact_usd is not None:
        lines.append(f"| Revenue impact | ${pm.impact.estimated_revenue_impact_usd:,.2f} |")
    if pm.impact.services_affected:
        lines.append(f"| Services affected | {', '.join(pm.impact.services_affected)} |")
    lines.append("")

    # Timeline
    h(2, "Timeline")
    lines.append("| Time (UTC) | Actor | Action | Impact |")
    lines.append("|---|---|---|---|")
    for entry in sorted(pm.timeline, key=lambda e: e.timestamp):
        impact_cell = entry.impact or "—"
        lines.append(
            f"| {entry.timestamp.strftime('%H:%M:%S')} "
            f"| {entry.actor} "
            f"| {entry.action} "
            f"| {impact_cell} |"
        )
    lines.append("")

    # Root causes
    section("Root Causes", pm.root_causes)

    # Contributing factors
    if pm.contributing_factors:
        section("Contributing Factors", pm.contributing_factors)

    # Action items
    h(2, "Action Items")
    lines.append("| # | Title | Owner | Due | Priority | Status |")
    lines.append("|---|---|---|---|---|---|")
    for i, item in enumerate(pm.action_items, 1):
        ticket = f" ({item.jira_ticket})" if item.jira_ticket else ""
        lines.append(
            f"| {i} | {item.title}{ticket} | {item.owner} "
            f"| {item.due_date.strftime('%Y-%m-%d')} "
            f"| {item.priority} | {item.status} |"
        )
    lines.append("")

    # What went well
    if pm.what_went_well:
        section("What Went Well", pm.what_went_well)

    # Lessons learned
    if pm.lessons_learned:
        section("Lessons Learned", pm.lessons_learned)

    logger.info("Generated post-mortem markdown: %d lines", len(lines))
    return "\n".join(lines)


# ── 3. Grafana dashboard models ────────────────────────────────────────────────

class GrafanaPanelTarget(BaseModel):
    """One PromQL query that powers a Grafana panel."""

    expr: str = Field(description="PromQL query expression")
    legend_format: str = Field(default="{{service}}", alias="legendFormat")
    ref_id: str = Field(default="A", alias="refId")

    model_config = {"populate_by_name": True}


class GrafanaPanel(BaseModel):
    """One panel (visualisation) in a Grafana dashboard."""

    id: int
    title: str
    type: Literal["timeseries", "stat", "gauge", "bargauge", "table", "heatmap"] = "timeseries"
    targets: list[GrafanaPanelTarget]
    # Grid position
    grid_pos_x: int = Field(default=0, alias="gridPosX")
    grid_pos_y: int = Field(default=0, alias="gridPosY")
    grid_pos_w: int = Field(default=12, alias="gridPosW")
    grid_pos_h: int = Field(default=8, alias="gridPosH")
    # Optional unit
    unit: str | None = None
    description: str = ""

    model_config = {"populate_by_name": True}

    def to_grafana_dict(self) -> dict[str, Any]:
        """Serialise to the dict structure Grafana's API expects."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "description": self.description,
            "targets": [
                {
                    "expr": t.expr,
                    "legendFormat": t.legend_format,
                    "refId": t.ref_id,
                }
                for t in self.targets
            ],
            "gridPos": {
                "x": self.grid_pos_x,
                "y": self.grid_pos_y,
                "w": self.grid_pos_w,
                "h": self.grid_pos_h,
            },
            **({"fieldConfig": {"defaults": {"unit": self.unit}}} if self.unit else {}),
        }


class GrafanaDashboardSpec(BaseModel):
    """
    Grafana dashboard specification.

    Call to_grafana_json() and POST to Grafana's HTTP API:
        POST /api/dashboards/db
        {"dashboard": <this json>, "overwrite": true}
    """

    title: str
    uid: str = Field(description="Unique identifier — used in URLs and API lookups")
    datasource: str = Field(default="prometheus", description="Data source name in Grafana")
    panels: list[GrafanaPanel]
    time_from: str = Field(default="now-1h", alias="timeFrom")
    time_to: str = Field(default="now", alias="timeTo")
    refresh: str = Field(default="30s", description="Auto-refresh interval")
    tags: list[str] = Field(default_factory=list)
    description: str = ""

    model_config = {"populate_by_name": True}

    def to_grafana_json(self) -> str:
        """
        Serialise to the JSON string accepted by Grafana's dashboard API.

        Returns:
            Pretty-printed JSON string.
        """
        dashboard_dict: dict[str, Any] = {
            "title": self.title,
            "uid": self.uid,
            "description": self.description,
            "tags": self.tags,
            "time": {"from": self.time_from, "to": self.time_to},
            "refresh": self.refresh,
            "panels": [p.to_grafana_dict() for p in self.panels],
            "schemaVersion": 39,  # Grafana schema version
        }
        logger.info("Serialised dashboard '%s' with %d panels", self.title, len(self.panels))
        return json.dumps(dashboard_dict, indent=2)


def build_red_dashboard(service: str, datasource: str = "prometheus") -> GrafanaDashboardSpec:
    """
    Build a standard RED (Rate / Error rate / Duration) dashboard for a service.

    Args:
        service: Service name used in Prometheus label selectors.
        datasource: Grafana data source name.

    Returns:
        GrafanaDashboardSpec ready for serialisation and upload.
    """
    panels = [
        # Panel 1: Request rate (R)
        GrafanaPanel(
            id=1,
            title="Request Rate (req/s)",
            type="timeseries",
            description="Number of requests per second. Spike = traffic surge.",
            targets=[
                GrafanaPanelTarget(
                    expr=f'sum(rate(http_requests_total{{service="{service}"}}[5m]))',
                    legend_format="req/s",
                    ref_id="A",
                )
            ],
            grid_pos_x=0, grid_pos_y=0, grid_pos_w=8, grid_pos_h=8,
            unit="reqps",
        ),
        # Panel 2: Error rate (E)
        GrafanaPanel(
            id=2,
            title="Error Rate (%)",
            type="timeseries",
            description="Percentage of requests that returned 5xx. SLO: < 0.1%.",
            targets=[
                GrafanaPanelTarget(
                    expr=(
                        f'100 * sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[5m])) '
                        f'/ sum(rate(http_requests_total{{service="{service}"}}[5m]))'
                    ),
                    legend_format="error %",
                    ref_id="A",
                )
            ],
            grid_pos_x=8, grid_pos_y=0, grid_pos_w=8, grid_pos_h=8,
            unit="percent",
        ),
        # Panel 3: P95 latency (D)
        GrafanaPanel(
            id=3,
            title="P95 Latency (s)",
            type="timeseries",
            description="95th percentile response time. SLO: P95 < 5 s.",
            targets=[
                GrafanaPanelTarget(
                    expr=(
                        f'histogram_quantile(0.95, '
                        f'sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
                    ),
                    legend_format="p95",
                    ref_id="A",
                )
            ],
            grid_pos_x=16, grid_pos_y=0, grid_pos_w=8, grid_pos_h=8,
            unit="s",
        ),
        # Panel 4: LLM token throughput (LLM-specific extension)
        GrafanaPanel(
            id=4,
            title="Token Throughput (tokens/s)",
            type="timeseries",
            description="Combined input+output tokens per second.",
            targets=[
                GrafanaPanelTarget(
                    expr=f'sum(rate(llm_tokens_total{{service="{service}"}}[5m]))',
                    legend_format="tokens/s",
                    ref_id="A",
                )
            ],
            grid_pos_x=0, grid_pos_y=8, grid_pos_w=12, grid_pos_h=8,
        ),
        # Panel 5: Cumulative LLM cost
        GrafanaPanel(
            id=5,
            title="LLM Cost (USD/min)",
            type="stat",
            description="Rate of USD spend on LLM calls. Watch for runaway loops.",
            targets=[
                GrafanaPanelTarget(
                    expr=f'sum(rate(llm_cost_usd_total{{service="{service}"}}[1m])) * 60',
                    legend_format="$/min",
                    ref_id="A",
                )
            ],
            grid_pos_x=12, grid_pos_y=8, grid_pos_w=12, grid_pos_h=8,
            unit="currencyUSD",
        ),
    ]

    return GrafanaDashboardSpec(
        title=f"{service} — RED + LLM Observability",
        uid=f"{service}-red",
        datasource=datasource,
        panels=panels,
        tags=["llm", "red", "slo", service],
        description=f"RED dashboard with LLM cost and token throughput for service: {service}",
    )


# ── 4. Example usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # === Post-mortem demo ===
    pm = PostMortem(
        incident_id="INC-2024-042",
        title="LLM Gateway Context Window Overflow — 23% Error Rate",
        severity=Severity.SEV2,
        status=IncidentStatus.CLOSED,
        detected_at=datetime(2024, 11, 8, 14, 35, 0, tzinfo=UTC),
        mitigated_at=datetime(2024, 11, 8, 14, 58, 0, tzinfo=UTC),
        resolved_at=datetime(2024, 11, 8, 16, 20, 0, tzinfo=UTC),
        summary=(
            "On 2024-11-08 at 14:35 UTC, the LLM gateway began returning 503 errors for 23% of "
            "requests due to context window overflow. Long-running chat sessions accumulated "
            "history beyond 200k tokens. The context trimming service had an off-by-one bug "
            "introduced in the 14:00 UTC deployment. Mitigation was achieved at 14:58 by rolling "
            "back the deployment. Root cause fix (corrected trimming logic + integration tests) "
            "was deployed at 16:20."
        ),
        impact=ImpactEstimate(
            affected_users=3_400,
            duration_minutes=23,
            error_budget_consumed_pct=18.7,
            estimated_revenue_impact_usd=1_200.0,
            services_affected=["llm-gateway", "chat-api"],
        ),
        timeline=[
            TimelineEntry(
                timestamp=datetime(2024, 11, 8, 14, 0, 0, tzinfo=UTC),
                actor="deploy-bot",
                action="Deployed llm-gateway v2.4.1 to production",
                impact="Context trimming logic changed",
            ),
            TimelineEntry(
                timestamp=datetime(2024, 11, 8, 14, 35, 0, tzinfo=UTC),
                actor="prometheus-alertmanager",
                action="LLM_Gateway_Availability_Critical alert fired",
                impact="On-call paged",
            ),
            TimelineEntry(
                timestamp=datetime(2024, 11, 8, 14, 40, 0, tzinfo=UTC),
                actor="alice@eng",
                action="Acknowledged alert, started investigation",
            ),
            TimelineEntry(
                timestamp=datetime(2024, 11, 8, 14, 52, 0, tzinfo=UTC),
                actor="alice@eng",
                action="Identified off-by-one in context trim function via trace comparison",
                impact="Root cause identified",
            ),
            TimelineEntry(
                timestamp=datetime(2024, 11, 8, 14, 58, 0, tzinfo=UTC),
                actor="alice@eng",
                action="Rolled back to llm-gateway v2.4.0",
                impact="Error rate dropped to 0%, incident mitigated",
            ),
            TimelineEntry(
                timestamp=datetime(2024, 11, 8, 16, 20, 0, tzinfo=UTC),
                actor="bob@eng",
                action="Deployed v2.4.2 with fix and new integration tests",
                impact="Incident resolved",
            ),
        ],
        root_causes=[
            "Off-by-one error in context window trim function introduced in v2.4.1",
            "No integration test for maximum context length edge case",
        ],
        contributing_factors=[
            "Staging environment uses 8k token context limit; bug only manifests near 200k",
            "No automated canary analysis comparing error rates between v2.4.0 and v2.4.1",
        ],
        action_items=[
            ActionItem(
                title="Add integration test: context window at 95%, 99%, 100% capacity",
                owner="bob@eng",
                due_date=datetime(2024, 11, 15, tzinfo=UTC),
                priority="P1",
                jira_ticket="ENG-4421",
            ),
            ActionItem(
                title="Configure canary deployment to auto-rollback on > 1% error rate",
                owner="platform@eng",
                due_date=datetime(2024, 11, 22, tzinfo=UTC),
                priority="P2",
                jira_ticket="ENG-4422",
            ),
            ActionItem(
                title="Update staging to use same context window size as production",
                owner="alice@eng",
                due_date=datetime(2024, 11, 29, tzinfo=UTC),
                priority="P2",
                jira_ticket="ENG-4423",
            ),
        ],
        what_went_well=[
            "Alert fired within 5 minutes of the error rate exceeding the SLO threshold",
            "On-call identified root cause within 12 minutes of acknowledgement",
            "Rollback completed without database migration risk",
        ],
        lessons_learned=[
            "Staging context window must match production — configuration drift hides bugs",
            "LLM-specific integration tests (context limits, prompt length edge cases) need a dedicated test suite",
        ],
    )

    md = generate_post_mortem_markdown(pm)
    logger.info("Post-mortem TTM: %s minutes", pm.time_to_mitigate_minutes)
    logger.info("Post-mortem generated (%d chars)", len(md))
    print(md[:500] + "\n... [truncated for demo]")

    # === Dashboard demo ===
    dashboard = build_red_dashboard("llm-gateway")
    dashboard_json = dashboard.to_grafana_json()
    logger.info("Dashboard JSON: %d bytes, %d panels", len(dashboard_json), len(dashboard.panels))

    # Write to file (in production: POST to Grafana API)
    # with open("dashboard.json", "w") as f:
    #     f.write(dashboard_json)
    print(dashboard_json[:500] + "\n... [truncated for demo]")
```

### Test file

```python
# tests/unit/test_incident_tools.py
"""Unit tests for post-mortem and dashboard generation. No LLM calls, no network."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from incident_tools import (
    ActionItem,
    GrafanaDashboardSpec,
    GrafanaPanel,
    GrafanaPanelTarget,
    ImpactEstimate,
    PostMortem,
    Severity,
    TimelineEntry,
    build_red_dashboard,
    generate_post_mortem_markdown,
)


def _minimal_pm(**overrides: object) -> PostMortem:
    """Return a minimal valid PostMortem for test use."""
    defaults: dict[str, object] = {
        "incident_id": "INC-TEST-001",
        "title": "Test incident",
        "severity": Severity.SEV3,
        "detected_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "mitigated_at": datetime(2024, 1, 1, 12, 30, tzinfo=UTC),
        "resolved_at": datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
        "summary": "Test summary.",
        "impact": ImpactEstimate(
            affected_users=10,
            duration_minutes=60,
            error_budget_consumed_pct=1.0,
        ),
        "timeline": [
            TimelineEntry(
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                actor="system",
                action="Alert fired",
            )
        ],
        "root_causes": ["Bug in module X"],
        "action_items": [
            ActionItem(
                title="Fix bug",
                owner="eng@example.com",
                due_date=datetime(2024, 1, 8, tzinfo=UTC),
            )
        ],
    }
    defaults.update(overrides)
    return PostMortem(**defaults)  # type: ignore[arg-type]


class TestPostMortem:
    def test_time_to_mitigate(self) -> None:
        pm = _minimal_pm()
        assert pm.time_to_mitigate_minutes == 30.0

    def test_time_to_resolve(self) -> None:
        pm = _minimal_pm()
        assert pm.time_to_resolve_minutes == 60.0

    def test_mitigated_before_detected_rejected(self) -> None:
        with pytest.raises(ValueError, match="after detected_at"):
            _minimal_pm(
                mitigated_at=datetime(2024, 1, 1, 11, 0, tzinfo=UTC)  # before detected_at
            )

    def test_empty_root_causes_rejected(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            _minimal_pm(root_causes=[])

    def test_empty_action_items_rejected(self) -> None:
        with pytest.raises(Exception):
            _minimal_pm(action_items=[])


class TestGenerateMarkdown:
    def test_contains_incident_id(self) -> None:
        pm = _minimal_pm()
        md = generate_post_mortem_markdown(pm)
        assert "INC-TEST-001" in md

    def test_contains_all_section_headers(self) -> None:
        pm = _minimal_pm()
        md = generate_post_mortem_markdown(pm)
        for section in ["Summary", "Impact", "Timeline", "Root Causes", "Action Items"]:
            assert section in md

    def test_timeline_entries_sorted_by_time(self) -> None:
        now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        pm = _minimal_pm(
            timeline=[
                TimelineEntry(timestamp=now + timedelta(minutes=5), actor="B", action="B happened"),
                TimelineEntry(timestamp=now, actor="A", action="A happened"),
            ]
        )
        md = generate_post_mortem_markdown(pm)
        idx_a = md.index("A happened")
        idx_b = md.index("B happened")
        assert idx_a < idx_b, "Earlier event should appear first in markdown"

    def test_impact_table_included(self) -> None:
        pm = _minimal_pm()
        md = generate_post_mortem_markdown(pm)
        assert "Affected users" in md
        assert "Error budget consumed" in md

    def test_revenue_impact_shown_when_set(self) -> None:
        pm = _minimal_pm(
            impact=ImpactEstimate(
                affected_users=100,
                duration_minutes=30,
                error_budget_consumed_pct=2.0,
                estimated_revenue_impact_usd=500.0,
            )
        )
        md = generate_post_mortem_markdown(pm)
        assert "500" in md


class TestGrafanaDashboard:
    def test_red_dashboard_has_five_panels(self) -> None:
        dash = build_red_dashboard("llm-gateway")
        assert len(dash.panels) == 5

    def test_to_grafana_json_is_valid_json(self) -> None:
        dash = build_red_dashboard("my-service")
        json_str = dash.to_grafana_json()
        parsed = json.loads(json_str)
        assert parsed["title"] == "my-service — RED + LLM Observability"
        assert parsed["uid"] == "my-service-red"
        assert len(parsed["panels"]) == 5

    def test_panel_grid_positions_set(self) -> None:
        dash = build_red_dashboard("svc")
        json_str = dash.to_grafana_json()
        parsed = json.loads(json_str)
        for panel in parsed["panels"]:
            assert "gridPos" in panel
            assert all(k in panel["gridPos"] for k in ("x", "y", "w", "h"))

    def test_service_name_in_promql(self) -> None:
        dash = build_red_dashboard("my-unique-service")
        json_str = dash.to_grafana_json()
        assert "my-unique-service" in json_str

    def test_custom_datasource_in_spec(self) -> None:
        dash = build_red_dashboard("svc", datasource="thanos")
        assert dash.datasource == "thanos"
```

---

## Exercises

1. Add a `IncidentStatus` state machine to `PostMortem` using a `@model_validator` that enforces: `resolved_at` cannot be set if `status != RESOLVED`, and `mitigated_at` is required before `resolved_at`. Write tests for all invalid transitions.

2. Extend `build_red_dashboard` with an optional `slo_target: float` parameter. When provided, add a threshold line to the error rate panel by including Grafana's `thresholds` config in the panel dict. Write a test that verifies the threshold value appears in the JSON when `slo_target=0.001` is passed.

3. Write a function `diff_dashboards(old: GrafanaDashboardSpec, new: GrafanaDashboardSpec) -> dict[str, list[str]]` that returns `{"added": [...], "removed": [...], "modified": [...]}` panel titles. Write tests for each category.

4. (Assessment prep) Write a `RunbookEntry` Pydantic model with fields `alert_name`, `severity`, `description`, `diagnostic_steps: list[str]`, and `escalation_path`. Write a `generate_runbook_markdown(entries: list[RunbookEntry]) -> str` function and at least three unit tests.

(Solutions: ../solutions/week-11/day-7.md)

---

## Self-check

- [ ] I can list the five phases of the incident response lifecycle and state the goal of each phase.
- [ ] I can explain what makes a post-mortem "blameless" and why blame makes post-mortems less effective.
- [ ] I can describe the RED dashboard pattern (Rate, Error rate, Duration) and add two LLM-specific panels to it.
- [ ] I can serialise a Pydantic model to Grafana dashboard JSON and explain why "dashboard-as-code" is preferable to clicking in the UI.

---

## Further reading

- [Google SRE Book — Chapter 15: Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)
- [Grafana dashboard JSON model reference](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/)
- [Grafonnet (Jsonnet DSL for Grafana)](https://grafana.github.io/grafonnet/API/dashboard/index.html)
- [Grafana HTTP API — Create/Update dashboard](https://grafana.com/docs/grafana/latest/developers/http_api/dashboard/)
- [The RED Method — Tom Wilkie](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)
