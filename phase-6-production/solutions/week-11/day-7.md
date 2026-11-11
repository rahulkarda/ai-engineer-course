# Week 11 · Day 7 Solutions

## Incident Response, Post-Mortems, and Dashboards

---

## Exercise 1

**Add `timeline_to_markdown()` to `PostMortem` — formats timeline as a sorted bullet list with timestamps.**

```python
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class TimelineEntry(BaseModel):
    """A single event in an incident timeline."""

    timestamp: datetime
    description: str
    author: str | None = None

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Timeline entry description must not be empty.")
        return v.strip()


class PostMortem(BaseModel):
    """Structured post-mortem document for a production incident."""

    incident_id: str
    title: str
    date: datetime
    severity: str                           # SEV1 / SEV2 / SEV3 / SEV4
    duration_minutes: int
    affected_users: int
    data_loss: bool
    root_cause: str
    timeline: list[TimelineEntry] = []
    action_items: list[str] = []
    lessons_learned: list[str] = []         # Exercise 3

    def timeline_to_markdown(self) -> str:
        """Format the incident timeline as a sorted Markdown bullet list.

        Returns:
            Multiline string with one bullet per timeline entry, sorted by timestamp.
            Returns a placeholder line if the timeline is empty.
        """
        if not self.timeline:
            return "- *(No timeline entries recorded.)*"

        sorted_entries = sorted(self.timeline, key=lambda e: e.timestamp)
        lines: list[str] = []
        for entry in sorted_entries:
            ts_str = entry.timestamp.strftime("%Y-%m-%d %H:%M UTC")
            author_suffix = f" *(by {entry.author})*" if entry.author else ""
            lines.append(f"- **{ts_str}** — {entry.description}{author_suffix}")

        logger.debug(
            "Rendered %d timeline entries for incident %s",
            len(lines),
            self.incident_id,
        )
        return "\n".join(lines)
```

**Example:**

```python
from datetime import datetime, timezone

pm = PostMortem(
    incident_id="INC-2025-042",
    title="API Gateway timeout storm",
    date=datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc),
    severity="SEV2",
    duration_minutes=47,
    affected_users=3200,
    data_loss=False,
    root_cause="Misconfigured upstream timeout propagation after deploy.",
    timeline=[
        TimelineEntry(
            timestamp=datetime(2025, 6, 15, 14, 3, tzinfo=timezone.utc),
            description="Alerts fired: p99 latency > 5s",
            author="pagerduty-bot",
        ),
        TimelineEntry(
            timestamp=datetime(2025, 6, 15, 13, 58, tzinfo=timezone.utc),
            description="Deploy v2.14.1 completed",
            author="ci-cd-bot",
        ),
        TimelineEntry(
            timestamp=datetime(2025, 6, 15, 14, 47, tzinfo=timezone.utc),
            description="Rolled back to v2.13.9, latency normalised",
            author="alice",
        ),
    ],
)

print(pm.timeline_to_markdown())
# - **2025-06-15 13:58 UTC** — Deploy v2.14.1 completed *(by ci-cd-bot)*
# - **2025-06-15 14:03 UTC** — Alerts fired: p99 latency > 5s *(by pagerduty-bot)*
# - **2025-06-15 14:47 UTC** — Rolled back to v2.13.9, latency normalised *(by alice)*
```

---

## Exercise 2

**Severity scoring function — returns SEV1/SEV2/SEV3/SEV4 given duration, affected users, and data loss flag.**

```python
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    SEV1 = "SEV1"   # Critical — immediate all-hands
    SEV2 = "SEV2"   # Major — on-call team engaged
    SEV3 = "SEV3"   # Moderate — next-business-day fix
    SEV4 = "SEV4"   # Minor — backlog ticket


def score_severity(
    duration_minutes: int,
    affected_users: int,
    data_loss: bool,
) -> SeverityLevel:
    """Compute a severity level from incident impact metrics.

    Scoring rules (applied in descending priority):

    SEV1 — any of:
        * data_loss is True
        * affected_users >= 10 000
        * duration_minutes >= 120

    SEV2 — any of:
        * affected_users >= 1 000
        * duration_minutes >= 30

    SEV3 — any of:
        * affected_users >= 100
        * duration_minutes >= 10

    SEV4 — everything else (low impact, brief duration, no data loss).

    Args:
        duration_minutes: Total incident duration in minutes.
        affected_users: Number of unique affected users.
        data_loss: True if any customer data was lost or corrupted.

    Returns:
        SeverityLevel enum value.
    """
    if data_loss or affected_users >= 10_000 or duration_minutes >= 120:
        sev = SeverityLevel.SEV1
    elif affected_users >= 1_000 or duration_minutes >= 30:
        sev = SeverityLevel.SEV2
    elif affected_users >= 100 or duration_minutes >= 10:
        sev = SeverityLevel.SEV3
    else:
        sev = SeverityLevel.SEV4

    logger.info(
        "Severity scored as %s (duration=%dm, users=%d, data_loss=%s)",
        sev.value,
        duration_minutes,
        affected_users,
        data_loss,
    )
    return sev
```

**Scoring matrix summary:**

| Condition | SEV |
|---|---|
| Data loss = True | SEV1 |
| ≥ 10 000 affected users | SEV1 |
| ≥ 120 min duration | SEV1 |
| ≥ 1 000 users or ≥ 30 min | SEV2 |
| ≥ 100 users or ≥ 10 min | SEV3 |
| All else | SEV4 |

---

## Exercise 3

**Add `lessons_learned` to `PostMortem` and update `generate_post_mortem_markdown()`.**

The `lessons_learned` field was already added to the `PostMortem` model above. Below is the
updated markdown generator:

```python
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def generate_post_mortem_markdown(pm: PostMortem) -> str:
    """Render a PostMortem object as a formatted Markdown document.

    Sections:
        - Header (incident metadata)
        - Root Cause
        - Timeline
        - Action Items
        - Lessons Learned  ← new in Exercise 3

    Args:
        pm: Fully populated PostMortem instance.

    Returns:
        Markdown string suitable for a wiki page or PR comment.
    """
    lines: list[str] = []

    # --- Header ---
    lines += [
        f"# Post-Mortem: {pm.title}",
        "",
        f"**Incident ID:** {pm.incident_id}  ",
        f"**Date:** {pm.date.strftime('%Y-%m-%d')}  ",
        f"**Severity:** {pm.severity}  ",
        f"**Duration:** {pm.duration_minutes} minutes  ",
        f"**Affected Users:** {pm.affected_users:,}  ",
        f"**Data Loss:** {'Yes ⚠️' if pm.data_loss else 'No'}  ",
        "",
    ]

    # --- Root Cause ---
    lines += [
        "## Root Cause",
        "",
        pm.root_cause,
        "",
    ]

    # --- Timeline ---
    lines += [
        "## Timeline",
        "",
        pm.timeline_to_markdown(),
        "",
    ]

    # --- Action Items ---
    lines += ["## Action Items", ""]
    if pm.action_items:
        for item in pm.action_items:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- *(None recorded.)*")
    lines.append("")

    # --- Lessons Learned (Exercise 3) ---
    lines += ["## Lessons Learned", ""]
    if pm.lessons_learned:
        for lesson in pm.lessons_learned:
            lines.append(f"- {lesson}")
    else:
        lines.append("- *(None recorded.)*")
    lines.append("")

    doc = "\n".join(lines)
    logger.info("Generated post-mortem markdown (%d chars) for %s", len(doc), pm.incident_id)
    return doc


# --- Full example ---
if __name__ == "__main__":
    from datetime import datetime, timezone

    pm = PostMortem(
        incident_id="INC-2025-042",
        title="API Gateway timeout storm",
        date=datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc),
        severity=score_severity(47, 3200, False).value,
        duration_minutes=47,
        affected_users=3200,
        data_loss=False,
        root_cause="Misconfigured upstream timeout propagation after deploy v2.14.1.",
        timeline=[
            TimelineEntry(
                timestamp=datetime(2025, 6, 15, 13, 58, tzinfo=timezone.utc),
                description="Deploy v2.14.1 completed",
                author="ci-cd-bot",
            ),
            TimelineEntry(
                timestamp=datetime(2025, 6, 15, 14, 3, tzinfo=timezone.utc),
                description="Alerts fired: p99 latency > 5s",
            ),
        ],
        action_items=[
            "Add timeout propagation integration test to CI pipeline.",
            "Review all upstream timeout configs before next deploy window.",
        ],
        lessons_learned=[
            "Deploy windows on Fridays increase blast radius — consider freeze policy.",
            "Timeout misconfiguration is not caught by unit tests alone; need load tests.",
            "Runbook for timeout storms was missing; authored and linked during incident.",
        ],
    )

    print(generate_post_mortem_markdown(pm))
```

---

## Exercise 4

**Extend `GrafanaDashboardSpec` with `repeat_variable: str | None` and emit `"repeat"` in JSON.**

```python
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, model_serializer

logger = logging.getLogger(__name__)


class GrafanaPanelTarget(BaseModel):
    """A single Prometheus query target for a Grafana panel."""

    expr: str
    legend_format: str = "{{model}}"
    ref_id: str = "A"


class GrafanaPanel(BaseModel):
    """Specification for a single Grafana panel (graph, stat, etc.)."""

    id: int
    title: str
    type: Literal["timeseries", "stat", "table", "heatmap"] = "timeseries"
    targets: list[GrafanaPanelTarget] = []
    grid_pos: dict[str, int] = {"h": 8, "w": 12, "x": 0, "y": 0}
    # Exercise 4: repeat this panel for each value of the named template variable
    repeat_variable: str | None = None

    def to_grafana_json(self) -> dict[str, Any]:
        """Serialise panel to Grafana JSON format.

        If repeat_variable is set, adds a "repeat" key so Grafana repeats the
        panel for each value of the named template variable (e.g. "model").
        """
        data: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "gridPos": self.grid_pos,
            "targets": [
                {"expr": t.expr, "legendFormat": t.legend_format, "refId": t.ref_id}
                for t in self.targets
            ],
        }
        if self.repeat_variable is not None:
            data["repeat"] = self.repeat_variable
            logger.debug(
                "Panel '%s' will repeat by variable '%s'", self.title, self.repeat_variable
            )
        return data


class GrafanaDashboardSpec(BaseModel):
    """Top-level Grafana dashboard specification."""

    title: str
    uid: str
    refresh: str = "30s"
    time_range: dict[str, str] = {"from": "now-1h", "to": "now"}
    panels: list[GrafanaPanel] = []
    # Template variables available in this dashboard (e.g. "model", "env")
    template_variables: list[str] = []

    def to_grafana_json(self) -> dict[str, Any]:
        """Serialise the full dashboard to a Grafana-compatible JSON payload."""
        templating: list[dict[str, Any]] = [
            {
                "name": var,
                "type": "query",
                "datasource": "Prometheus",
                "refresh": 2,
            }
            for var in self.template_variables
        ]

        dashboard: dict[str, Any] = {
            "title": self.title,
            "uid": self.uid,
            "refresh": self.refresh,
            "time": self.time_range,
            "panels": [p.to_grafana_json() for p in self.panels],
            "templating": {"list": templating},
            "schemaVersion": 36,
            "version": 1,
        }
        return dashboard

    def to_json_string(self, indent: int = 2) -> str:
        payload = self.to_grafana_json()
        logger.info("Serialised Grafana dashboard '%s' to JSON", self.title)
        return json.dumps(payload, indent=indent)


# --- Usage example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dashboard = GrafanaDashboardSpec(
        title="LLM Observability — Per-Model",
        uid="llm-obs-per-model",
        template_variables=["model", "env"],
        panels=[
            GrafanaPanel(
                id=1,
                title="Request Latency — $model",
                type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket{model="$model"}[5m]))',
                        legend_format="p95 {{model}}",
                    )
                ],
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
                # This panel will repeat for every value of the "model" variable
                repeat_variable="model",
            ),
            GrafanaPanel(
                id=2,
                title="Error Rate",
                type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='rate(llm_requests_total{status="error"}[5m])',
                        legend_format="errors/s",
                    )
                ],
                grid_pos={"h": 4, "w": 6, "x": 12, "y": 0},
                # No repeat — single panel for total error rate
            ),
        ],
    )

    print(dashboard.to_json_string())
```

**Sample JSON output for the first panel** (truncated):

```json
{
  "id": 1,
  "title": "Request Latency — $model",
  "type": "timeseries",
  "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
  "targets": [...],
  "repeat": "model"
}
```

The `"repeat": "model"` key instructs Grafana to clone the panel once per unique value of
the `model` template variable, enabling per-model breakdowns without manually duplicating
panels.

---

## Integrated Tests

```python
# tests/unit/test_postmortem.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from observability.postmortem import (
    PostMortem,
    TimelineEntry,
    SeverityLevel,
    score_severity,
    generate_post_mortem_markdown,
    GrafanaDashboardSpec,
    GrafanaPanel,
    GrafanaPanelTarget,
)


class TestTimelineToMarkdown:
    def _pm(self, entries: list[TimelineEntry]) -> PostMortem:
        return PostMortem(
            incident_id="INC-001",
            title="Test",
            date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity="SEV3",
            duration_minutes=15,
            affected_users=50,
            data_loss=False,
            root_cause="test",
            timeline=entries,
        )

    def test_empty_timeline_returns_placeholder(self) -> None:
        pm = self._pm([])
        md = pm.timeline_to_markdown()
        assert "No timeline entries" in md

    def test_entries_sorted_by_timestamp(self) -> None:
        entries = [
            TimelineEntry(
                timestamp=datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc),
                description="Later event",
            ),
            TimelineEntry(
                timestamp=datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc),
                description="Earlier event",
            ),
        ]
        md = self._pm(entries).timeline_to_markdown()
        lines = md.strip().splitlines()
        assert "13:00" in lines[0]
        assert "14:00" in lines[1]

    def test_author_included_when_present(self) -> None:
        entries = [
            TimelineEntry(
                timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
                description="Deployed",
                author="alice",
            )
        ]
        md = self._pm(entries).timeline_to_markdown()
        assert "alice" in md


class TestSeverityScoring:
    def test_data_loss_is_sev1(self) -> None:
        assert score_severity(5, 10, data_loss=True) == SeverityLevel.SEV1

    def test_ten_thousand_users_is_sev1(self) -> None:
        assert score_severity(5, 10_000, data_loss=False) == SeverityLevel.SEV1

    def test_120_minutes_is_sev1(self) -> None:
        assert score_severity(120, 0, data_loss=False) == SeverityLevel.SEV1

    def test_1000_users_is_sev2(self) -> None:
        assert score_severity(5, 1_000, data_loss=False) == SeverityLevel.SEV2

    def test_30_minutes_is_sev2(self) -> None:
        assert score_severity(30, 0, data_loss=False) == SeverityLevel.SEV2

    def test_100_users_is_sev3(self) -> None:
        assert score_severity(5, 100, data_loss=False) == SeverityLevel.SEV3

    def test_low_impact_is_sev4(self) -> None:
        assert score_severity(5, 10, data_loss=False) == SeverityLevel.SEV4


class TestLessonsLearnedInMarkdown:
    def test_lessons_appear_in_output(self) -> None:
        pm = PostMortem(
            incident_id="INC-002",
            title="Test",
            date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity="SEV3",
            duration_minutes=15,
            affected_users=50,
            data_loss=False,
            root_cause="A root cause.",
            lessons_learned=["Lesson one", "Lesson two"],
        )
        md = generate_post_mortem_markdown(pm)
        assert "## Lessons Learned" in md
        assert "Lesson one" in md
        assert "Lesson two" in md

    def test_empty_lessons_shows_placeholder(self) -> None:
        pm = PostMortem(
            incident_id="INC-003",
            title="Test",
            date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity="SEV4",
            duration_minutes=5,
            affected_users=5,
            data_loss=False,
            root_cause="Minor issue.",
        )
        md = generate_post_mortem_markdown(pm)
        assert "None recorded" in md


class TestGrafanaDashboardRepeat:
    def test_repeat_variable_emits_repeat_key(self) -> None:
        panel = GrafanaPanel(
            id=1,
            title="Latency",
            repeat_variable="model",
        )
        data = panel.to_grafana_json()
        assert data["repeat"] == "model"

    def test_no_repeat_variable_omits_repeat_key(self) -> None:
        panel = GrafanaPanel(id=2, title="Error Rate")
        data = panel.to_grafana_json()
        assert "repeat" not in data

    def test_dashboard_serialises_panels(self) -> None:
        dashboard = GrafanaDashboardSpec(
            title="Test",
            uid="test-uid",
            panels=[
                GrafanaPanel(id=1, title="P1", repeat_variable="model"),
                GrafanaPanel(id=2, title="P2"),
            ],
        )
        data = dashboard.to_grafana_json()
        assert len(data["panels"]) == 2
        assert data["panels"][0]["repeat"] == "model"
        assert "repeat" not in data["panels"][1]
```

**Run with:**

```bash
pytest tests/unit/test_postmortem.py -v
```
