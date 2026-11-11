# Day 4 Solutions: Multi-tenancy, PII, and SLOs

## Exercise 1: Tenant Isolation Level Selection

### Problem
You have customers ranging from individual developers to regulated financial enterprises. Design the criteria for choosing between namespace isolation, schema isolation, and cluster isolation.

### Answer

**Isolation level comparison:**

| Level | What's shared | What's isolated | Cost | Compliance |
|---|---|---|---|---|
| **Row-level** (filter) | Everything incl. infra | Only data (by filter) | $ | None |
| **Schema** | DB engine, compute | Tables, indexes per tenant | $$ | Low |
| **Namespace** (k8s) | Cluster, node pool | Network policies, RBAC, quotas | $$$ | Medium |
| **Cluster** | Nothing | Full k8s cluster per tenant | $$$$ | High |

**Decision criteria:**

```
IF tenant is in regulated industry (finance, healthcare, government)
    AND compliance requires "no shared infrastructure" (FedRAMP, SOC2 Type II, HIPAA BAA)
    → CLUSTER isolation

ELSE IF tenant has contractual data residency requirements (EU, China)
    → CLUSTER isolation (per-region cluster)

ELSE IF tenant requires network-level isolation (pen-test showed cross-tenant traffic risk)
    OR tenant SLA requires guaranteed compute (no noisy neighbor)
    → NAMESPACE isolation with dedicated node pool

ELSE IF tenant volume is high (>100K requests/day) with burst risk
    → NAMESPACE isolation (quota enforcement + network policy)

ELSE IF tenant is SMB or developer tier, cost is primary concern
    → SCHEMA isolation for data, shared namespace with RBAC for compute

ELSE (smallest tier, trust is implicit, cost-optimize)
    → ROW-LEVEL isolation (metadata filter in every query)
```

```python
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    NONE = "none"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    FEDRAMP = "fedramp"
    GDPR = "gdpr"


class IsolationLevel(str, Enum):
    ROW_LEVEL = "row_level"
    SCHEMA = "schema"
    NAMESPACE = "namespace"
    CLUSTER = "cluster"


@dataclass
class TenantRequirements:
    tenant_id: str
    compliance_frameworks: list[ComplianceFramework]
    data_residency_regions: list[str]           # e.g., ["eu-west-1"]
    requires_dedicated_compute: bool
    requires_network_isolation: bool
    daily_request_volume: int
    contractual_isolation_clause: bool          # Explicitly agreed isolation in contract
    pen_tested: bool                             # Has tenant run a pentest requiring isolation?


@dataclass
class IsolationDecision:
    tenant_id: str
    level: IsolationLevel
    reasons: list[str]
    k8s_namespace: str
    vector_store_scope: str     # "cluster" | "collection" | "schema" | "filter"
    estimated_monthly_overhead_usd: float


ISOLATION_COSTS_USD = {
    IsolationLevel.ROW_LEVEL:  0.0,
    IsolationLevel.SCHEMA:     50.0,    # Extra DB compute/storage
    IsolationLevel.NAMESPACE:  300.0,   # Extra k8s management overhead
    IsolationLevel.CLUSTER:    2000.0,  # Dedicated cluster (control plane, ingress, monitoring)
}

REGULATED_FRAMEWORKS = {ComplianceFramework.HIPAA, ComplianceFramework.FEDRAMP}


def select_isolation_level(req: TenantRequirements) -> IsolationDecision:
    reasons: list[str] = []
    level = IsolationLevel.ROW_LEVEL  # Start with minimum

    # Hard requirements → cluster isolation
    if any(f in REGULATED_FRAMEWORKS for f in req.compliance_frameworks):
        level = IsolationLevel.CLUSTER
        reasons.append(f"Regulated framework requires cluster isolation: {req.compliance_frameworks}")

    elif req.data_residency_regions:
        level = IsolationLevel.CLUSTER
        reasons.append(f"Data residency requirement: {req.data_residency_regions}")

    elif req.contractual_isolation_clause:
        level = IsolationLevel.CLUSTER
        reasons.append("Contractual isolation clause — explicit SLA commitment")

    # Strong requirements → namespace
    elif req.requires_network_isolation or req.pen_tested:
        level = IsolationLevel.NAMESPACE
        reasons.append("Network isolation required (pen-test or security assessment)")

    elif req.requires_dedicated_compute:
        level = IsolationLevel.NAMESPACE
        reasons.append("Dedicated compute SLA — namespace with node affinity")

    elif req.daily_request_volume > 100_000:
        level = IsolationLevel.NAMESPACE
        reasons.append(f"High volume ({req.daily_request_volume}/day) — quota enforcement needed")

    # Medium requirements → schema
    elif ComplianceFramework.GDPR in req.compliance_frameworks:
        level = IsolationLevel.SCHEMA
        reasons.append("GDPR: schema isolation enables targeted deletion without full-table scan")

    elif req.daily_request_volume > 10_000:
        level = IsolationLevel.SCHEMA
        reasons.append(f"Medium volume ({req.daily_request_volume}/day) — schema isolation for query performance")

    # Otherwise row-level
    else:
        reasons.append("Small tenant — row-level filter sufficient")

    # Derive concrete resource names
    safe_id = req.tenant_id.lower().replace("-", "")
    k8s_namespace = f"tenant-{safe_id}" if level in (IsolationLevel.NAMESPACE, IsolationLevel.CLUSTER) else "shared"
    vector_store_scope = {
        IsolationLevel.CLUSTER:    "cluster",
        IsolationLevel.NAMESPACE:  "collection",
        IsolationLevel.SCHEMA:     "schema",
        IsolationLevel.ROW_LEVEL:  "filter",
    }[level]

    decision = IsolationDecision(
        tenant_id=req.tenant_id,
        level=level,
        reasons=reasons,
        k8s_namespace=k8s_namespace,
        vector_store_scope=vector_store_scope,
        estimated_monthly_overhead_usd=ISOLATION_COSTS_USD[level],
    )
    logger.info(
        "Tenant %s → isolation=%s k8s_ns=%s vector=%s cost=$%.0f/mo reasons=%s",
        req.tenant_id, level.value, k8s_namespace, vector_store_scope,
        decision.estimated_monthly_overhead_usd, reasons,
    )
    return decision
```

**Migration path**: start tenants at row-level, promote to schema/namespace as volume grows. Never demote (contractual risk). Build a migration script that copies data to new isolation scope, then atomically updates the routing table.

---

## Exercise 2: PII Redaction Pipeline for RAG

### Problem
Your RAG system stores customer documents that may contain PII (names, SSNs, emails, phone numbers, addresses). Design a pipeline that handles PII in stored chunks AND at query time. When should you redact at ingest vs at retrieval?

### Answer

**Two separate concerns:**
1. **Stored chunks**: PII redacted at ingest → no PII in vector store or document store
2. **Query-time**: User's query may contain PII → don't log it; don't embed it in audit trails

**When to redact at ingest vs retrieval:**

| Scenario | Ingest-time redaction | Query-time redaction |
|---|---|---|
| PII in stored documents (patient records, contracts) | YES — redact before indexing | N/A (already clean) |
| User query contains PII ("what is John Doe's account?") | N/A (not stored) | YES — scrub from logs |
| PII needed for search accuracy (searching BY a person's name) | Partial — pseudonymize, don't redact | Expand pseudonym in retrieval |
| Multi-tenant: tenant B queries tenant A's PII | Not relevant (isolation prevents this) | N/A |

**Key design decision: redact vs pseudonymize**

- **Redact**: replace `John Doe` with `[PERSON]` — simplest, but breaks name-based retrieval
- **Pseudonymize**: replace `John Doe` with a deterministic token `PERSON_7f3a9b` — preserves searchability, reversible with key, GDPR-compliant (right to erasure = delete key mapping)

For legal/medical RAG: pseudonymize for ingest, redact for LLM context window.

```python
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Compiled PII patterns (order matters: more specific first)
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN",   re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("PHONE", re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b')),
    ("EMAIL", re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b')),
    ("IP",    re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
    # Named entity patterns are handled by NER model in production
    # These are heuristic fallbacks:
    ("POSTAL_CODE", re.compile(r'\b\d{5}(?:-\d{4})?\b')),
    ("CREDIT_CARD", re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')),
]


@dataclass
class RedactionResult:
    original_text: str
    redacted_text: str
    pseudonymized_text: str
    entities_found: list[tuple[str, str, str]]  # (entity_type, original_value, token)
    redaction_count: int


def _pseudonym_token(entity_type: str, value: str, tenant_id: str) -> str:
    """Deterministic pseudonym. Same value + tenant always maps to same token."""
    digest = hashlib.sha256(f"{tenant_id}:{entity_type}:{value}".encode()).hexdigest()[:8]
    return f"{entity_type}_{digest}"


def redact_pii(
    text: str,
    tenant_id: str,
    mode: str = "pseudonymize",  # "redact" | "pseudonymize"
) -> RedactionResult:
    """
    Redact or pseudonymize PII in text.
    mode="pseudonymize": replaces with reversible tokens (for indexing)
    mode="redact": replaces with [ENTITY_TYPE] (for LLM context)
    """
    entities_found: list[tuple[str, str, str]] = []
    result = text

    for entity_type, pattern in PII_PATTERNS:
        for match in pattern.finditer(result):
            original = match.group()
            if mode == "pseudonymize":
                replacement = _pseudonym_token(entity_type, original, tenant_id)
            else:
                replacement = f"[{entity_type}]"
            entities_found.append((entity_type, original, replacement))

        if mode == "pseudonymize":
            result = pattern.sub(
                lambda m, et=entity_type: _pseudonym_token(et, m.group(), tenant_id),
                result,
            )
        else:
            result = pattern.sub(f"[{entity_type}]", result)

    return RedactionResult(
        original_text=text,
        redacted_text=result if mode == "redact" else text,
        pseudonymized_text=result if mode == "pseudonymize" else text,
        entities_found=entities_found,
        redaction_count=len(entities_found),
    )


class PIIAwarePipeline:
    """
    RAG pipeline with PII handling at both ingest and retrieval time.
    """

    def __init__(self, tenant_id: str) -> None:
        self._tenant_id = tenant_id
        # token_map: pseudonym_token → original value (stored encrypted, not here)
        self._token_map: dict[str, str] = {}

    def process_for_ingest(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Process document chunk for storage.
        Returns (pseudonymized_text, token_to_original_map).
        Store token_map encrypted; never store original PII in vector DB.
        """
        result = redact_pii(text, self._tenant_id, mode="pseudonymize")
        token_map = {token: original for _, original, token in result.entities_found}

        if result.redaction_count > 0:
            logger.info(
                "Ingest: pseudonymized %d PII entities in chunk (tenant=%s)",
                result.redaction_count, self._tenant_id,
            )

        return result.pseudonymized_text, token_map

    def process_for_llm_context(
        self,
        retrieved_chunks: list[str],
        requester_has_pii_access: bool = False,
    ) -> list[str]:
        """
        Prepare retrieved chunks for LLM context window.
        If requester lacks PII access: redact all pseudonyms too.
        If requester has PII access: de-pseudonymize (re-insert original values).
        """
        if requester_has_pii_access:
            # Expand pseudonyms back to original values
            result = []
            for chunk in retrieved_chunks:
                expanded = chunk
                for token, original in self._token_map.items():
                    expanded = expanded.replace(token, original)
                result.append(expanded)
            return result
        else:
            # Replace pseudonyms with generic [ENTITY_TYPE] labels
            result = []
            for chunk in retrieved_chunks:
                clean = chunk
                for token, original in self._token_map.items():
                    # Extract entity type from token format: TYPE_hexdigest
                    entity_type = token.rsplit("_", 1)[0]
                    clean = clean.replace(token, f"[{entity_type}]")
                result.append(clean)
            return result

    def scrub_query_from_logs(self, query: str) -> str:
        """Return a safe version of the query for audit logs. Original never logged."""
        result = redact_pii(query, self._tenant_id, mode="redact")
        if result.redaction_count > 0:
            logger.debug("Query contained %d PII entities — scrubbed for logging", result.redaction_count)
        return result.redacted_text
```

**GDPR right to erasure implementation:**
```python
def handle_erasure_request(tenant_id: str, subject_id: str, db) -> int:
    """
    Delete all PII for a data subject. With pseudonymization, this means:
    1. Find all pseudonym tokens that map to this subject's data
    2. Delete the token_map entries (decouples pseudonym from real data)
    3. The pseudonym tokens in the vector store become meaningless noise
    No need to reindex — pseudonyms without a key are just opaque strings.
    """
    deleted = db.delete_token_mappings(tenant_id=tenant_id, subject_id=subject_id)
    logger.info("Erasure request: tenant=%s subject=%s deleted=%d token mappings", tenant_id, subject_id, deleted)
    return deleted
```

---

## Exercise 3: Cost SLO Design

### Problem
Design per-tenant monthly cost budgets with burst allowances. How do you enforce them without disrupting legitimate usage?

### Answer

**Three-tier budget model:**

```
Base budget:   Normal monthly spend limit (hard cap if exceeded)
Soft limit:    80% of base budget → notify tenant
Burst budget:  Up to 3× hourly baseline for ≤4h/day (burst is temporary, not sustained)
```

**Enforcement levels (in order of escalation):**

| Budget State | Action |
|---|---|
| <70% monthly | No action |
| 70–80% monthly | Dashboard warning to tenant |
| 80–90% monthly | Email alert to tenant admin |
| 90–100% monthly | In-app banner; throttle to 50% of peak rate |
| >100% monthly | Hard cap — requests return 429 with `Retry-After: next month` |
| Hourly spike >burst_allowance | Throttle for that hour; no monthly penalty |

```python
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    THROTTLED = "throttled"
    CAPPED = "capped"


@dataclass
class TenantBudgetConfig:
    tenant_id: str
    monthly_budget_usd: float
    burst_multiplier: float = 3.0           # Max hourly rate relative to baseline
    burst_max_hours_per_day: int = 4        # Max hours per day at burst rate
    throttle_rate_fraction: float = 0.5     # 50% of normal rate when throttled


@dataclass
class BudgetStatus:
    tenant_id: str
    month_to_date_usd: float
    monthly_budget_usd: float
    budget_fraction_used: float
    state: BudgetState
    throttle_fraction: float                # 1.0 = no throttle, 0.5 = half rate
    burst_hours_used_today: int
    message: str


def evaluate_budget_status(
    config: TenantBudgetConfig,
    month_to_date_usd: float,
    current_hourly_usd: float,
    hourly_baseline_usd: float,
    burst_hours_used_today: int,
) -> BudgetStatus:
    """
    Determine current budget state and throttle fraction.
    Called on each request (cached for 60s to avoid per-request DB lookups).
    """
    fraction = month_to_date_usd / config.monthly_budget_usd if config.monthly_budget_usd > 0 else 0.0

    # Burst check: is current hour above burst allowance?
    is_bursting = (
        hourly_baseline_usd > 0
        and current_hourly_usd > hourly_baseline_usd * config.burst_multiplier
    )
    burst_hours_exceeded = burst_hours_used_today >= config.burst_max_hours_per_day

    # Hard cap: monthly budget exhausted
    if fraction >= 1.0:
        return BudgetStatus(
            tenant_id=config.tenant_id,
            month_to_date_usd=month_to_date_usd,
            monthly_budget_usd=config.monthly_budget_usd,
            budget_fraction_used=fraction,
            state=BudgetState.CAPPED,
            throttle_fraction=0.0,
            burst_hours_used_today=burst_hours_used_today,
            message=f"Monthly budget ${config.monthly_budget_usd:.0f} exhausted. Resets next billing cycle.",
        )

    # Throttle: approaching limit
    if fraction >= 0.90:
        return BudgetStatus(
            tenant_id=config.tenant_id,
            month_to_date_usd=month_to_date_usd,
            monthly_budget_usd=config.monthly_budget_usd,
            budget_fraction_used=fraction,
            state=BudgetState.THROTTLED,
            throttle_fraction=config.throttle_rate_fraction,
            burst_hours_used_today=burst_hours_used_today,
            message=f"Budget {fraction:.0%} used — throttled to {config.throttle_rate_fraction:.0%} capacity",
        )

    # Burst exceeded today
    if is_bursting and burst_hours_exceeded:
        return BudgetStatus(
            tenant_id=config.tenant_id,
            month_to_date_usd=month_to_date_usd,
            monthly_budget_usd=config.monthly_budget_usd,
            budget_fraction_used=fraction,
            state=BudgetState.THROTTLED,
            throttle_fraction=config.throttle_rate_fraction,
            burst_hours_used_today=burst_hours_used_today,
            message=f"Daily burst allowance ({config.burst_max_hours_per_day}h) exhausted",
        )

    # Warning zone
    if fraction >= 0.80:
        state = BudgetState.ALERT if fraction >= 0.90 else BudgetState.WARNING
        return BudgetStatus(
            tenant_id=config.tenant_id,
            month_to_date_usd=month_to_date_usd,
            monthly_budget_usd=config.monthly_budget_usd,
            budget_fraction_used=fraction,
            state=state,
            throttle_fraction=1.0,
            burst_hours_used_today=burst_hours_used_today,
            message=f"Budget {fraction:.0%} used — {100*(1-fraction):.0f}% remaining",
        )

    return BudgetStatus(
        tenant_id=config.tenant_id,
        month_to_date_usd=month_to_date_usd,
        monthly_budget_usd=config.monthly_budget_usd,
        budget_fraction_used=fraction,
        state=BudgetState.NORMAL,
        throttle_fraction=1.0,
        burst_hours_used_today=burst_hours_used_today,
        message="Within budget",
    )


class TokenBucketThrottler:
    """
    Token bucket throttler for rate-limited tenants.
    Fills at throttle_fraction × normal_rate tokens/second.
    """

    def __init__(self, normal_rate_rps: float, burst_capacity: int = 10) -> None:
        self._normal_rate = normal_rate_rps
        self._burst_capacity = burst_capacity
        self._tokens = float(burst_capacity)
        self._last_refill = time.monotonic()

    def allow_request(self, throttle_fraction: float) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        refill_rate = self._normal_rate * throttle_fraction
        self._tokens = min(
            float(self._burst_capacity),
            self._tokens + elapsed * refill_rate,
        )

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True

        logger.debug("Request throttled. Token bucket empty. Rate=%.1f rps", refill_rate)
        return False
```

---

## Exercise 4: Latency SLO Definition

### Problem
Define P50/P95/P99 latency targets for three call patterns: RAG pipeline, agent (multi-step), and LLM direct call. Justify each target.

### Answer

**Latency components per call type:**

**LLM direct call** (single API call, streaming):
- TTFT (time to first token): the user-visible metric for interactive use
- Total duration: relevant for batch, cost estimation

**RAG pipeline** (embed query → retrieve → rerank → LLM → stream):
- Embedding: 30–80ms (fast model)
- Vector search: 20–50ms (well-tuned HNSW)
- Reranking: 50–200ms (cross-encoder model)
- LLM: TTFT 200–800ms + generation time
- Total TTFT: 300–1100ms

**Agent** (multiple LLM calls + tool calls, variable steps):
- Each LLM hop: 300–1000ms
- Each tool call: 100ms–5s depending on tool
- Typical 3-step agent: 2–8s total

**SLO targets:**

| Call Type | P50 | P95 | P99 | Hard Breach |
|---|---|---|---|---|
| LLM direct (TTFT) | 300ms | 800ms | 1500ms | 3000ms |
| LLM direct (total, 500 token response) | 2s | 5s | 10s | 20s |
| RAG pipeline (TTFT) | 600ms | 1500ms | 3000ms | 5000ms |
| RAG pipeline (total) | 3s | 8s | 15s | 30s |
| Agent (3-step, total) | 5s | 15s | 30s | 60s |
| Agent (5-step, total) | 10s | 30s | 60s | 120s |

**Rationale:**
- TTFT matters more than total for interactive UI — show the user something is happening
- RAG has irreducible retrieval latency (200–500ms) — P50 can't be below 500ms realistically
- Agent SLOs are looser because users accept waiting when they know a complex task is running
- P99 for LLM direct at 1500ms: network jitter, cold token buffer, occasional model slowdown
- Hard breach threshold (double P99): triggers circuit breaker / fallback model

```python
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CallType(str, Enum):
    LLM_DIRECT = "llm_direct"
    RAG_PIPELINE = "rag_pipeline"
    AGENT = "agent"


@dataclass
class LatencySLO:
    call_type: CallType
    p50_ms: int
    p95_ms: int
    p99_ms: int
    hard_breach_ms: int
    measure: str  # "ttft" | "total"


LATENCY_SLOS: dict[tuple[CallType, str], LatencySLO] = {
    (CallType.LLM_DIRECT, "ttft"):   LatencySLO(CallType.LLM_DIRECT, 300, 800, 1500, 3000, "ttft"),
    (CallType.LLM_DIRECT, "total"):  LatencySLO(CallType.LLM_DIRECT, 2000, 5000, 10000, 20000, "total"),
    (CallType.RAG_PIPELINE, "ttft"): LatencySLO(CallType.RAG_PIPELINE, 600, 1500, 3000, 5000, "ttft"),
    (CallType.RAG_PIPELINE, "total"):LatencySLO(CallType.RAG_PIPELINE, 3000, 8000, 15000, 30000, "total"),
    (CallType.AGENT, "total"):       LatencySLO(CallType.AGENT, 5000, 15000, 30000, 60000, "total"),
}


@dataclass
class LatencyObservation:
    call_type: CallType
    measure: str
    duration_ms: float
    slo_p50_met: bool
    slo_p95_met: bool
    slo_p99_met: bool
    hard_breach: bool
    tenant_id: str
    trace_id: str


def check_slo(
    call_type: CallType,
    measure: str,
    duration_ms: float,
    tenant_id: str,
    trace_id: str,
) -> LatencyObservation:
    """
    Check a single request's latency against SLOs.
    In production, aggregate these into Prometheus histograms and
    alert when the aggregated quantile breaches SLO.
    """
    slo = LATENCY_SLOS.get((call_type, measure))
    if slo is None:
        logger.warning("No SLO defined for (%s, %s)", call_type, measure)
        return LatencyObservation(
            call_type=call_type, measure=measure, duration_ms=duration_ms,
            slo_p50_met=True, slo_p95_met=True, slo_p99_met=True, hard_breach=False,
            tenant_id=tenant_id, trace_id=trace_id,
        )

    obs = LatencyObservation(
        call_type=call_type,
        measure=measure,
        duration_ms=duration_ms,
        slo_p50_met=duration_ms <= slo.p50_ms,
        slo_p95_met=duration_ms <= slo.p95_ms,
        slo_p99_met=duration_ms <= slo.p99_ms,
        hard_breach=duration_ms > slo.hard_breach_ms,
        tenant_id=tenant_id,
        trace_id=trace_id,
    )

    if obs.hard_breach:
        logger.error(
            "SLO HARD BREACH: %s %s duration=%.0fms hard_limit=%dms tenant=%s trace=%s",
            call_type.value, measure, duration_ms, slo.hard_breach_ms, tenant_id, trace_id,
        )
    elif not obs.slo_p99_met:
        logger.warning(
            "SLO P99 breach: %s %s duration=%.0fms p99=%dms tenant=%s trace=%s",
            call_type.value, measure, duration_ms, slo.p99_ms, tenant_id, trace_id,
        )

    return obs


class LatencyTimer:
    """Context manager for timing and SLO checking."""

    def __init__(
        self,
        call_type: CallType,
        measure: str,
        tenant_id: str,
        trace_id: str,
        ttft_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._call_type = call_type
        self._measure = measure
        self._tenant_id = tenant_id
        self._trace_id = trace_id
        self._ttft_callback = ttft_callback
        self._start: float = 0.0
        self._ttft_recorded: bool = False

    def __enter__(self) -> "LatencyTimer":
        self._start = time.monotonic()
        return self

    def record_first_token(self) -> None:
        if not self._ttft_recorded and self._ttft_callback:
            ttft_ms = (time.monotonic() - self._start) * 1000
            self._ttft_recorded = True
            self._ttft_callback(ttft_ms)
            check_slo(self._call_type, "ttft", ttft_ms, self._tenant_id, self._trace_id)

    def __exit__(self, *_) -> None:
        total_ms = (time.monotonic() - self._start) * 1000
        check_slo(self._call_type, self._measure, total_ms, self._tenant_id, self._trace_id)
```

**Prometheus SLO alerting rule:**
```yaml
# Alert when p95 latency is breached over a 5-minute window
groups:
  - name: latency_slo
    rules:
      - alert: RAGPipelineP95Breach
        expr: |
          histogram_quantile(0.95,
            sum(rate(request_duration_ms_bucket{call_type="rag_pipeline"}[5m])) by (le)
          ) > 1500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG P95 latency {{ $value }}ms > SLO 1500ms"

      - alert: AgentHardBreach
        expr: |
          histogram_quantile(0.99,
            sum(rate(request_duration_ms_bucket{call_type="agent"}[5m])) by (le)
          ) > 60000
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Agent P99 latency {{ $value }}ms > hard breach 60000ms"
```

**SLO error budget:**
- 99.5% of requests must meet P95 target (0.5% budget for P95 violations)
- P99 violations burn error budget 20× faster than P95 — fix them first
- Hard breaches burn the budget immediately — trigger incident response
