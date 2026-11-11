# Day 3 Solutions: LLM Serving at Scale

## Exercise 1: Routing Strategy for Mixed Workloads

### Problem
Design a routing strategy for three workload types:
- **Real-time chat**: latency-sensitive (<500ms TTFT), interactive, short-medium prompts
- **Batch analytics**: throughput-sensitive, long documents, latency-tolerant (minutes OK)
- **Code generation**: medium latency tolerance (<3s), requires high-quality models, often large outputs

### Answer

**Routing dimensions:**
1. **Model tier**: haiku (fast/cheap) → sonnet (balanced) → opus (quality/slow)
2. **Serving mode**: real-time (low batch size) vs batch (high throughput mode)
3. **Queue**: priority queues per workload type; batch jobs don't starve chat

**Routing decision matrix:**

| Workload | Model | Mode | Max Queue Wait | Priority |
|---|---|---|---|---|
| Real-time chat | claude-haiku-5-5 | streaming | 200ms | HIGH |
| Code generation | claude-sonnet-5-5 | streaming | 2s | MEDIUM |
| Batch analytics | claude-sonnet-5-5 or claude-haiku-5-5 | batch | 5min | LOW |
| Long doc summarization | claude-haiku-5-5 (cost) | batch | 10min | LOW |

**Key insight**: never route batch work to real-time serving capacity. Batch jobs that land on real-time pods block short requests. Use separate autoscaling groups.

```python
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class WorkloadType(str, Enum):
    REALTIME_CHAT = "realtime_chat"
    CODE_GENERATION = "code_generation"
    BATCH_ANALYTICS = "batch_analytics"
    LONG_DOC = "long_doc"


class ServingMode(str, Enum):
    STREAMING = "streaming"
    BATCH = "batch"


class Priority(int, Enum):
    HIGH = 10
    MEDIUM = 5
    LOW = 1


@dataclass
class RoutingDecision:
    workload_type: WorkloadType
    model: str
    serving_mode: ServingMode
    priority: Priority
    max_queue_wait_ms: int
    endpoint_pool: str  # k8s service name


@dataclass
class RequestFeatures:
    prompt_tokens: int
    is_streaming_requested: bool
    is_interactive: bool      # User is waiting at keyboard
    expected_output_tokens: int
    workload_hint: Optional[WorkloadType] = None  # Caller-provided hint
    tenant_sla_tier: str = "standard"  # "standard" | "premium"


def classify_workload(features: RequestFeatures) -> WorkloadType:
    """
    Classify request into workload type.
    Caller hint is used if provided and valid; otherwise inferred from features.
    """
    if features.workload_hint:
        return features.workload_hint

    # Heuristic classification
    if features.is_interactive and features.prompt_tokens < 2000:
        return WorkloadType.REALTIME_CHAT

    if features.prompt_tokens > 8000 and not features.is_interactive:
        return WorkloadType.LONG_DOC

    # Detect code generation: high expected output, moderate input
    if features.expected_output_tokens > 1000 and features.prompt_tokens < 4000:
        return WorkloadType.CODE_GENERATION

    if not features.is_streaming_requested and not features.is_interactive:
        return WorkloadType.BATCH_ANALYTICS

    return WorkloadType.REALTIME_CHAT  # Default to conservative


ROUTING_TABLE: dict[WorkloadType, RoutingDecision] = {
    WorkloadType.REALTIME_CHAT: RoutingDecision(
        workload_type=WorkloadType.REALTIME_CHAT,
        model="claude-haiku-5-5",
        serving_mode=ServingMode.STREAMING,
        priority=Priority.HIGH,
        max_queue_wait_ms=200,
        endpoint_pool="llm-realtime-pool",
    ),
    WorkloadType.CODE_GENERATION: RoutingDecision(
        workload_type=WorkloadType.CODE_GENERATION,
        model="claude-sonnet-5-5",
        serving_mode=ServingMode.STREAMING,
        priority=Priority.MEDIUM,
        max_queue_wait_ms=2000,
        endpoint_pool="llm-realtime-pool",
    ),
    WorkloadType.BATCH_ANALYTICS: RoutingDecision(
        workload_type=WorkloadType.BATCH_ANALYTICS,
        model="claude-sonnet-5-5",
        serving_mode=ServingMode.BATCH,
        priority=Priority.LOW,
        max_queue_wait_ms=300_000,  # 5 minutes
        endpoint_pool="llm-batch-pool",
    ),
    WorkloadType.LONG_DOC: RoutingDecision(
        workload_type=WorkloadType.LONG_DOC,
        model="claude-haiku-5-5",
        serving_mode=ServingMode.BATCH,
        priority=Priority.LOW,
        max_queue_wait_ms=600_000,  # 10 minutes
        endpoint_pool="llm-batch-pool",
    ),
}


class WorkloadRouter:
    def __init__(self) -> None:
        self._queue_depths: dict[str, int] = {
            "llm-realtime-pool": 0,
            "llm-batch-pool": 0,
        }

    def route(self, features: RequestFeatures) -> RoutingDecision:
        workload = classify_workload(features)
        decision = ROUTING_TABLE[workload]

        # Premium SLA: upgrade model tier for non-batch workloads
        if features.tenant_sla_tier == "premium" and workload != WorkloadType.LONG_DOC:
            decision = RoutingDecision(
                **{**decision.__dict__, "model": "claude-sonnet-5-5"}
            )

        # Circuit breaker: if realtime pool is overloaded, degrade gracefully
        if (
            decision.endpoint_pool == "llm-realtime-pool"
            and self._queue_depths["llm-realtime-pool"] > 50
            and workload == WorkloadType.CODE_GENERATION
        ):
            logger.warning(
                "Realtime pool queue depth %d — routing code_generation to batch",
                self._queue_depths["llm-realtime-pool"],
            )
            decision = ROUTING_TABLE[WorkloadType.BATCH_ANALYTICS]

        logger.info(
            "Routed workload=%s model=%s pool=%s priority=%s",
            workload.value, decision.model, decision.endpoint_pool, decision.priority.name,
        )
        return decision

    def update_queue_depth(self, pool: str, depth: int) -> None:
        self._queue_depths[pool] = depth
```

**Batch aggregation for analytics:**
```python
import asyncio
from collections import defaultdict


class BatchAggregator:
    """
    Collects batch requests and submits them together for throughput.
    Max batch size: 50 requests or 5s wait, whichever comes first.
    """
    MAX_BATCH_SIZE = 50
    MAX_WAIT_SECONDS = 5.0

    def __init__(self, submit_fn) -> None:
        self._queues: dict[str, list] = defaultdict(list)
        self._submit_fn = submit_fn

    async def add(self, model: str, request: dict) -> dict:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._queues[model].append((request, future))

        if len(self._queues[model]) >= self.MAX_BATCH_SIZE:
            await self._flush(model)

        return await asyncio.wait_for(future, timeout=self.MAX_WAIT_SECONDS + 10)

    async def _flush(self, model: str) -> None:
        batch = self._queues.pop(model, [])
        if not batch:
            return
        requests = [r for r, _ in batch]
        futures = [f for _, f in batch]
        results = await self._submit_fn(model, requests)
        for future, result in zip(futures, results):
            if not future.done():
                future.set_result(result)
```

---

## Exercise 2: Continuous Batching with PagedAttention

### Problem
Explain how iteration-level scheduling (continuous batching) works, how PagedAttention enables it, and what happens when a sequence is evicted mid-generation.

### Answer

**Traditional static batching problem:**
All sequences in a batch must finish before the next batch starts. A 5-token request waits for a 500-token request to finish. GPU utilization collapses when batch members have variable lengths.

**Continuous batching (Orca / vLLM model):**

In continuous batching, the scheduler operates at the *iteration* (token generation step) level, not the request level:

1. At each decode step, the scheduler checks: "are any sequences finished? Any new requests waiting?"
2. Finished sequences are immediately evicted; their KV cache pages are freed
3. New requests are immediately admitted into the freed capacity — same forward pass
4. Result: GPU is never idle waiting for slow sequences

**PagedAttention enables this by:**
- Breaking KV cache into fixed-size *pages* (e.g., 16 token blocks per page)
- Pages are allocated on demand as a sequence grows — no pre-allocated contiguous memory
- When a sequence finishes, its pages are returned to a free-page pool immediately
- New sequences can be admitted as soon as enough free pages exist

```
Traditional:  [seq1: 500 tokens][seq2: 5 tokens  ][seq3: 200 tokens]
              Seq2 done at t=5 but must wait until t=500 for seq1.
              Waste: 495 idle decode steps.

Continuous:   [seq1: step t=1  ][seq2: step t=1  ][seq3: step t=1  ]
              At t=5: seq2 done, evicted → seq4 admitted to same GPU memory
              [seq1: step t=6  ][seq4: step t=1  ][seq3: step t=6  ]
```

**Eviction mid-generation:**

When the KV cache is full and a new high-priority request arrives, the scheduler must evict a partially-complete sequence. Two strategies:

1. **Preemption with recomputation**: Drop the evicted sequence's KV cache. When it resumes, recompute the KV cache from the prompt tokens. Cost: O(prompt_length) recomputation. Acceptable for short prompts; expensive for 8K token contexts.

2. **Swap to CPU**: Copy KV cache pages to CPU DRAM. When the sequence resumes, swap back. Cost: PCIe bandwidth (~32 GB/s). At 1536 layers × 2 (K+V) × 32 heads × 128 head_dim × 4 bytes ≈ 48KB per token × 500 tokens ≈ 24MB per sequence swap. At 32 GB/s: ~0.75ms swap. Acceptable.

```python
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"       # KV cache on CPU
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class KVCachePage:
    page_id: int
    token_count: int
    on_gpu: bool = True


@dataclass
class Sequence:
    seq_id: str
    prompt_tokens: int
    generated_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    priority: int = 5
    pages: list[KVCachePage] = field(default_factory=list)
    preemption_count: int = 0


class ContinuousBatchScheduler:
    """
    Simplified continuous batching scheduler.
    Real implementation: vLLM's BlockSpaceManager + Scheduler.
    """

    PAGE_SIZE = 16  # tokens per KV cache page

    def __init__(self, total_gpu_pages: int, max_batch_size: int = 256) -> None:
        self._total_pages = total_gpu_pages
        self._free_pages = total_gpu_pages
        self._max_batch = max_batch_size
        self._running: list[Sequence] = []
        self._waiting: list[Sequence] = []
        self._swapped: list[Sequence] = []

    def _pages_needed(self, seq: Sequence) -> int:
        total_tokens = seq.prompt_tokens + seq.generated_tokens + 1  # +1 for next token
        return (total_tokens + self.PAGE_SIZE - 1) // self.PAGE_SIZE

    def schedule_iteration(self) -> list[Sequence]:
        """
        Called once per decode step.
        Returns the batch of sequences to run this iteration.
        """
        # 1. Evict finished sequences first — free their pages
        finished = [s for s in self._running if s.status == SequenceStatus.FINISHED]
        for seq in finished:
            freed = len(seq.pages)
            self._free_pages += freed
            self._running.remove(seq)
            logger.debug("Evicted finished seq=%s freed=%d pages", seq.seq_id, freed)

        # 2. Try to swap back preempted sequences if pages available
        for seq in list(self._swapped):
            needed = self._pages_needed(seq)
            if self._free_pages >= needed:
                self._free_pages -= needed
                seq.status = SequenceStatus.RUNNING
                self._swapped.remove(seq)
                self._running.append(seq)
                logger.info("Swapped in seq=%s needed=%d pages", seq.seq_id, needed)

        # 3. Admit new sequences
        while self._waiting and len(self._running) < self._max_batch:
            # Sort by priority (higher first), then by arrival order
            self._waiting.sort(key=lambda s: -s.priority)
            candidate = self._waiting[0]
            needed = self._pages_needed(candidate)

            if self._free_pages >= needed:
                self._free_pages -= needed
                candidate.status = SequenceStatus.RUNNING
                self._waiting.pop(0)
                self._running.append(candidate)
                logger.info("Admitted seq=%s needed=%d pages free=%d", candidate.seq_id, needed, self._free_pages)
            else:
                # No space — try to preempt lowest-priority running sequence
                if self._running:
                    victim = min(self._running, key=lambda s: s.priority)
                    if victim.priority < candidate.priority:
                        self._preempt(victim)
                        continue
                break  # Can't admit more

        return list(self._running)

    def _preempt(self, seq: Sequence) -> None:
        """Preempt a running sequence: swap its KV cache to CPU."""
        seq.preemption_count += 1
        seq.status = SequenceStatus.SWAPPED
        freed = len(seq.pages)
        self._free_pages += freed
        self._running.remove(seq)
        self._swapped.append(seq)
        logger.warning(
            "Preempted seq=%s (priority=%d) freed=%d pages preemption_count=%d",
            seq.seq_id, seq.priority, freed, seq.preemption_count,
        )

    def add_request(self, seq: Sequence) -> None:
        self._waiting.append(seq)

    def mark_token_generated(self, seq_id: str) -> None:
        for seq in self._running:
            if seq.seq_id == seq_id:
                seq.generated_tokens += 1
                # Allocate new page if current pages are full
                total_tokens = seq.prompt_tokens + seq.generated_tokens
                pages_needed = (total_tokens + self.PAGE_SIZE - 1) // self.PAGE_SIZE
                while len(seq.pages) < pages_needed:
                    if self._free_pages > 0:
                        page = KVCachePage(page_id=len(seq.pages), token_count=self.PAGE_SIZE)
                        seq.pages.append(page)
                        self._free_pages -= 1
                    else:
                        # Must preempt to get a page
                        victims = [s for s in self._running if s.seq_id != seq_id]
                        if victims:
                            self._preempt(min(victims, key=lambda s: s.priority))
                break
```

---

## Exercise 3: Custom HPA Metrics for LLM Serving Pods

### Problem
Standard Kubernetes HPA metrics (CPU, memory) are poor proxies for LLM serving load. Design custom HPA metrics that actually reflect when to scale.

### Answer

**Why CPU/memory fail for LLM serving:**
- A vLLM pod at 100% GPU utilization shows only 15% CPU — CPU HPA won't scale
- Memory is nearly constant once KV cache is allocated — memory HPA won't trigger
- The real bottleneck is: **KV cache page exhaustion** and **queue depth**

**Correct scaling signals:**

| Metric | Scale-out threshold | Scale-in threshold | Source |
|---|---|---|---|
| `llm_kv_cache_utilization` | > 0.80 | < 0.40 | vLLM `/metrics` Prometheus |
| `llm_queue_depth_p95_seconds` | > 1.0s | < 0.1s | App Prometheus |
| `llm_gpu_utilization` | > 0.85 | < 0.30 | DCGM exporter |
| `llm_requests_per_second` | > 80/pod | < 10/pod | App Prometheus |
| `llm_active_sequences` | > 200 | < 50 | vLLM `/metrics` |

```python
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScalingRecommendation:
    current_replicas: int
    recommended_replicas: int
    reason: str
    urgent: bool  # If True, skip stabilization window


@dataclass
class ServingMetrics:
    kv_cache_utilization: float          # 0.0–1.0
    queue_depth_p95_seconds: float       # seconds requests wait in queue
    gpu_utilization: float               # 0.0–1.0
    requests_per_second: float
    active_sequences: int
    preemption_rate: float               # sequences preempted per second


class LLMServingAutoscaler:
    """
    Custom autoscaling logic that reads LLM-specific metrics.
    In production, this becomes a KEDA ScaledObject or custom metrics API server.
    """

    MIN_REPLICAS = 2
    MAX_REPLICAS = 20
    SCALE_UP_COOLDOWN_SECONDS = 60
    SCALE_DOWN_COOLDOWN_SECONDS = 300  # Conservative: scale down slowly

    def __init__(self) -> None:
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0

    def recommend_scaling(
        self,
        metrics: ServingMetrics,
        current_replicas: int,
    ) -> ScalingRecommendation:
        now = time.monotonic()
        reasons: list[str] = []
        scale_out_votes = 0
        scale_in_votes = 0
        urgent = False

        # Primary signal: KV cache pressure
        if metrics.kv_cache_utilization > 0.90:
            scale_out_votes += 3
            urgent = True
            reasons.append(f"kv_cache_util={metrics.kv_cache_utilization:.0%} (CRITICAL)")
        elif metrics.kv_cache_utilization > 0.80:
            scale_out_votes += 2
            reasons.append(f"kv_cache_util={metrics.kv_cache_utilization:.0%}")
        elif metrics.kv_cache_utilization < 0.40:
            scale_in_votes += 1

        # Queue depth
        if metrics.queue_depth_p95_seconds > 2.0:
            scale_out_votes += 2
            urgent = True
            reasons.append(f"queue_p95={metrics.queue_depth_p95_seconds:.1f}s")
        elif metrics.queue_depth_p95_seconds > 1.0:
            scale_out_votes += 1
            reasons.append(f"queue_p95={metrics.queue_depth_p95_seconds:.1f}s")
        elif metrics.queue_depth_p95_seconds < 0.1:
            scale_in_votes += 1

        # GPU utilization
        if metrics.gpu_utilization > 0.90:
            scale_out_votes += 2
            reasons.append(f"gpu_util={metrics.gpu_utilization:.0%}")
        elif metrics.gpu_utilization < 0.30:
            scale_in_votes += 2

        # Preemption rate indicates KV cache thrashing
        if metrics.preemption_rate > 5.0:
            scale_out_votes += 1
            reasons.append(f"preemption_rate={metrics.preemption_rate:.1f}/s")

        # Compute recommended replicas
        if scale_out_votes >= 2:
            # Scale proportional to pressure
            factor = 1.5 if urgent else 1.25
            recommended = min(self.MAX_REPLICAS, int(current_replicas * factor) + 1)
            action = "scale_out"
        elif scale_in_votes >= 2 and scale_out_votes == 0:
            recommended = max(self.MIN_REPLICAS, int(current_replicas * 0.75))
            action = "scale_in"
        else:
            recommended = current_replicas
            action = "hold"

        # Apply cooldown
        if action == "scale_out" and (now - self._last_scale_up) < self.SCALE_UP_COOLDOWN_SECONDS and not urgent:
            logger.info("Scale-out suppressed by cooldown")
            recommended = current_replicas
        elif action == "scale_in" and (now - self._last_scale_down) < self.SCALE_DOWN_COOLDOWN_SECONDS:
            logger.info("Scale-in suppressed by cooldown")
            recommended = current_replicas

        if recommended > current_replicas:
            self._last_scale_up = now
        elif recommended < current_replicas:
            self._last_scale_down = now

        return ScalingRecommendation(
            current_replicas=current_replicas,
            recommended_replicas=recommended,
            reason="; ".join(reasons) if reasons else "metrics within bounds",
            urgent=urgent,
        )
```

**KEDA ScaledObject configuration:**
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-serving-scaler
spec:
  scaleTargetRef:
    name: llm-serving-deployment
  minReplicaCount: 2
  maxReplicaCount: 20
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: llm_kv_cache_utilization
        threshold: "0.80"
        query: avg(llm_kv_cache_utilization{service="llm-serving"})
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: llm_queue_depth_p95_seconds
        threshold: "1.0"
        query: histogram_quantile(0.95, sum(rate(llm_queue_duration_seconds_bucket[2m])) by (le))
```

---

## Exercise 4: Cost Anomaly Detection for Per-Tenant Spikes

### Problem
Design alerting for per-tenant cost spikes. What signals to use, what baselines, how to avoid alert fatigue.

### Answer

**Challenge**: LLM cost spikes are often legitimate (user runs a big batch job) or malicious (prompt injection loop). Need to distinguish.

**Anomaly detection approach: Z-score over rolling 7-day baseline with rate-of-change check**

A one-hour cost spike that's 3× the hourly average → alert.
A monthly cost that's 2× but rose gradually → informational warning, not page.

```python
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class TenantCostWindow:
    tenant_id: str
    hourly_costs_usd: list[float]        # Last 168 hours (7 days)
    daily_costs_usd: list[float]         # Last 30 days
    monthly_budget_usd: float
    burst_allowance_multiplier: float    # e.g., 3.0 = 3× baseline allowed for 1h


@dataclass
class CostAnomaly:
    tenant_id: str
    anomaly_type: str       # "hourly_spike" | "daily_spike" | "budget_breach" | "rate_of_change"
    current_value: float
    baseline_value: float
    z_score: float
    severity: str           # "warning" | "critical"
    recommended_action: str


def compute_z_score(values: list[float], current: float) -> tuple[float, float]:
    """Returns (z_score, mean). Returns (0, 0) if insufficient data."""
    if len(values) < 24:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0, 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return 0.0, mean
    return (current - mean) / std, mean


def detect_cost_anomalies(
    window: TenantCostWindow,
    current_hourly_usd: float,
    current_daily_usd: float,
    month_to_date_usd: float,
) -> list[CostAnomaly]:
    anomalies: list[CostAnomaly] = []

    # 1. Hourly spike check
    hourly_z, hourly_mean = compute_z_score(window.hourly_costs_usd, current_hourly_usd)
    if hourly_z > 4.0:
        anomalies.append(CostAnomaly(
            tenant_id=window.tenant_id,
            anomaly_type="hourly_spike",
            current_value=current_hourly_usd,
            baseline_value=hourly_mean,
            z_score=hourly_z,
            severity="critical",
            recommended_action="Throttle tenant to baseline rate; alert tenant admin",
        ))
    elif hourly_z > 2.5 and current_hourly_usd > hourly_mean * window.burst_allowance_multiplier:
        anomalies.append(CostAnomaly(
            tenant_id=window.tenant_id,
            anomaly_type="hourly_spike",
            current_value=current_hourly_usd,
            baseline_value=hourly_mean,
            z_score=hourly_z,
            severity="warning",
            recommended_action="Monitor; notify tenant if sustained for 2+ hours",
        ))

    # 2. Budget breach check
    remaining_budget = window.monthly_budget_usd - month_to_date_usd
    days_elapsed = max(1, len(window.daily_costs_usd))
    days_in_month = 30
    expected_spend = (month_to_date_usd / days_elapsed) * days_in_month

    if month_to_date_usd > window.monthly_budget_usd:
        anomalies.append(CostAnomaly(
            tenant_id=window.tenant_id,
            anomaly_type="budget_breach",
            current_value=month_to_date_usd,
            baseline_value=window.monthly_budget_usd,
            z_score=0.0,
            severity="critical",
            recommended_action="Enforce hard cap; block new requests; notify tenant",
        ))
    elif expected_spend > window.monthly_budget_usd * 1.2:
        anomalies.append(CostAnomaly(
            tenant_id=window.tenant_id,
            anomaly_type="budget_breach",
            current_value=expected_spend,
            baseline_value=window.monthly_budget_usd,
            z_score=0.0,
            severity="warning",
            recommended_action=f"On pace to exceed budget by {expected_spend/window.monthly_budget_usd:.0%}; alert tenant",
        ))

    # 3. Rate of change: sudden acceleration in daily spend
    if len(window.daily_costs_usd) >= 7:
        recent_avg = sum(window.daily_costs_usd[-3:]) / 3
        prior_avg = sum(window.daily_costs_usd[-7:-3]) / 4
        if prior_avg > 0 and recent_avg > prior_avg * 2.5:
            anomalies.append(CostAnomaly(
                tenant_id=window.tenant_id,
                anomaly_type="rate_of_change",
                current_value=recent_avg,
                baseline_value=prior_avg,
                z_score=0.0,
                severity="warning",
                recommended_action="3-day avg 2.5× 4-day prior avg; investigate usage pattern change",
            ))

    for a in anomalies:
        logger.warning(
            "COST ANOMALY tenant=%s type=%s severity=%s current=$%.2f baseline=$%.2f z=%.1f",
            a.tenant_id, a.anomaly_type, a.severity, a.current_value, a.baseline_value, a.z_score,
        )

    return anomalies


class CostAnomalyNotifier:
    """Send anomaly alerts. Use LLM to generate natural-language explanation for tenant admins."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_tenant_alert(self, anomaly: CostAnomaly) -> str:
        """Generate a human-readable alert message for the tenant admin."""
        response = self._client.messages.create(
            model="claude-haiku-5-5",  # Use cheapest model for alert generation
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a brief, professional alert email (2-3 sentences) for a tenant admin "
                    f"about a cost anomaly. "
                    f"Anomaly type: {anomaly.anomaly_type}. "
                    f"Current spend: ${anomaly.current_value:.2f}. "
                    f"Normal baseline: ${anomaly.baseline_value:.2f}. "
                    f"Recommended action: {anomaly.recommended_action}. "
                    f"Be factual and actionable, not alarming."
                ),
            }],
        )
        return response.content[0].text
```

**Alert routing:**
- `critical` → PagerDuty (platform ops) + email to tenant admin
- `warning` → Slack #cost-alerts (platform ops) + dashboard widget
- Suppress duplicate alerts: same `(tenant_id, anomaly_type)` within 1-hour window
- Auto-resolve: if anomaly clears for 2 consecutive hours, send resolution notification

**Anti-patterns to avoid:**
- Don't alert on absolute cost — small tenants' $5 spikes are meaningless
- Don't alert on monthly total until >70% of month elapsed (too early = false positive)
- Don't use daily averages for hourly spike detection — diurnal patterns create constant noise
