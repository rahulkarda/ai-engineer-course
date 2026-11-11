# Day 1 Solutions: Production RAG Platform at Scale

## Exercise 1: Chunking Strategy for Legal Document Corpus

### Problem
Legal documents have variable-length clauses — some clauses are 2 sentences, others span 3 pages. Standard fixed-size chunking destroys semantic units (splits a clause mid-sentence) and creates retrieval noise (chunks that start in the middle of a clause lack context).

### Answer

**Strategy: Hierarchical semantic chunking with clause-boundary detection**

Legal documents have natural boundary signals. Use a two-pass approach:

**Pass 1 — Structural segmentation**
Split on hard structural boundaries first:
- Section headings (`§`, numbered headings like `1.`, `1.1`, `Article I`)
- Paragraph breaks followed by a capitalized clause label
- "WHEREAS", "NOW THEREFORE", "IN WITNESS WHEREOF" markers

**Pass 2 — Adaptive size enforcement**
After structural split, apply size constraints:
- If chunk < 100 tokens: merge with adjacent chunk (orphan clause)
- If chunk > 800 tokens: re-split on sentence boundaries, preserving the first sentence as a "header" that gets prepended to each sub-chunk (context carry-forward)
- If chunk is 100–800 tokens: keep as-is

**Why not fixed-size sliding window?**
Sliding window with overlap is fine for homogeneous prose (news articles, wiki). Legal clauses are semantic atoms — splitting "The Licensee shall not" from "sublicense, sell, or transfer the rights granted herein" makes both chunks useless for retrieval.

**Why not purely semantic (sentence embeddings)?**
Pure semantic splitting (e.g., splitting when cosine similarity drops) fails on legal boilerplate because adjacent boilerplate sentences are highly similar (high cosine → no split), even when they belong to distinct legal obligations.

```python
import re
import logging
from dataclasses import dataclass
from typing import Generator

logger = logging.getLogger(__name__)

# Legal boundary patterns in priority order
HARD_BOUNDARIES = [
    re.compile(r'^\s*(§\s*\d+|ARTICLE\s+[IVXLCDM]+|Section\s+\d+)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*(WHEREAS|NOW[,\s]+THEREFORE|IN WITNESS WHEREOF)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*\d+\.\d*\s+[A-Z]', re.MULTILINE),  # Numbered clauses
]

SOFT_BOUNDARIES = re.compile(r'\.\s+(?=[A-Z])')  # Sentence boundary fallback


@dataclass
class Chunk:
    text: str
    source_doc_id: str
    clause_ref: str          # e.g., "Section 3.2"
    char_start: int
    char_end: int
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return len(text) // 4  # rough: 1 token ≈ 4 chars for English legal text


def split_on_hard_boundaries(text: str) -> list[tuple[str, str]]:
    """Returns list of (heading_ref, clause_text) tuples."""
    segments: list[tuple[str, str]] = []
    positions = []

    for pattern in HARD_BOUNDARIES:
        for m in pattern.finditer(text):
            positions.append((m.start(), m.group().strip()))

    positions.sort(key=lambda x: x[0])
    if not positions:
        return [("", text)]

    for i, (pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        segments.append((heading, text[pos:end]))

    return segments


def adaptive_chunk(
    doc_id: str,
    text: str,
    min_tokens: int = 100,
    max_tokens: int = 800,
) -> Generator[Chunk, None, None]:
    """Hierarchical chunker for legal documents."""
    segments = split_on_hard_boundaries(text)
    char_offset = 0

    for clause_ref, segment_text in segments:
        tok = estimate_tokens(segment_text)

        if tok < min_tokens:
            # Too small — yield as-is, caller can merge
            logger.debug("Small clause %s (%d tokens), yielding as orphan", clause_ref, tok)
            yield Chunk(
                text=segment_text,
                source_doc_id=doc_id,
                clause_ref=clause_ref,
                char_start=char_offset,
                char_end=char_offset + len(segment_text),
                token_estimate=tok,
            )

        elif tok <= max_tokens:
            yield Chunk(
                text=segment_text,
                source_doc_id=doc_id,
                clause_ref=clause_ref,
                char_start=char_offset,
                char_end=char_offset + len(segment_text),
                token_estimate=tok,
            )

        else:
            # Too large — split on sentence boundaries, carry forward first sentence
            sentences = SOFT_BOUNDARIES.split(segment_text)
            header = sentences[0] + ". " if sentences else ""
            buffer = header
            sub_start = char_offset

            for sent in sentences[1:]:
                candidate = buffer + sent + ". "
                if estimate_tokens(candidate) > max_tokens:
                    yield Chunk(
                        text=buffer.strip(),
                        source_doc_id=doc_id,
                        clause_ref=clause_ref,
                        char_start=sub_start,
                        char_end=sub_start + len(buffer),
                        token_estimate=estimate_tokens(buffer),
                    )
                    sub_start += len(buffer)
                    buffer = header + sent + ". "
                else:
                    buffer = candidate

            if buffer.strip():
                yield Chunk(
                    text=buffer.strip(),
                    source_doc_id=doc_id,
                    clause_ref=clause_ref,
                    char_start=sub_start,
                    char_end=sub_start + len(buffer),
                    token_estimate=estimate_tokens(buffer),
                )

        char_offset += len(segment_text)
```

**Key metadata to store per chunk:**
- `clause_ref` (section number) — enables clause-level citation
- `char_start/end` — enables highlight-in-source-doc UX
- Parent document ID + version hash — enables staleness detection when documents are updated

---

## Exercise 2: Vector Store Sharding for 100M+ Documents with Multi-Tenant Isolation

### Problem
100M+ documents. Multiple tenants. Need: (1) query latency < 100ms at p95, (2) tenant A's data must not be visible to tenant B, (3) one large tenant must not starve small tenants.

### Answer

**Decision tree: namespace vs schema vs cluster isolation**

| Concern | Solution |
|---|---|
| Latency at 100M docs | Shard by tenant prefix into separate collections/namespaces |
| Strict data isolation (compliance) | One Qdrant/Weaviate collection per tenant |
| Resource fairness | Dedicated replicas for large tenants, shared pool for small |

**Recommended architecture: Tiered sharding**

```
Tier 1 (Large tenants, >1M docs):  dedicated Qdrant collection + dedicated replicas
Tier 2 (Mid tenants, 10K–1M docs): shared cluster, isolated namespaces/collections
Tier 3 (Small tenants, <10K docs):  shared collection, filtered by tenant_id metadata
```

**Why per-collection for large tenants?**
At 100M total documents, a single collection with metadata filter (`where tenant_id = X`) still scans the full HNSW graph before filtering — O(log N) of the total index, not just tenant X's subset. With dedicated collections, query touches only tenant X's vectors: O(log n_x).

**Sharding scheme for Tier 1 dedicated collections:**

```python
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    LARGE = "large"    # >1M docs
    MID = "mid"        # 10K–1M docs
    SMALL = "small"    # <10K docs


@dataclass
class TenantConfig:
    tenant_id: str
    tier: TenantTier
    doc_count_estimate: int
    dedicated_replicas: int = 0
    shard_count: int = 1


def get_collection_name(tenant_id: str, tier: TenantTier) -> str:
    """Deterministic collection name. Tier 3 shares a collection."""
    if tier == TenantTier.SMALL:
        return "shared_small_tenants"
    # Sanitize tenant_id for use as collection name
    safe_id = tenant_id.replace("-", "_").lower()
    return f"tenant_{safe_id}"


def compute_shard_count(doc_count: int, vectors_per_shard: int = 5_000_000) -> int:
    """Each shard holds ~5M vectors for HNSW performance at <50ms p95."""
    return max(1, (doc_count + vectors_per_shard - 1) // vectors_per_shard)


def assign_tenant_tier(doc_count: int) -> TenantTier:
    if doc_count > 1_000_000:
        return TenantTier.LARGE
    if doc_count > 10_000:
        return TenantTier.MID
    return TenantTier.SMALL


class ShardRouter:
    """Routes upsert/query to correct collection+shard."""

    def __init__(self, tenant_configs: list[TenantConfig]) -> None:
        self._configs: dict[str, TenantConfig] = {
            t.tenant_id: t for t in tenant_configs
        }

    def get_query_collection(self, tenant_id: str) -> tuple[str, Optional[dict]]:
        """Returns (collection_name, metadata_filter_or_None)."""
        cfg = self._configs.get(tenant_id)
        if cfg is None:
            raise KeyError(f"Unknown tenant: {tenant_id}")

        collection = get_collection_name(tenant_id, cfg.tier)
        if cfg.tier == TenantTier.SMALL:
            # Must filter by tenant_id in shared collection
            return collection, {"tenant_id": tenant_id}
        # Dedicated collection — no filter needed (isolation by construction)
        return collection, None

    def get_upsert_shard(self, tenant_id: str, doc_id: str) -> tuple[str, int]:
        """Returns (collection_name, shard_key) for writes."""
        cfg = self._configs[tenant_id]
        collection = get_collection_name(tenant_id, cfg.tier)
        # Consistent hash for even distribution across shards
        shard_key = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % cfg.shard_count
        return collection, shard_key
```

**HNSW tuning at scale:**
- `ef_construction=128`, `m=16` for the index build (standard)
- Increase `ef` at query time to 64–128 for high-recall use cases (legal = high recall requirement)
- Enable quantization (scalar int8) to reduce memory: 100M × 1536-dim × 4B = 614GB raw; with int8 quantization = 153GB

**Multi-tenant replica placement:**
- Large tenant replicas: pin to dedicated node pool (Kubernetes node affinity)
- Shared pool (Tier 2/3): 3 replicas, spread across AZs
- Read path: always route to nearest replica (latency); write path: leader only

---

## Exercise 3: Hybrid Search Alpha Tuning

### Problem
Hybrid search combines dense (embedding cosine) and sparse (BM25/TF-IDF) retrieval with a weight `alpha ∈ [0,1]` where 1.0 = pure dense, 0.0 = pure sparse. When should alpha be high vs low?

### Answer

**Decision rule: alpha is driven by query type, not document type**

| Query type | Alpha | Rationale |
|---|---|---|
| Keyword/exact match ("Force Majeure clause section 8.3") | 0.1–0.3 | Sparse wins on exact terms; dense will hallucinate similar-sounding clauses |
| Semantic/conceptual ("what happens if supplier can't deliver due to natural disaster") | 0.7–0.9 | Dense wins; sparse misses paraphrases |
| Entity lookup ("Acme Corp liability cap") | 0.3–0.5 | Hybrid balanced; entity name benefits sparse, legal concept benefits dense |
| Short 1–3 word queries ("termination notice") | 0.4–0.6 | Sparse handles lexical precision; dense handles meaning |
| Long natural language questions | 0.75+ | Dense handles intent; sparse adds recall on key nouns |

**Practical alpha tuning procedure:**

```python
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str


@dataclass
class AlphaTuningConfig:
    alpha: float
    precision_at_5: float
    recall_at_10: float
    mrr: float


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    alpha: float,
    k: int = 60,
) -> list[SearchResult]:
    """
    RRF-based hybrid fusion.
    alpha=1.0 → pure dense, alpha=0.0 → pure sparse.
    k=60 is the standard RRF constant from the 2009 Cormack paper.
    """
    scores: dict[str, float] = {}

    for rank, result in enumerate(dense_results):
        scores[result.doc_id] = scores.get(result.doc_id, 0.0)
        scores[result.doc_id] += alpha * (1.0 / (k + rank + 1))

    for rank, result in enumerate(sparse_results):
        scores[result.doc_id] = scores.get(result.doc_id, 0.0)
        scores[result.doc_id] += (1.0 - alpha) * (1.0 / (k + rank + 1))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Merge text from either list
    text_map = {r.doc_id: r.text for r in dense_results + sparse_results}
    return [
        SearchResult(doc_id=doc_id, score=scores[doc_id], text=text_map.get(doc_id, ""))
        for doc_id in sorted_ids
    ]


def classify_query_alpha(query: str) -> float:
    """
    Heuristic alpha classifier. In production, replace with a
    trained lightweight classifier (logistic regression on query features).
    """
    query_lower = query.lower()
    tokens = query_lower.split()
    n_tokens = len(tokens)

    # Signals for high sparse weight (low alpha)
    exact_signals = [
        any(c.isdigit() for c in query),          # Contains numbers → likely exact
        "section" in query_lower,
        "clause" in query_lower,
        "article" in query_lower,
        n_tokens <= 3,                              # Short queries benefit from BM25
    ]

    # Signals for high dense weight (high alpha)
    semantic_signals = [
        n_tokens >= 10,                             # Long questions
        "?" in query,                               # Question form
        "what" in tokens or "how" in tokens or "why" in tokens,
        "mean" in query_lower or "explain" in query_lower,
    ]

    sparse_score = sum(exact_signals)
    dense_score = sum(semantic_signals)

    if sparse_score > dense_score:
        alpha = max(0.2, 0.5 - 0.1 * sparse_score)
    elif dense_score > sparse_score:
        alpha = min(0.9, 0.5 + 0.1 * dense_score)
    else:
        alpha = 0.5

    logger.debug(
        "Query=%r sparse_signals=%d dense_signals=%d alpha=%.2f",
        query[:50], sparse_score, dense_score, alpha,
    )
    return alpha


def tune_alpha_offline(
    eval_queries: list[dict],  # [{"query": str, "relevant_ids": list[str]}]
    search_fn: Callable[[str, float], list[SearchResult]],
    alpha_candidates: list[float] | None = None,
) -> float:
    """Grid search for best alpha on eval set. Returns optimal alpha."""
    if alpha_candidates is None:
        alpha_candidates = [round(a, 1) for a in np.arange(0.0, 1.1, 0.1)]

    results: list[AlphaTuningConfig] = []

    for alpha in alpha_candidates:
        precisions, recalls, rrs = [], [], []

        for item in eval_queries:
            query = item["query"]
            relevant = set(item["relevant_ids"])
            retrieved = search_fn(query, alpha)

            top5_ids = [r.doc_id for r in retrieved[:5]]
            top10_ids = [r.doc_id for r in retrieved[:10]]

            p5 = sum(1 for i in top5_ids if i in relevant) / 5
            r10 = sum(1 for i in top10_ids if i in relevant) / max(len(relevant), 1)

            rank = next(
                (i + 1 for i, r in enumerate(retrieved) if r.doc_id in relevant),
                None,
            )
            rr = 1.0 / rank if rank else 0.0

            precisions.append(p5)
            recalls.append(r10)
            rrs.append(rr)

        cfg = AlphaTuningConfig(
            alpha=alpha,
            precision_at_5=float(np.mean(precisions)),
            recall_at_10=float(np.mean(recalls)),
            mrr=float(np.mean(rrs)),
        )
        results.append(cfg)
        logger.info("alpha=%.1f P@5=%.3f R@10=%.3f MRR=%.3f", alpha, cfg.precision_at_5, cfg.recall_at_10, cfg.mrr)

    # Optimize for MRR (best balance of precision and rank quality)
    best = max(results, key=lambda c: c.mrr)
    logger.info("Best alpha=%.1f MRR=%.3f", best.alpha, best.mrr)
    return best.alpha
```

**Key insight:** don't use a single static alpha. Use a query classifier that routes to different alpha values per query type. For legal RAG, a good default split:
- Citation queries (contain numbers/section refs): alpha=0.2
- Obligation queries (what must X do): alpha=0.7
- General Q&A: alpha=0.5

---

## Exercise 4: Eval Gate Design

### Problem
Design an automated eval gate that runs before deploying a new RAG pipeline version. What metrics to use, what thresholds, what to do on failure.

### Answer

**Gate architecture: three-layer funnel**

```
Layer 1 (Hard gate): Regression on golden set — if any metric drops >5% relative, block.
Layer 2 (Soft gate): New metric targets — must meet minimum bars for new functionality.
Layer 3 (Shadow gate): 5% shadow traffic comparison against prod for 24h — alert on drift.
```

**Metrics and thresholds:**

| Metric | Target | Hard block if | Measurement |
|---|---|---|---|
| Faithfulness (LLM-as-judge) | ≥0.85 | <0.80 | Judge: "Is answer supported by context?" |
| Answer Relevance | ≥0.80 | <0.75 | Judge: "Does answer address the question?" |
| Context Recall | ≥0.75 | <0.70 | # relevant chunks retrieved / total relevant |
| Context Precision | ≥0.70 | <0.65 | # relevant chunks in top-K / K |
| Latency p95 | ≤2000ms | >3000ms | End-to-end wall time |
| Latency p99 | ≤5000ms | >8000ms | End-to-end wall time |
| Hallucination rate | ≤0.05 | >0.10 | Judge: claims not in source docs |

```python
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class EvalResult:
    metric_name: str
    value: float
    threshold: float
    hard_block_threshold: float
    passed: bool
    hard_blocked: bool


@dataclass
class GateDecision:
    deploy: bool
    results: list[EvalResult] = field(default_factory=list)
    blocking_metrics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


METRIC_THRESHOLDS = {
    "faithfulness":       {"target": 0.85, "hard_block": 0.80},
    "answer_relevance":   {"target": 0.80, "hard_block": 0.75},
    "context_recall":     {"target": 0.75, "hard_block": 0.70},
    "context_precision":  {"target": 0.70, "hard_block": 0.65},
    "hallucination_rate": {"target": 0.05, "hard_block": 0.10},  # lower is better
}

FAITHFULNESS_PROMPT = """You are an expert evaluator for a legal RAG system.

Context retrieved:
{context}

Question asked:
{question}

Answer generated:
{answer}

Score the answer on FAITHFULNESS (0.0–1.0):
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Most claims supported, minor extrapolation
- 0.0: Answer contains claims not in context (hallucination)

Respond with JSON only: {{"score": <float>, "reason": "<one sentence>"}}"""


def judge_faithfulness(
    question: str,
    context: str,
    answer: str,
) -> float:
    """LLM-as-judge for faithfulness. Uses claude-haiku-5-5 for cost efficiency at scale."""
    import json

    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": FAITHFULNESS_PROMPT.format(
                context=context[:3000],
                question=question,
                answer=answer,
            ),
        }],
    )

    try:
        result = json.loads(response.content[0].text)
        return float(result["score"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Judge parse error: %s", e)
        return 0.0


def run_eval_gate(
    eval_dataset: list[dict],  # [{"question", "context", "answer", "ground_truth"}]
    latency_p95_ms: float,
    latency_p99_ms: float,
    baseline_metrics: Optional[dict[str, float]] = None,
) -> GateDecision:
    """
    Run the full eval gate. Returns deploy=True only if all hard blocks pass.

    baseline_metrics: production metrics from current deployed version.
                     If provided, also checks for >5% regression.
    """
    # Compute metrics over eval dataset
    faithfulness_scores = []
    for item in eval_dataset:
        score = judge_faithfulness(
            question=item["question"],
            context=item["context"],
            answer=item["answer"],
        )
        faithfulness_scores.append(score)

    computed = {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        "latency_p95_ms": latency_p95_ms,
        "latency_p99_ms": latency_p99_ms,
    }

    results: list[EvalResult] = []
    blocking_metrics: list[str] = []
    warnings: list[str] = []

    for metric, thresholds in METRIC_THRESHOLDS.items():
        value = computed.get(metric, 0.0)
        target = thresholds["target"]
        hard_block = thresholds["hard_block"]

        # For rate metrics (lower is better), invert comparison
        is_rate = metric.endswith("_rate")
        if is_rate:
            passed = value <= target
            hard_blocked = value > hard_block
        else:
            passed = value >= target
            hard_blocked = value < hard_block

        result = EvalResult(
            metric_name=metric,
            value=value,
            threshold=target,
            hard_block_threshold=hard_block,
            passed=passed,
            hard_blocked=hard_blocked,
        )
        results.append(result)

        if hard_blocked:
            blocking_metrics.append(metric)
            logger.error("HARD BLOCK: %s=%.3f below threshold=%.3f", metric, value, hard_block)
        elif not passed:
            warnings.append(f"{metric}={value:.3f} below target={target:.3f}")
            logger.warning("SOFT FAIL: %s=%.3f below target=%.3f", metric, value, target)

        # Regression check against baseline
        if baseline_metrics and metric in baseline_metrics:
            baseline_val = baseline_metrics[metric]
            if baseline_val > 0:
                regression = (baseline_val - value) / baseline_val
                if regression > 0.05:  # >5% relative regression
                    blocking_metrics.append(f"{metric}_regression")
                    logger.error(
                        "REGRESSION: %s dropped %.1f%% from baseline %.3f to %.3f",
                        metric, regression * 100, baseline_val, value,
                    )

    deploy = len(blocking_metrics) == 0
    return GateDecision(deploy=deploy, results=results, blocking_metrics=blocking_metrics, warnings=warnings)
```

**Gate integration in CI/CD:**

```yaml
# .github/workflows/rag-deploy.yml (pattern)
eval-gate:
  runs-on: ubuntu-latest
  steps:
    - name: Run RAG eval gate
      run: python scripts/run_eval_gate.py --dataset eval/golden_set.json
    - name: Check gate decision
      run: |
        if [ "$(cat gate_result.json | jq '.deploy')" != "true" ]; then
          echo "Eval gate FAILED. Blocking deploy."
          exit 1
        fi
```

**Thresholds rationale:**
- Faithfulness 0.80 hard block: legal RAG with <80% faithfulness is a liability (wrong legal advice)
- Context recall 0.70 hard block: if you miss 30% of relevant clauses, answers are dangerously incomplete
- Latency 3000ms p95: contracts UI timeout is typically 5s; 3s leaves buffer for network
- 5% regression rule: catches slow model degradation that stays above absolute thresholds
