# Day 5 Solutions: Eval and Monitoring

## Exercise 1: LLM-as-Judge Prompt Design for RAG Answer Quality

### Problem
Design a complete LLM-as-judge evaluation system for RAG answer quality. Specify the criteria, scoring rubric, prompt template, and how to calibrate the judge against human labels.

### Answer

**Core RAG quality dimensions (RAGAS-inspired but extended):**

| Dimension | Question the judge answers | Weight |
|---|---|---|
| Faithfulness | "Are all claims in the answer grounded in the retrieved context?" | 35% |
| Answer Relevance | "Does the answer directly address what was asked?" | 25% |
| Context Utilization | "Does the answer use the most relevant parts of the context?" | 20% |
| Completeness | "Does the answer cover all aspects of the question?" | 20% |

**Judge prompt design principles:**
1. One criterion per prompt call (avoids criterion bleed — judge averaging across dimensions)
2. Chain-of-thought before score (scratchpad reduces score variance)
3. Score on 0–4 integer scale (easier for LLM than 0.0–1.0 float; convert after)
4. Anchor examples in the prompt (calibration via few-shot)

```python
import json
import logging
import os
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Prompt templates ---

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI-generated answer
is faithfully grounded in the provided source context.

TASK: Rate the FAITHFULNESS of the answer to the context on a scale of 0-4.

SCORING RUBRIC:
4 - Excellent: Every claim in the answer is explicitly supported by the context. No hallucination.
3 - Good: Nearly all claims supported. At most one minor inference not explicitly in context.
2 - Acceptable: Most claims supported, but one or more claims are inferred or extended beyond context.
1 - Poor: Significant claims in the answer are not supported by context or contradict it.
0 - Fail: Answer is mostly or entirely not grounded in context. Major hallucination.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

INSTRUCTIONS:
1. List each factual claim in the answer.
2. For each claim, note whether it is explicitly supported, inferable, or not in the context.
3. Assign a score 0-4 based on the rubric above.
4. Output JSON only.

Output format: {{"claims": [{{"claim": "...", "supported": true/false/null}}], "score": <0-4>, "reason": "<one sentence>"}}"""


RELEVANCE_PROMPT = """You are an expert evaluator assessing whether an AI-generated answer
actually addresses the user's question.

TASK: Rate the ANSWER RELEVANCE on a scale of 0-4.

SCORING RUBRIC:
4 - Excellent: Answer directly and completely addresses the question asked. On-topic throughout.
3 - Good: Answer addresses the main question, with minor tangents or missing one aspect.
2 - Acceptable: Answer partially addresses the question; some key aspects missing or off-topic.
1 - Poor: Answer is mostly off-topic or addresses a different interpretation of the question.
0 - Fail: Answer does not address the question at all.

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

INSTRUCTIONS:
1. Identify the core information need expressed by the question.
2. Assess what fraction of the answer directly serves that need.
3. Assign a score 0-4 based on the rubric above.
4. Output JSON only.

Output format: {{"information_need": "...", "on_topic_fraction": <0.0-1.0>, "score": <0-4>, "reason": "<one sentence>"}}"""


COMPLETENESS_PROMPT = """You are an expert evaluator assessing whether an AI-generated answer
is complete relative to the available context and the question asked.

TASK: Rate the COMPLETENESS of the answer on a scale of 0-4.

SCORING RUBRIC:
4 - Excellent: Answer covers all relevant aspects present in the context. Nothing important omitted.
3 - Good: Covers most relevant aspects. At most one minor point from context is missing.
2 - Acceptable: Covers the main point but omits 1-2 relevant aspects from context.
1 - Poor: Covers only part of what the context and question require. Significant gaps.
0 - Fail: Answer is severely incomplete. Misses most of the relevant information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

INSTRUCTIONS:
1. List the key aspects the question asks about that are present in the context.
2. Check which aspects the answer covers.
3. Assign a score 0-4 based on the rubric above.
4. Output JSON only.

Output format: {{"required_aspects": ["..."], "covered_aspects": ["..."], "score": <0-4>, "reason": "<one sentence>"}}"""


# --- Judge implementation ---

@dataclass
class DimensionScore:
    dimension: str
    score: int           # 0–4
    normalized: float    # 0.0–1.0
    reason: str
    raw_response: str


@dataclass
class RAGEvalResult:
    question: str
    answer: str
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    weighted_score: float = 0.0
    passed: bool = False

    # Weights for composite score
    WEIGHTS: dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.35,
        "relevance": 0.25,
        "completeness": 0.20,
        "context_utilization": 0.20,
    })


DIMENSION_PROMPTS = {
    "faithfulness": FAITHFULNESS_PROMPT,
    "relevance": RELEVANCE_PROMPT,
    "completeness": COMPLETENESS_PROMPT,
}

WEIGHTS = {
    "faithfulness": 0.35,
    "relevance": 0.25,
    "completeness": 0.20,
}


def _call_judge(prompt: str) -> tuple[int, str]:
    """Call the judge LLM. Returns (score_0_to_4, reason)."""
    response = client.messages.create(
        model="claude-haiku-5-5",  # Fast + cheap for high-volume eval
        max_tokens=512,
        temperature=0.0,  # Deterministic for consistency
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text

    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        reason = parsed.get("reason", "")
        return score, reason
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Judge parse error: %s | raw=%r", exc, raw[:200])
        return 0, f"parse_error: {exc}"


def evaluate_rag_answer(
    question: str,
    context: str,
    answer: str,
    passing_threshold: float = 0.75,
) -> RAGEvalResult:
    """
    Evaluate a RAG answer across all dimensions.
    Returns composite weighted score (0.0–1.0).
    """
    result = RAGEvalResult(question=question, answer=answer)

    for dimension, prompt_template in DIMENSION_PROMPTS.items():
        prompt = prompt_template.format(
            context=context[:4000],
            question=question,
            answer=answer,
        )
        score_0_4, reason = _call_judge(prompt)
        normalized = score_0_4 / 4.0

        dim_score = DimensionScore(
            dimension=dimension,
            score=score_0_4,
            normalized=normalized,
            reason=reason,
            raw_response="",
        )
        result.dimension_scores.append(dim_score)
        logger.debug("Judge %s score=%d (%.2f): %s", dimension, score_0_4, normalized, reason)

    # Compute weighted composite
    total_weight = sum(WEIGHTS.values())
    weighted_sum = sum(
        ds.normalized * WEIGHTS.get(ds.dimension, 0.0)
        for ds in result.dimension_scores
    )
    result.weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    result.passed = result.weighted_score >= passing_threshold

    logger.info(
        "RAG eval: weighted_score=%.3f passed=%s question=%r",
        result.weighted_score, result.passed, question[:60],
    )
    return result


# --- Judge calibration ---

def calibrate_judge(
    human_labeled_dataset: list[dict],  # [{"question", "context", "answer", "human_score": 0-4}]
    dimension: str = "faithfulness",
) -> dict:
    """
    Measure judge agreement with human labels.
    Compute: MAE, correlation, accuracy within ±1.
    """
    judge_scores = []
    human_scores = []
    prompt_template = DIMENSION_PROMPTS[dimension]

    for item in human_labeled_dataset:
        prompt = prompt_template.format(
            context=item["context"][:4000],
            question=item["question"],
            answer=item["answer"],
        )
        score, _ = _call_judge(prompt)
        judge_scores.append(score)
        human_scores.append(item["human_score"])

    if not judge_scores:
        return {}

    n = len(judge_scores)
    mae = mean(abs(j - h) for j, h in zip(judge_scores, human_scores))
    exact_match = sum(1 for j, h in zip(judge_scores, human_scores) if j == h) / n
    within_one = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= 1) / n

    # Pearson correlation
    j_mean = mean(judge_scores)
    h_mean = mean(human_scores)
    j_std = stdev(judge_scores) if len(judge_scores) > 1 else 1.0
    h_std = stdev(human_scores) if len(human_scores) > 1 else 1.0
    correlation = (
        mean((j - j_mean) * (h - h_mean) for j, h in zip(judge_scores, human_scores))
        / (j_std * h_std)
        if j_std > 0 and h_std > 0 else 0.0
    )

    report = {
        "dimension": dimension,
        "n_samples": n,
        "mae": mae,
        "exact_match_rate": exact_match,
        "within_one_rate": within_one,
        "pearson_correlation": correlation,
        "acceptable": within_one >= 0.85 and mae <= 0.5,  # Calibration acceptance criteria
    }
    logger.info("Judge calibration: %s", report)
    return report
```

---

## Exercise 2: Embedding Drift Detection

### Problem
Over time, the distribution of documents and queries in your RAG system shifts. Implement drift detection that alerts when the embedding distribution has shifted enough to degrade retrieval quality.

### Answer

**What causes embedding drift:**
1. **Query drift**: users start asking different kinds of questions (topic shift)
2. **Document drift**: new documents have different vocabulary/topics than indexed corpus
3. **Model drift**: embedding model updated — all embeddings now in a different space

**Detection methods (in order of complexity):**

| Method | What it detects | Computational cost |
|---|---|---|
| Mean cosine shift | Global centroid movement | Low |
| Distribution percentile shift | Shape changes | Low |
| MMD (Maximum Mean Discrepancy) | Full distribution shift | Medium |
| Coverage gap | Queries not covered by corpus | Low |

```python
import logging
import os
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

import anthropic
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingWindow:
    """Rolling window of embeddings for drift monitoring."""
    window_name: str
    embeddings: list[list[float]]  # Shape: [n_samples, dim]
    timestamps: list[float]
    max_window_size: int = 10_000

    def add(self, embedding: list[float]) -> None:
        self.embeddings.append(embedding)
        self.timestamps.append(time.time())
        if len(self.embeddings) > self.max_window_size:
            self.embeddings.pop(0)
            self.timestamps.pop(0)

    def as_array(self) -> np.ndarray:
        return np.array(self.embeddings, dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.embeddings)


@dataclass
class DriftAlert:
    metric: str
    value: float
    baseline_value: float
    threshold: float
    severity: str  # "warning" | "critical"
    recommendation: str


def cosine_centroid_shift(
    baseline: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Compute shift in centroid direction between two embedding windows.
    Returns 1 - cosine_similarity(centroid_baseline, centroid_current).
    0 = identical, 2 = opposite direction (extreme drift).
    """
    baseline_centroid = baseline.mean(axis=0)
    current_centroid = current.mean(axis=0)

    norm_b = np.linalg.norm(baseline_centroid)
    norm_c = np.linalg.norm(current_centroid)

    if norm_b == 0 or norm_c == 0:
        return 0.0

    cosine_sim = np.dot(baseline_centroid, current_centroid) / (norm_b * norm_c)
    return float(1.0 - cosine_sim)


def compute_mmd(
    baseline: np.ndarray,
    current: np.ndarray,
    kernel_bandwidth: float = 1.0,
    subsample: int = 500,
) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    0 = identical distributions. Typical drift threshold: 0.05.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where k is the RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*bandwidth^2))
    """
    # Subsample for efficiency
    n_b = min(subsample, len(baseline))
    n_c = min(subsample, len(current))

    idx_b = np.random.choice(len(baseline), n_b, replace=False)
    idx_c = np.random.choice(len(current), n_c, replace=False)

    b = baseline[idx_b].astype(np.float32)
    c = current[idx_c].astype(np.float32)

    def rbf_kernel(x: np.ndarray, y: np.ndarray) -> float:
        # Pairwise squared distances
        xx = np.sum(x**2, axis=1, keepdims=True)
        yy = np.sum(y**2, axis=1, keepdims=True)
        dist_sq = xx + yy.T - 2.0 * x @ y.T
        return float(np.mean(np.exp(-dist_sq / (2.0 * kernel_bandwidth**2))))

    mmd2 = rbf_kernel(b, b) + rbf_kernel(c, c) - 2.0 * rbf_kernel(b, c)
    return max(0.0, mmd2) ** 0.5  # Return MMD (not MMD^2)


def coverage_gap(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    coverage_threshold: float = 0.7,
) -> float:
    """
    Fraction of queries that have NO corpus chunk with cosine similarity > threshold.
    High value = queries not covered by corpus = retrieval will fail.
    """
    if len(query_embeddings) == 0 or len(corpus_embeddings) == 0:
        return 0.0

    # Normalize for cosine similarity
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    c_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)

    # For each query, find max cosine similarity to any corpus chunk
    # Batch to avoid OOM: process 100 queries at a time
    uncovered = 0
    batch_size = 100
    for i in range(0, len(q_norm), batch_size):
        batch = q_norm[i:i+batch_size]
        similarities = batch @ c_norm.T  # [batch, n_corpus]
        max_sims = similarities.max(axis=1)
        uncovered += int((max_sims < coverage_threshold).sum())

    return uncovered / len(query_embeddings)


class EmbeddingDriftMonitor:
    """
    Monitor embedding distributions for drift.
    Call update() periodically (e.g., every 1000 requests).
    """

    CENTROID_SHIFT_WARNING = 0.05   # 5% directional shift
    CENTROID_SHIFT_CRITICAL = 0.15
    MMD_WARNING = 0.03
    MMD_CRITICAL = 0.08
    COVERAGE_GAP_WARNING = 0.10     # 10% of queries uncovered
    COVERAGE_GAP_CRITICAL = 0.25

    def __init__(self, baseline_window: EmbeddingWindow) -> None:
        self._baseline = baseline_window
        self._corpus_embeddings: Optional[np.ndarray] = None

    def set_corpus_embeddings(self, embeddings: np.ndarray) -> None:
        """Call when corpus is re-indexed."""
        self._corpus_embeddings = embeddings

    def detect_drift(
        self,
        current_query_window: EmbeddingWindow,
    ) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []

        if self._baseline.size < 100 or current_query_window.size < 100:
            logger.debug("Insufficient samples for drift detection")
            return alerts

        baseline_arr = self._baseline.as_array()
        current_arr = current_query_window.as_array()

        # 1. Centroid shift
        shift = cosine_centroid_shift(baseline_arr, current_arr)
        logger.info("Centroid shift: %.4f", shift)

        if shift > self.CENTROID_SHIFT_CRITICAL:
            alerts.append(DriftAlert(
                metric="centroid_shift",
                value=shift,
                baseline_value=0.0,
                threshold=self.CENTROID_SHIFT_CRITICAL,
                severity="critical",
                recommendation=(
                    "Query distribution shifted significantly. "
                    "Reindex with fresh corpus or retrain embedding model. "
                    "Check if user base has changed."
                ),
            ))
        elif shift > self.CENTROID_SHIFT_WARNING:
            alerts.append(DriftAlert(
                metric="centroid_shift",
                value=shift,
                baseline_value=0.0,
                threshold=self.CENTROID_SHIFT_WARNING,
                severity="warning",
                recommendation="Monitor trend. Schedule corpus refresh if sustained.",
            ))

        # 2. MMD
        mmd = compute_mmd(baseline_arr, current_arr)
        logger.info("MMD: %.4f", mmd)

        if mmd > self.MMD_CRITICAL:
            alerts.append(DriftAlert(
                metric="mmd",
                value=mmd,
                baseline_value=0.0,
                threshold=self.MMD_CRITICAL,
                severity="critical",
                recommendation=(
                    "Full distribution shift detected. "
                    "Trigger re-evaluation of retrieval quality on recent queries."
                ),
            ))
        elif mmd > self.MMD_WARNING:
            alerts.append(DriftAlert(
                metric="mmd",
                value=mmd,
                baseline_value=0.0,
                threshold=self.MMD_WARNING,
                severity="warning",
                recommendation="Distribution shift starting. Monitor eval metrics.",
            ))

        # 3. Coverage gap
        if self._corpus_embeddings is not None:
            gap = coverage_gap(self._corpus_embeddings, current_arr)
            logger.info("Coverage gap: %.3f", gap)

            if gap > self.COVERAGE_GAP_CRITICAL:
                alerts.append(DriftAlert(
                    metric="coverage_gap",
                    value=gap,
                    baseline_value=0.0,
                    threshold=self.COVERAGE_GAP_CRITICAL,
                    severity="critical",
                    recommendation=(
                        f"{gap:.0%} of queries have no good corpus match. "
                        "Add documents covering new query topics."
                    ),
                ))

        for alert in alerts:
            logger.warning(
                "DRIFT ALERT: %s=%.4f (threshold=%.4f) severity=%s",
                alert.metric, alert.value, alert.threshold, alert.severity,
            )

        return alerts
```

---

## Exercise 3: Agent Failure Mode Diagnosis — Detecting Tool Call Loops

### Problem
An agent enters a tool call loop: it calls the same tool repeatedly with the same (or slightly varying) arguments, never making progress. Design detection and recovery for this failure mode.

### Answer

**Loop patterns to detect:**

1. **Exact loop**: identical tool + identical args, N times in a row
2. **Semantic loop**: same tool, slightly different args (LLM rephrasing), no progress
3. **Oscillation**: alternating between tool A and tool B with no state change
4. **Stall**: tool call rate drops to near-zero, LLM keeps generating but calling nothing

```python
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LoopType(str, Enum):
    EXACT = "exact"           # Same tool + same args
    SEMANTIC = "semantic"     # Same tool, similar args, no new state
    OSCILLATION = "oscillation"  # Alternating tools
    STALL = "stall"           # No tool calls for too long


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[str]
    timestamp: float
    call_index: int


@dataclass
class LoopDetection:
    loop_type: LoopType
    affected_tools: list[str]
    evidence: str
    call_indices: list[int]
    recommended_action: str


def _args_fingerprint(args: dict[str, Any]) -> str:
    """Deterministic hash of tool arguments for exact-loop detection."""
    return hashlib.md5(
        str(sorted(args.items())).encode()
    ).hexdigest()[:12]


def _extract_key_nouns(text: str) -> set[str]:
    """
    Poor-man's semantic comparison — extract non-stopword tokens.
    In production, use embedding similarity of the args string.
    """
    stopwords = {"the", "a", "an", "is", "are", "in", "of", "and", "or", "for", "with"}
    return {w.lower() for w in text.split() if w.lower() not in stopwords and len(w) > 3}


class AgentLoopDetector:
    """
    Stateful detector for agent tool call loops.
    Call record_tool_call() after each tool invocation.
    Call check_for_loops() periodically (e.g., every 3 calls).
    """

    EXACT_LOOP_THRESHOLD = 3     # Same fingerprint N times → exact loop
    SEMANTIC_LOOP_THRESHOLD = 4  # Same tool, similar args, N times → semantic loop
    OSCILLATION_WINDOW = 6       # Check last N calls for oscillation
    STALL_TIMEOUT_SECONDS = 30   # No tool call for N seconds → stall

    def __init__(self, max_history: int = 20) -> None:
        self._history: list[ToolCall] = []
        self._max_history = max_history
        self._last_call_time = time.monotonic()
        # fingerprint → list of call indices
        self._fingerprint_counts: dict[str, list[int]] = defaultdict(list)

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any], result: Optional[str] = None) -> None:
        call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            timestamp=time.monotonic(),
            call_index=len(self._history),
        )
        self._history.append(call)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        fp = f"{tool_name}:{_args_fingerprint(arguments)}"
        self._fingerprint_counts[fp].append(call.call_index)
        self._last_call_time = time.monotonic()
        logger.debug("Recorded tool call: %s (index=%d)", tool_name, call.call_index)

    def check_for_loops(self) -> list[LoopDetection]:
        detections: list[LoopDetection] = []
        recent = self._history[-self.OSCILLATION_WINDOW:] if len(self._history) >= 3 else self._history

        if not recent:
            return detections

        # 1. Exact loop detection
        for fp, indices in self._fingerprint_counts.items():
            recent_indices = [i for i in indices if i >= len(self._history) - self._max_history]
            if len(recent_indices) >= self.EXACT_LOOP_THRESHOLD:
                tool_name = fp.split(":")[0]
                detections.append(LoopDetection(
                    loop_type=LoopType.EXACT,
                    affected_tools=[tool_name],
                    evidence=f"Tool '{tool_name}' called {len(recent_indices)}× with identical arguments",
                    call_indices=recent_indices[-self.EXACT_LOOP_THRESHOLD:],
                    recommended_action=(
                        "Inject tool result summary into context: "
                        "'You have already called {tool} {n} times with the same arguments "
                        "and received: {result_summary}. Try a different approach.'"
                    ),
                ))

        # 2. Semantic loop: same tool, varying args, no progress in results
        tool_groups: dict[str, list[ToolCall]] = defaultdict(list)
        for call in recent:
            tool_groups[call.tool_name].append(call)

        for tool_name, calls in tool_groups.items():
            if len(calls) < self.SEMANTIC_LOOP_THRESHOLD:
                continue

            # Check if results are all similar (no progress)
            results = [c.result or "" for c in calls]
            result_tokens = [_extract_key_nouns(r) for r in results if r]
            if len(result_tokens) >= 2:
                # If Jaccard similarity between all pairs is high, results are the same
                pairs = [(result_tokens[i], result_tokens[j])
                         for i in range(len(result_tokens))
                         for j in range(i + 1, len(result_tokens))]
                avg_jaccard = mean(
                    len(a & b) / max(len(a | b), 1) for a, b in pairs
                ) if pairs else 0.0

                if avg_jaccard > 0.7:
                    detections.append(LoopDetection(
                        loop_type=LoopType.SEMANTIC,
                        affected_tools=[tool_name],
                        evidence=(
                            f"Tool '{tool_name}' called {len(calls)}× with varying args "
                            f"but results are {avg_jaccard:.0%} similar (no progress)"
                        ),
                        call_indices=[c.call_index for c in calls],
                        recommended_action=(
                            "Summarize past attempts and inject into context. "
                            "Force agent to try a different tool or reformulate the approach."
                        ),
                    ))

        # 3. Oscillation: alternating A→B→A→B
        if len(recent) >= 4:
            tool_seq = [c.tool_name for c in recent]
            if (tool_seq[-1] == tool_seq[-3] and tool_seq[-2] == tool_seq[-4]
                    and tool_seq[-1] != tool_seq[-2]):
                detections.append(LoopDetection(
                    loop_type=LoopType.OSCILLATION,
                    affected_tools=list({tool_seq[-1], tool_seq[-2]}),
                    evidence=f"Oscillation detected: ...{' → '.join(tool_seq[-4:])}",
                    call_indices=[c.call_index for c in recent[-4:]],
                    recommended_action=(
                        "Agent is switching between two tools without resolution. "
                        "Force a synthesis step: ask agent to summarize what it knows "
                        "and what specific information is still missing."
                    ),
                ))

        # 4. Stall: no tool call for too long
        idle_seconds = time.monotonic() - self._last_call_time
        if idle_seconds > self.STALL_TIMEOUT_SECONDS and self._history:
            detections.append(LoopDetection(
                loop_type=LoopType.STALL,
                affected_tools=[],
                evidence=f"No tool call for {idle_seconds:.0f}s (threshold={self.STALL_TIMEOUT_SECONDS}s)",
                call_indices=[],
                recommended_action=(
                    "Agent may be stuck in generation. Check LLM stream for activity. "
                    "If stream is alive but no tool calls, inject: "
                    "'Please make a tool call to continue or state your final answer.'"
                ),
            ))

        for d in detections:
            logger.error(
                "LOOP DETECTED: type=%s tools=%s evidence=%s",
                d.loop_type.value, d.affected_tools, d.evidence,
            )

        return detections


def mean(values):
    """Local mean to avoid import for simple use."""
    if not values:
        return 0.0
    return sum(values) / len(values)


class LoopBreaker:
    """
    Takes loop detection results and generates recovery interventions.
    """

    MAX_RETRIES_BEFORE_ABORT = 2  # How many loop interventions before giving up

    def __init__(self) -> None:
        self._intervention_count = 0

    def generate_intervention(self, detection: LoopDetection) -> str:
        """Generate a message to inject into agent context to break the loop."""
        self._intervention_count += 1

        if self._intervention_count > self.MAX_RETRIES_BEFORE_ABORT:
            return (
                "SYSTEM: You have been unable to complete this task after multiple attempts. "
                "Please respond to the user with your best partial answer and clearly state "
                "what you were unable to determine. Do not make any more tool calls."
            )

        if detection.loop_type == LoopType.EXACT:
            return (
                f"SYSTEM: You have called the same tool with identical arguments "
                f"{len(detection.call_indices)} times. The tool is not giving you new "
                f"information. Try: (1) a different tool, (2) different arguments, "
                f"or (3) answer based on what you already know."
            )
        elif detection.loop_type == LoopType.SEMANTIC:
            return (
                f"SYSTEM: Your recent tool calls to '{detection.affected_tools[0]}' "
                f"are all returning similar results. You are not making progress. "
                f"Synthesize what you have learned and either reach a conclusion "
                f"or explicitly state what information you cannot find."
            )
        elif detection.loop_type == LoopType.OSCILLATION:
            tools = " and ".join(detection.affected_tools)
            return (
                f"SYSTEM: You are alternating between {tools} without reaching a conclusion. "
                f"Stop and synthesize: What do you know? What is the specific gap? "
                f"Choose ONE next action or provide your best answer."
            )
        else:  # STALL
            return (
                "SYSTEM: You appear to be stuck. Please either make a tool call to "
                "continue your investigation or provide your final answer now."
            )
```

---

## Exercise 4: Canary Deployment for Model Upgrades

### Problem
You want to upgrade from one model version to a newer one. Design a canary strategy: traffic splits, success metrics, rollback triggers, and promotion criteria.

### Answer

**Canary stages:**

```
Stage 1 (Shadow):   0% canary traffic, 100% shadow (canary called but response not served)
Stage 2 (Canary 1): 5% canary, 95% baseline — alert on any hard degradation
Stage 3 (Canary 2): 20% canary — compare metrics with statistical significance
Stage 4 (Canary 3): 50% canary — final confidence check before full rollout
Stage 5 (Full):     100% canary (old model becomes standby for 24h, then retired)
```

**Rollback triggers (automatic):**

| Metric | Rollback if | Measure window |
|---|---|---|
| Error rate | canary > baseline × 2 | 5 minutes |
| P95 latency | canary > baseline × 1.5 | 10 minutes |
| Faithfulness score | canary < baseline − 0.05 | 100 requests |
| User negative feedback | canary > baseline × 2 | 15 minutes |
| Cost per request | canary > baseline × 1.3 | 10 minutes |

```python
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


class DeployStage(str, Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5"
    CANARY_20 = "canary_20"
    CANARY_50 = "canary_50"
    FULL = "full"
    ROLLED_BACK = "rolled_back"


STAGE_TRAFFIC: dict[DeployStage, float] = {
    DeployStage.SHADOW:     0.00,
    DeployStage.CANARY_5:   0.05,
    DeployStage.CANARY_20:  0.20,
    DeployStage.CANARY_50:  0.50,
    DeployStage.FULL:       1.00,
    DeployStage.ROLLED_BACK: 0.00,
}

STAGE_MIN_REQUESTS: dict[DeployStage, int] = {
    DeployStage.SHADOW:     200,
    DeployStage.CANARY_5:   100,
    DeployStage.CANARY_20:  500,
    DeployStage.CANARY_50:  1000,
    DeployStage.FULL:       0,
    DeployStage.ROLLED_BACK: 0,
}

STAGE_SOAK_MINUTES: dict[DeployStage, int] = {
    DeployStage.SHADOW:     30,
    DeployStage.CANARY_5:   60,
    DeployStage.CANARY_20:  120,
    DeployStage.CANARY_50:  240,
    DeployStage.FULL:       0,
    DeployStage.ROLLED_BACK: 0,
}


@dataclass
class ModelMetrics:
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latency_samples: list[float] = field(default_factory=list)
    faithfulness_scores: list[float] = field(default_factory=list)
    total_cost_usd: float = 0.0
    negative_feedback_count: int = 0

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.request_count, 1)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(0.95 * len(sorted_samples))
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def avg_faithfulness(self) -> float:
        return sum(self.faithfulness_scores) / max(len(self.faithfulness_scores), 1)

    @property
    def cost_per_request(self) -> float:
        return self.total_cost_usd / max(self.request_count, 1)


@dataclass
class RollbackEvent:
    timestamp: float
    stage: DeployStage
    trigger_metric: str
    canary_value: float
    baseline_value: float
    reason: str


class CanaryDeploymentController:
    """
    Controls canary deployment of a new model version.
    Automatically advances stages or triggers rollback based on metrics.
    """

    def __init__(
        self,
        baseline_model: str,
        canary_model: str,
    ) -> None:
        self._baseline_model = baseline_model
        self._canary_model = canary_model
        self._stage = DeployStage.SHADOW
        self._stage_start_time = time.monotonic()
        self._baseline_metrics = ModelMetrics()
        self._canary_metrics = ModelMetrics()
        self._rollback_events: list[RollbackEvent] = []
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def route_request(self) -> str:
        """Returns which model to use for this request."""
        traffic_fraction = STAGE_TRAFFIC[self._stage]

        if self._stage == DeployStage.SHADOW:
            # Always use baseline for user; record canary response separately
            return self._baseline_model

        if random.random() < traffic_fraction:
            return self._canary_model
        return self._baseline_model

    def record_request(
        self,
        model: str,
        latency_ms: float,
        is_error: bool,
        faithfulness_score: Optional[float],
        cost_usd: float,
        user_gave_negative_feedback: bool = False,
    ) -> None:
        metrics = self._canary_metrics if model == self._canary_model else self._baseline_metrics
        metrics.request_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.latency_samples.append(latency_ms)
        metrics.total_cost_usd += cost_usd
        if is_error:
            metrics.error_count += 1
        if faithfulness_score is not None:
            metrics.faithfulness_scores.append(faithfulness_score)
        if user_gave_negative_feedback:
            metrics.negative_feedback_count += 1

        # Check rollback triggers after each canary request
        if model == self._canary_model:
            self._check_rollback_triggers()

    def _check_rollback_triggers(self) -> None:
        """Check if any rollback trigger is breached."""
        b = self._baseline_metrics
        c = self._canary_metrics

        if c.request_count < 50:
            return  # Not enough data

        triggers = [
            ("error_rate", c.error_rate, b.error_rate * 2.0, 2.0),
            ("p95_latency_ms", c.p95_latency_ms, b.p95_latency_ms * 1.5, 1.5),
            ("cost_per_request", c.cost_per_request, b.cost_per_request * 1.3, 1.3),
        ]

        for metric_name, canary_val, threshold, multiplier in triggers:
            baseline_val = getattr(b, metric_name.replace("_ms", "").replace("_per_request", ""))
            if isinstance(baseline_val, property):
                baseline_val = getattr(b, metric_name)

            if canary_val > threshold and threshold > 0:
                self._trigger_rollback(
                    metric=metric_name,
                    canary_value=canary_val,
                    baseline_value=threshold / multiplier,
                    reason=f"{metric_name} ({canary_val:.3f}) > {multiplier}× baseline ({threshold/multiplier:.3f})",
                )
                return

        # Faithfulness check (lower is worse)
        if (len(c.faithfulness_scores) >= 100
                and len(b.faithfulness_scores) >= 100
                and c.avg_faithfulness < b.avg_faithfulness - 0.05):
            self._trigger_rollback(
                metric="faithfulness",
                canary_value=c.avg_faithfulness,
                baseline_value=b.avg_faithfulness,
                reason=f"Faithfulness dropped {b.avg_faithfulness - c.avg_faithfulness:.3f} below baseline",
            )

    def _trigger_rollback(self, metric: str, canary_value: float, baseline_value: float, reason: str) -> None:
        logger.error(
            "ROLLBACK TRIGGERED at stage=%s: %s canary=%.3f baseline=%.3f",
            self._stage.value, reason, canary_value, baseline_value,
        )
        event = RollbackEvent(
            timestamp=time.monotonic(),
            stage=self._stage,
            trigger_metric=metric,
            canary_value=canary_value,
            baseline_value=baseline_value,
            reason=reason,
        )
        self._rollback_events.append(event)
        self._stage = DeployStage.ROLLED_BACK

    def try_advance_stage(self) -> bool:
        """
        Check if conditions are met to advance to the next stage.
        Returns True if stage advanced.
        """
        if self._stage in (DeployStage.FULL, DeployStage.ROLLED_BACK):
            return False

        stage_order = [
            DeployStage.SHADOW,
            DeployStage.CANARY_5,
            DeployStage.CANARY_20,
            DeployStage.CANARY_50,
            DeployStage.FULL,
        ]
        current_idx = stage_order.index(self._stage)
        next_stage = stage_order[current_idx + 1]

        # Check soak time
        elapsed_minutes = (time.monotonic() - self._stage_start_time) / 60
        required_soak = STAGE_SOAK_MINUTES[self._stage]
        if elapsed_minutes < required_soak:
            logger.debug("Stage %s: %.0f min elapsed, need %d min", self._stage.value, elapsed_minutes, required_soak)
            return False

        # Check minimum request count
        req_count = self._canary_metrics.request_count
        required_requests = STAGE_MIN_REQUESTS[self._stage]
        if req_count < required_requests:
            logger.debug("Stage %s: %d requests, need %d", self._stage.value, req_count, required_requests)
            return False

        logger.info(
            "Advancing stage: %s → %s (soak=%.0fmin req_count=%d)",
            self._stage.value, next_stage.value, elapsed_minutes, req_count,
        )
        self._stage = next_stage
        self._stage_start_time = time.monotonic()
        return True

    @property
    def stage(self) -> DeployStage:
        return self._stage

    def get_deployment_summary(self) -> dict:
        return {
            "stage": self._stage.value,
            "baseline_model": self._baseline_model,
            "canary_model": self._canary_model,
            "canary_traffic_fraction": STAGE_TRAFFIC[self._stage],
            "canary_requests": self._canary_metrics.request_count,
            "canary_error_rate": self._canary_metrics.error_rate,
            "canary_p95_latency_ms": self._canary_metrics.p95_latency_ms,
            "canary_faithfulness": self._canary_metrics.avg_faithfulness,
            "baseline_error_rate": self._baseline_metrics.error_rate,
            "baseline_p95_latency_ms": self._baseline_metrics.p95_latency_ms,
            "rollback_count": len(self._rollback_events),
        }
```

**Promotion criteria for final Full rollout:**
1. Canary error rate ≤ baseline error rate × 1.1 (within 10%)
2. Canary P95 latency ≤ baseline P95 × 1.1
3. Canary faithfulness ≥ baseline faithfulness − 0.01 (within 1%)
4. No rollback events in the most recent 500 canary requests
5. All checks passed for entire 50% stage soak period (240 min minimum)

If any criterion fails during the 50% stage: automatic rollback to 5% for investigation, not full rollback. Reserve full rollback for hard breaches.

**Post-rollout:**
- Keep baseline model on standby (0% traffic) for 24h after Full stage reached
- Monitor production metrics for 24h before marking baseline as retired
- Keep baseline model weights for 30 days (instant rollback if a silent regression emerges)
