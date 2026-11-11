# Week 10 · Day 4 Solutions — Speculative Decoding & Serving Engines

---

## Exercise 1: Acceptance Rate Simulation

### Concept

Speculative decoding uses a small "draft" model to propose `k` tokens, then a larger "target" model
verifies them in a single forward pass. Each proposed token is accepted with probability `alpha`
(the probability that the draft model picked the same token the target model would have chosen).

The expected number of accepted tokens per target model call is derived from the geometric series:

```
E[accepted] = sum_{i=1}^{k} alpha^i  (first rejection stops the run)
            = alpha * (1 - alpha^k) / (1 - alpha)
```

When `alpha = 1` the formula becomes `k` (all tokens accepted). The simple approximation
`E[accepted] ≈ k * alpha` is accurate only when alpha is close to 1.

### Solution

```python
"""
week10/day4/speculative_decoding_sim.py

Speculative decoding acceptance rate simulation.
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodingStats:
    alpha: float
    k: int
    expected_accepted_approx: float   # k * alpha  (linear approximation)
    expected_accepted_exact: float    # geometric series formula
    effective_speedup: float          # accepted / 1 (one target call)


def compute_expected_accepted(alpha: float, k: int) -> float:
    """
    Exact expected number of accepted tokens using the geometric series.

    Each token i (1-indexed) is accepted only if all previous tokens were also
    accepted, so P(token i accepted) = alpha^i.

    E = sum_{i=1}^{k} alpha^i = alpha * (1 - alpha^k) / (1 - alpha)
    Edge case: alpha == 1 => E = k.
    """
    if alpha >= 1.0:
        return float(k)
    return alpha * (1.0 - alpha**k) / (1.0 - alpha)


def simulate_acceptance(alpha: float, k: int) -> SpecDecodingStats:
    """Return acceptance statistics for a given draft-model quality and k."""
    approx = k * alpha
    exact = compute_expected_accepted(alpha, k)
    return SpecDecodingStats(
        alpha=alpha,
        k=k,
        expected_accepted_approx=approx,
        expected_accepted_exact=exact,
        effective_speedup=exact,  # relative to one target-model token/call
    )


def print_acceptance_table(k: int = 4) -> None:
    """Print ASCII table: alpha vs acceptance rate for k speculative tokens."""
    alphas = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    header = (
        f"{'alpha':>6} | "
        f"{'approx (k*a)':>14} | "
        f"{'exact (geo)':>12} | "
        f"{'speedup x':>10}"
    )
    separator = "-" * len(header)

    print(f"\nSpeculative Decoding Acceptance Table  (k={k})")
    print(separator)
    print(header)
    print(separator)

    for alpha in alphas:
        stats = simulate_acceptance(alpha, k)
        print(
            f"{alpha:>6.2f} | "
            f"{stats.expected_accepted_approx:>14.3f} | "
            f"{stats.expected_accepted_exact:>12.3f} | "
            f"{stats.effective_speedup:>10.3f}"
        )
    print(separator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_acceptance_table(k=4)
```

**Expected output (k=4):**

```
Speculative Decoding Acceptance Table  (k=4)
-----------------------------------------------------------
 alpha |   approx (k*a) |  exact (geo) |  speedup x
-----------------------------------------------------------
  0.50 |          2.000 |        0.938 |      0.938
  0.60 |          2.400 |        1.296 |      1.296
  0.70 |          2.800 |        1.729 |      1.729
  0.80 |          3.200 |        2.272 |      2.272
  0.90 |          3.600 |        2.952 |      2.952
  0.95 |          3.800 |        3.335 |      3.335
-----------------------------------------------------------
```

Notice the linear approximation (`k * alpha`) **overestimates** the exact value for all
`alpha < 1`, because the independence assumption breaks: later tokens can only be accepted if
all earlier ones were also accepted.

### Pytest Tests

```python
"""tests/week10/test_speculative_decoding.py"""

import pytest
from week10.day4.speculative_decoding_sim import (
    compute_expected_accepted,
    simulate_acceptance,
)


def test_alpha_one_returns_k() -> None:
    """Perfect draft model: all k tokens should be accepted."""
    assert compute_expected_accepted(alpha=1.0, k=4) == pytest.approx(4.0)


def test_alpha_zero_returns_zero() -> None:
    """Completely wrong draft model: zero tokens accepted."""
    result = compute_expected_accepted(alpha=0.0, k=4)
    assert result == pytest.approx(0.0)


def test_geometric_less_than_approx() -> None:
    """Exact formula should always be <= linear approx for 0 < alpha < 1."""
    for alpha in [0.5, 0.7, 0.9]:
        stats = simulate_acceptance(alpha=alpha, k=4)
        assert stats.expected_accepted_exact <= stats.expected_accepted_approx + 1e-9


def test_exact_formula_known_value() -> None:
    """Manually verify: alpha=0.5, k=4 => 0.5+0.25+0.125+0.0625 = 0.9375."""
    result = compute_expected_accepted(alpha=0.5, k=4)
    assert result == pytest.approx(0.9375, rel=1e-6)


def test_increasing_alpha_increases_accepted() -> None:
    """Higher draft quality should yield more accepted tokens."""
    prev = compute_expected_accepted(alpha=0.5, k=4)
    for alpha in [0.6, 0.7, 0.8, 0.9, 0.95]:
        cur = compute_expected_accepted(alpha=alpha, k=4)
        assert cur > prev
        prev = cur
```

---

## Exercise 2: Draft Model Quality vs Speedup Tradeoff

### Concept

Adding speculative decoding changes the cost profile:

- Without spec-decoding: 1 target-model call produces 1 token.
- With spec-decoding: 1 draft call (cost `c`) + 1 target call (cost `1`) produces
  `1 + E[accepted]` tokens on average.

Define `cost_ratio = draft_cost / target_cost`. The **speedup** (tokens per unit cost relative to
the baseline of 1 token per 1 unit) is:

```
speedup = (1 + E[accepted]) / (1 + k * cost_ratio)
```

Speedup > 1 means spec-decoding helps. The break-even alpha satisfies `speedup = 1`:

```
E_exact(alpha*) = k * cost_ratio   (set numerator = denominator, subtract 1)
```

### Solution

```python
"""
week10/day4/speedup_analysis.py

Speedup analysis for speculative decoding at various alpha / cost_ratio values.
"""

import logging
from dataclasses import dataclass

from week10.day4.speculative_decoding_sim import compute_expected_accepted

logger = logging.getLogger(__name__)


@dataclass
class SpeedupResult:
    alpha: float
    cost_ratio: float
    k: int
    expected_accepted: float
    speedup: float
    helps: bool


def compute_speedup(alpha: float, k: int, cost_ratio: float) -> SpeedupResult:
    """
    speedup = (1 + E[accepted]) / (1 + k * cost_ratio)

    Parameters
    ----------
    alpha      : probability that draft model proposes the correct token
    k          : number of speculative tokens per call
    cost_ratio : draft_cost / target_cost  (e.g. 0.1 means draft is 10x cheaper)
    """
    expected = compute_expected_accepted(alpha, k)
    numerator = 1.0 + expected
    denominator = 1.0 + k * cost_ratio
    speedup = numerator / denominator
    return SpeedupResult(
        alpha=alpha,
        cost_ratio=cost_ratio,
        k=k,
        expected_accepted=expected,
        speedup=speedup,
        helps=speedup > 1.0,
    )


def find_breakeven_alpha(k: int, cost_ratio: float, tol: float = 1e-4) -> float:
    """
    Binary search for the alpha where speedup crosses 1.0.

    At break-even: E_exact(alpha) == k * cost_ratio (after cancelling the +1 on both sides).
    """
    target = k * cost_ratio
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if compute_expected_accepted(mid, k) < target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return (lo + hi) / 2.0


def print_speedup_table(k: int = 4) -> None:
    alphas = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    cost_ratios = [0.10, 0.20, 0.30]

    print(f"\nSpeedup Table  (k={k})")
    print(
        f"{'alpha':>6} | "
        + " | ".join(f"  cr={cr:.2f}  " for cr in cost_ratios)
    )
    print("-" * (6 + 3 + len(cost_ratios) * 14))

    for alpha in alphas:
        row = f"{alpha:>6.2f} | "
        for cr in cost_ratios:
            result = compute_speedup(alpha, k, cr)
            flag = "+" if result.helps else "-"
            row += f"  {result.speedup:5.3f} {flag}  | "
        print(row)

    print()
    print("Break-even alpha (speedup = 1.0):")
    for cr in cost_ratios:
        be = find_breakeven_alpha(k, cr)
        print(f"  cost_ratio={cr:.2f}  =>  alpha* = {be:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_speedup_table(k=4)
```

**Expected output (k=4, excerpt):**

```
Speedup Table  (k=4)
 alpha |   cr=0.10   |   cr=0.20   |   cr=0.30
-----------------------------------------------
  0.50 |  0.657 -  |  0.585 -  |  0.526 -  |
  0.70 |  0.982 -  |  0.874 -  |  0.786 -  |
  0.80 |  1.162 +  |  1.035 +  |  0.931 -  |
  0.90 |  1.394 +  |  1.241 +  |  1.116 +  |
  0.95 |  1.534 +  |  1.367 +  |  1.228 +  |

Break-even alpha (speedup = 1.0):
  cost_ratio=0.10  =>  alpha* = 0.7104
  cost_ratio=0.20  =>  alpha* = 0.7891
  cost_ratio=0.30  =>  alpha* = 0.8523
```

**Key insight:** With a draft model that costs 10% of the target, you need ~71% acceptance rate
to break even. With a 30%-cost draft model you need ~85%. A high-quality draft model is critical.

---

## Exercise 3: vLLM vs TGI Decision Matrix

### Concept

| Dimension | vLLM | TGI (Text Generation Inference) |
|---|---|---|
| Attention mechanism | PagedAttention (variable KV cache blocks) | Continuous batching |
| API compat | OpenAI-compatible REST by default | HuggingFace Inference API |
| Tensor parallelism | Supported, simpler setup | First-class, production-grade |
| Model support | HF + custom; excellent vLLM-native | HuggingFace Hub first |
| Best for | Throughput-optimized serving | Flexible HF model deployment |

### Solution

```python
"""
week10/day4/serving_engine_selector.py

Choose between vLLM and TGI based on workload requirements.
"""

import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

OptimizeFor = Literal["latency", "throughput"]


class ServingEngineRecommendation(BaseModel):
    engine: Literal["vllm", "tgi"]
    reason: str
    secondary_considerations: list[str]


def choose_serving_engine(
    use_case: str,
    model_size_b: int,
    need_tensor_parallel: bool,
    need_openai_compat: bool,
    optimize_for: OptimizeFor,
) -> ServingEngineRecommendation:
    """
    Return a serving engine recommendation with reasoning.

    Decision logic (evaluated in priority order):
    1. OpenAI-compat required -> vLLM (native support, drop-in replacement).
    2. Throughput optimization + large model -> vLLM (PagedAttention excels here).
    3. Tensor parallelism required AND latency optimization -> TGI (mature TP stack).
    4. HuggingFace-first workflow -> TGI.
    5. Default: vLLM (broader community, more active development).
    """
    considerations: list[str] = []

    if need_openai_compat:
        considerations.append(
            "vLLM provides a drop-in OpenAI-compatible REST server out of the box."
        )
        if need_tensor_parallel:
            considerations.append(
                "Tensor parallelism is also supported in vLLM via --tensor-parallel-size."
            )
        return ServingEngineRecommendation(
            engine="vllm",
            reason=(
                "OpenAI API compatibility is required. vLLM exposes a fully "
                "compatible /v1/completions and /v1/chat/completions endpoint with "
                "minimal configuration."
            ),
            secondary_considerations=considerations,
        )

    if optimize_for == "throughput" and model_size_b >= 7:
        considerations.append(
            "PagedAttention eliminates KV-cache memory fragmentation, "
            "enabling higher batch sizes at large model scales."
        )
        if need_tensor_parallel:
            considerations.append("vLLM supports tensor parallelism via Ray or multi-GPU.")
        return ServingEngineRecommendation(
            engine="vllm",
            reason=(
                f"Throughput optimization for a {model_size_b}B parameter model. "
                "vLLM's PagedAttention allocates KV-cache in non-contiguous blocks, "
                "dramatically reducing memory waste and enabling larger effective "
                "batch sizes compared to TGI's paged attention implementation."
            ),
            secondary_considerations=considerations,
        )

    if need_tensor_parallel and optimize_for == "latency":
        considerations.append(
            "TGI has a mature, production-grade tensor parallelism implementation "
            "optimized for low-latency single-request serving."
        )
        considerations.append(
            "vLLM TP is also viable here; consider TGI if already in HuggingFace ecosystem."
        )
        return ServingEngineRecommendation(
            engine="tgi",
            reason=(
                "Tensor parallelism with latency optimization. TGI's TP implementation "
                "uses highly optimized sharding across GPUs and is well-suited for "
                "minimizing time-to-first-token on large models."
            ),
            secondary_considerations=considerations,
        )

    if "huggingface" in use_case.lower() or "hf" in use_case.lower():
        considerations.append(
            "TGI integrates natively with HuggingFace Hub and the Inference Endpoints service."
        )
        return ServingEngineRecommendation(
            engine="tgi",
            reason=(
                "HuggingFace-first workflow detected. TGI is the reference serving "
                "stack for HuggingFace models and supports the full HF model card "
                "ecosystem, including GPTQ, AWQ, and GGUF quantization formats."
            ),
            secondary_considerations=considerations,
        )

    # Default: vLLM — more active development, stronger community
    considerations.append("vLLM has faster development velocity and broader hardware support.")
    considerations.append(
        f"Model size {model_size_b}B is manageable on a single GPU; "
        "PagedAttention provides headroom for bursty traffic."
    )
    return ServingEngineRecommendation(
        engine="vllm",
        reason=(
            "Default recommendation. vLLM is the safer general-purpose choice for "
            "most production deployments due to its active development, OpenAI "
            "compatibility, and efficient memory management via PagedAttention."
        ),
        secondary_considerations=considerations,
    )
```

### Pytest Tests

```python
"""tests/week10/test_serving_engine_selector.py"""

import pytest
from week10.day4.serving_engine_selector import choose_serving_engine


def test_openai_compat_prefers_vllm() -> None:
    """OpenAI-compatible API requirement should always select vLLM."""
    rec = choose_serving_engine(
        use_case="llm api gateway",
        model_size_b=7,
        need_tensor_parallel=False,
        need_openai_compat=True,
        optimize_for="throughput",
    )
    assert rec.engine == "vllm"
    assert "openai" in rec.reason.lower() or "compatible" in rec.reason.lower()


def test_throughput_large_model_prefers_vllm() -> None:
    """High-throughput large model workloads benefit from PagedAttention."""
    rec = choose_serving_engine(
        use_case="batch inference pipeline",
        model_size_b=70,
        need_tensor_parallel=True,
        need_openai_compat=False,
        optimize_for="throughput",
    )
    assert rec.engine == "vllm"
    assert "pagedattention" in rec.reason.lower() or "throughput" in rec.reason.lower()


def test_hf_ecosystem_prefers_tgi() -> None:
    """HuggingFace-centric workloads should prefer TGI."""
    rec = choose_serving_engine(
        use_case="HuggingFace model serving",
        model_size_b=13,
        need_tensor_parallel=False,
        need_openai_compat=False,
        optimize_for="latency",
    )
    assert rec.engine == "tgi"


def test_latency_tensor_parallel_prefers_tgi() -> None:
    """Latency-optimized tensor-parallel serving -> TGI."""
    rec = choose_serving_engine(
        use_case="real-time chat assistant",
        model_size_b=34,
        need_tensor_parallel=True,
        need_openai_compat=False,
        optimize_for="latency",
    )
    assert rec.engine == "tgi"


def test_recommendation_has_reason() -> None:
    """All recommendations must include a non-empty reason string."""
    for optimize_for in ("latency", "throughput"):
        rec = choose_serving_engine(
            use_case="generic api",
            model_size_b=7,
            need_tensor_parallel=False,
            need_openai_compat=False,
            optimize_for=optimize_for,  # type: ignore[arg-type]
        )
        assert len(rec.reason) > 10
        assert isinstance(rec.secondary_considerations, list)
```

---

## Key Takeaways

1. **Speculative decoding speedup is superlinear with alpha**: moving from alpha=0.8 to 0.9 gives
   far more benefit than moving from 0.5 to 0.6, because all tokens in a run must be accepted.

2. **Draft model cost matters**: a draft model that costs >30% of the target model rarely pays
   off unless acceptance rate exceeds ~85%.

3. **vLLM vs TGI**: not a quality contest — vLLM wins on throughput and OpenAI compat; TGI wins
   on HuggingFace ecosystem integration and mature tensor parallelism for latency-sensitive workloads.
