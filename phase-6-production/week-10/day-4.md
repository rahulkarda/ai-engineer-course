# Phase 6 · Week 10 · Day 4 — Speculative Decoding + Serving Engines

> Objective: Understand speculative decoding mechanics and production serving engines (vLLM, TGI).  ·  Time: ~1 hr  ·  Prereqs: [[day-3.md]]

## Concept

LLM inference is **memory-bandwidth-bound**, not compute-bound. The GPU sits mostly idle waiting for weights to be loaded per token. Speculative decoding exploits this slack.

**Speculative decoding** uses a small, fast *draft model* to propose several tokens at once. A large *target model* then verifies all proposed tokens in a **single forward pass** — because verification is a batch operation over known token positions (cheap), while generation requires sequential passes (expensive). Matching tokens are accepted; the first mismatch triggers rejection. Net result: 2–4x throughput gain on long generations with minimal quality loss.

**vLLM** is the dominant open-source inference engine. Key ideas:
- **PagedAttention**: KV cache managed in fixed-size pages (like OS virtual memory), eliminating fragmentation.
- **Continuous batching**: new requests join an in-flight batch mid-generation instead of waiting for the whole batch to finish.
- **OpenAI-compatible REST API**: drop-in replacement for OpenAI clients.

**Text Generation Inference (TGI)** — HuggingFace's serving stack:
- Tensor parallelism out of the box (split model across GPUs).
- Flash Attention 2 integration.
- Dynamic batching with token buckets.
- Docker-first deployment model.

**When to choose:**
| Criterion | vLLM | TGI |
|---|---|---|
| Multi-GPU tensor parallel | good | excellent |
| OpenAI API compatibility | native | via adapter |
| Custom model support | wide | HF Hub first |
| Memory efficiency | PagedAttention | dynamic blocks |
| Ecosystem | Python-native | Docker/HF-native |

## How it works internally

### Speculative decoding draft/verify loop

```
for step in range(max_new_tokens):
    # Draft phase — cheap, small model, runs K steps
    draft_tokens = draft_model.generate(context, K steps)   # K=4 typical

    # Verify phase — ONE forward pass through target model
    # target sees context + all K draft tokens simultaneously
    target_logits = target_model.forward(context + draft_tokens)

    # Acceptance sampling
    for i, (d_tok, t_logit) in enumerate(zip(draft_tokens, target_logits)):
        p_target = softmax(t_logit)[d_tok]
        p_draft  = draft_prob[i][d_tok]
        accept_prob = min(1.0, p_target / p_draft)
        if random() < accept_prob:
            accept token i
        else:
            # Reject: sample correction token from residual distribution
            # residual = max(0, p_target - p_draft), renormalized
            emit correction token, discard rest
            break
    # At best: accepted K tokens + 1 bonus target token = K+1 tokens per verify step
```

**Why it works without quality loss**: The acceptance criterion is mathematically equivalent to sampling from the target distribution — it is not an approximation. Rejected tokens get replaced by the true target distribution.

**Acceptance rate** `alpha` is the key metric. If the draft model is close to the target, `alpha` is high (~0.8–0.9) and speedup is large. If they diverge, `alpha` drops and overhead from running both models erases the gain.

**Speedup formula** (approximate):
```
speedup ≈ (1 - alpha^K) / (1 - alpha) × (cost_target / cost_draft_K_steps)
```

### vLLM worker architecture

```
Request → AsyncLLMEngine
           ├── Scheduler  (priority queue, preemption policy)
           ├── BlockManager  (PagedAttention: maps logical → physical KV blocks)
           └── Worker(s)  (one per GPU)
                └── ModelRunner → execute_model() → sampler → output tokens
```

The **Scheduler** runs a continuous batching loop: each iteration it:
1. Checks which sequences can be prefilled (prompt processing).
2. Checks which running sequences can be decoded (append one token).
3. Packs them into a single batch call.

**PagedAttention**: KV cache divided into blocks of fixed size (e.g., 16 tokens). The block manager tracks free/used blocks. Sequences share blocks for the common prefix (prefix caching). No pre-allocation per-sequence — memory is reclaimed immediately when a sequence ends.

### TGI router/shard/loader pattern

```
Client → Router (HTTP, load balance)
           └── Shard(s) (one per GPU, Python + Rust backend)
                └── Model Loader (safetensors, GGUF, GPTQ)
                     └── Flash Attention 2 kernel
```

TGI uses a **Rust router** for high-throughput request handling and Python shards for model execution. Tensor parallelism splits weight matrices across shards; each shard holds a slice of every attention head and MLP layer.

## Worked example

```python
"""
day4_speculative_decoding.py

Simulates speculative decoding acceptance mechanics without a real LLM.
Demonstrates:
  - How acceptance sampling works
  - How acceptance rate varies with draft model quality
  - Conceptual CLI invocations for vLLM and TGI
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

VOCAB_SIZE = 32_000
K_DRAFT_STEPS = 4  # tokens proposed per speculation step


# ---------------------------------------------------------------------------
# Token probability helpers
# ---------------------------------------------------------------------------

def make_distribution(vocab_size: int, seed: int) -> np.ndarray:
    """Return a random probability distribution over vocab_size tokens."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(vocab_size)
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def sample_token(probs: np.ndarray) -> int:
    return int(np.random.choice(len(probs), p=probs))


def residual_distribution(p_target: np.ndarray, p_draft: np.ndarray) -> np.ndarray:
    """
    When a draft token is rejected, the correction token is sampled from
    the 'residual' distribution: max(0, p_target - p_draft), renormalized.
    This ensures the overall output matches the target distribution exactly.
    """
    residual = np.maximum(0.0, p_target - p_draft)
    total = residual.sum()
    if total < 1e-9:
        # Fallback: sample from target directly
        return p_target
    return residual / total


# ---------------------------------------------------------------------------
# Mock draft and target models
# ---------------------------------------------------------------------------

@dataclass
class MockDraftModel:
    """
    Draft model: fast, smaller, lower quality.
    'quality' in [0, 1] — 1.0 means identical to target (100% accept rate).
    """
    quality: float  # how close draft is to target distribution
    vocab_size: int = VOCAB_SIZE

    def get_distribution(self, context_seed: int) -> np.ndarray:
        base = make_distribution(self.vocab_size, seed=context_seed)
        if self.quality >= 1.0:
            return base
        # Mix with noise to simulate lower-quality draft
        noise = make_distribution(self.vocab_size, seed=context_seed + 99_999)
        return self.quality * base + (1.0 - self.quality) * noise


@dataclass
class MockTargetModel:
    """
    Target model: slow, large, high quality — this is the ground truth.
    """
    vocab_size: int = VOCAB_SIZE

    def get_distribution(self, context_seed: int) -> np.ndarray:
        return make_distribution(self.vocab_size, seed=context_seed)

    def verify_batch(
        self,
        context_seed: int,
        draft_tokens: list[int],
    ) -> list[np.ndarray]:
        """
        Single forward pass: returns target distribution at each draft position.
        In a real model this is ONE matrix multiply over the full sequence.
        """
        distributions = []
        for offset, _ in enumerate(draft_tokens):
            # Each position sees the context + tokens before it
            distributions.append(
                make_distribution(self.vocab_size, seed=context_seed + offset)
            )
        return distributions


# ---------------------------------------------------------------------------
# Speculative decoder
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeDecoder:
    """
    Simulates the speculative decoding acceptance loop.

    Tracks:
      - tokens accepted per step
      - total draft tokens proposed
      - acceptance rate
    """
    draft_model: MockDraftModel
    target_model: MockTargetModel
    k: int = K_DRAFT_STEPS

    # Stats
    total_proposed: int = field(default=0, init=False)
    total_accepted: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)

    def decode_step(self, context_seed: int) -> list[int]:
        """
        One speculative decoding step.
        Returns list of emitted tokens (1 to K+1).
        """
        # --- Draft phase ---
        draft_tokens: list[int] = []
        draft_probs: list[np.ndarray] = []
        seed = context_seed
        for _ in range(self.k):
            p_draft = self.draft_model.get_distribution(seed)
            tok = sample_token(p_draft)
            draft_tokens.append(tok)
            draft_probs.append(p_draft)
            seed += 1

        # --- Verify phase (single target forward pass) ---
        target_dists = self.target_model.verify_batch(context_seed, draft_tokens)

        # --- Acceptance sampling ---
        accepted_tokens: list[int] = []
        for i, (d_tok, p_draft, p_target) in enumerate(
            zip(draft_tokens, draft_probs, target_dists)
        ):
            p_target_tok = p_target[d_tok]
            p_draft_tok = max(p_draft[d_tok], 1e-9)
            accept_prob = min(1.0, p_target_tok / p_draft_tok)

            if random.random() < accept_prob:
                accepted_tokens.append(d_tok)
            else:
                # Reject: emit correction token from residual distribution
                correction = sample_token(residual_distribution(p_target, p_draft))
                accepted_tokens.append(correction)
                self.total_proposed += i + 1
                self.total_accepted += i  # the rejected one doesn't count
                self.total_steps += 1
                return accepted_tokens

        # All K accepted — emit one bonus token from target
        bonus_tok = sample_token(target_dists[-1])
        accepted_tokens.append(bonus_tok)

        self.total_proposed += self.k
        self.total_accepted += self.k
        self.total_steps += 1
        return accepted_tokens

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def avg_tokens_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_accepted / self.total_steps


# ---------------------------------------------------------------------------
# Experiment: acceptance rate vs. draft model quality
# ---------------------------------------------------------------------------

def run_experiment(
    draft_quality: float,
    n_steps: int = 200,
    k: int = K_DRAFT_STEPS,
) -> dict[str, float]:
    draft = MockDraftModel(quality=draft_quality)
    target = MockTargetModel()
    decoder = SpeculativeDecoder(draft_model=draft, target_model=target, k=k)

    random.seed(42)
    np.random.seed(42)

    for step in range(n_steps):
        decoder.decode_step(context_seed=step * 100)

    return {
        "draft_quality": draft_quality,
        "acceptance_rate": round(decoder.acceptance_rate, 3),
        "avg_tokens_per_step": round(decoder.avg_tokens_per_step, 2),
        # Theoretical max: K+1 tokens per step when all accepted
        "efficiency_vs_baseline": round(decoder.avg_tokens_per_step / 1.0, 2),
    }


def main() -> None:
    log.info("Speculative decoding acceptance rate experiment")
    log.info("K (draft steps) = %d, vocab = %d", K_DRAFT_STEPS, VOCAB_SIZE)
    log.info("-" * 60)

    qualities = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    for q in qualities:
        result = run_experiment(draft_quality=q)
        log.info(
            "draft_quality=%.2f | accept_rate=%.3f | avg_tokens/step=%.2f",
            result["draft_quality"],
            result["acceptance_rate"],
            result["avg_tokens_per_step"],
        )

    log.info("-" * 60)
    log.info("With K=4 and quality=0.95, expect ~3-4x throughput gain")
    log.info("With K=4 and quality=0.50, draft overhead can negate gains")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Conceptual serving engine CLI invocations (NOT executed — reference only)
# ---------------------------------------------------------------------------

VLLM_SERVE_COMMAND = """
# vLLM: serve a model with OpenAI-compatible API
# Requires: pip install vllm
# Requires: GPU with sufficient VRAM

vllm serve meta-llama/Llama-3.2-8B-Instruct \\
    --tensor-parallel-size 1 \\
    --max-model-len 8192 \\
    --enable-prefix-caching \\
    --port 8000

# Speculative decoding with vLLM:
vllm serve meta-llama/Llama-3.2-70B-Instruct \\
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \\
    --num-speculative-tokens 5 \\
    --tensor-parallel-size 4 \\
    --port 8000

# Client call (OpenAI-compatible):
# curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \\
#   -d '{"model": "meta-llama/Llama-3.2-8B-Instruct", "prompt": "Hello", "max_tokens": 100}'
"""

TGI_SERVE_COMMAND = """
# TGI: serve via Docker (HuggingFace Text Generation Inference)
# Requires: Docker, NVIDIA drivers, huggingface-hub token

docker run --gpus all --shm-size 1g \\
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \\
    -p 8080:80 \\
    -v /data/tgi-models:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.2-8B-Instruct \\
    --num-shard 1 \\
    --max-input-length 4096 \\
    --max-total-tokens 8192

# Multi-GPU tensor parallel (4 GPUs):
# --num-shard 4

# Client call:
# curl http://localhost:8080/generate -H "Content-Type: application/json" \\
#   -d '{"inputs": "Hello", "parameters": {"max_new_tokens": 100}}'
"""
```

## Exercises

1. **Acceptance rate vs. K**: Modify the experiment loop to test K values (1, 2, 4, 8) at fixed draft quality 0.8. Plot or log a table of `avg_tokens_per_step` vs K. At what K does the return diminish?

2. **Break-even analysis**: The draft model costs ~10% of target model compute. Write a function `speedup_estimate(alpha: float, k: int, draft_cost_fraction: float) -> float` that returns the theoretical speedup. Find the `alpha` at which speedup < 1.0 (i.e., speculative decoding hurts).

3. **vLLM vs. TGI decision matrix**: Given four deployment scenarios below, write a `choose_engine(scenario: dict) -> str` function that returns `"vllm"`, `"tgi"`, or `"either"` with a reason string. Scenarios to handle: (a) single GPU, OpenAI SDK client, (b) 8-GPU cluster, HuggingFace model hub, tensor parallel required, (c) research prototype, frequent model swaps, (d) production, prefix caching critical, high concurrency.

(Solutions: ../solutions/week-10/day-4.md)

## Self-check

- [ ] I can explain why speculative decoding does not reduce output quality (acceptance sampling proof sketch).
- [ ] I can describe the vLLM scheduler loop: what happens each iteration.
- [ ] I can explain PagedAttention and why it reduces memory fragmentation.
- [ ] I can state when to prefer TGI over vLLM (tensor parallelism, HuggingFace-native deployment).
- [ ] I can implement a mock speculative decoder that correctly computes the residual distribution on rejection.

## References

- Speculative decoding paper: https://arxiv.org/abs/2211.17192
- vLLM documentation: https://docs.vllm.ai
- TGI documentation: https://huggingface.co/docs/text-generation-inference
