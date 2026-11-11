# Phase 3 · Week 07 · Day 7 — Sunday: Project Completion + Week 7 Assessment

> Objective: Polish the week-6 toolkit into a shippable project, run the full test suite, and assess Week 7 comprehension.  ·  Time: ~4 hr  ·  Prereqs: [[day-6]]

---

## Schedule

| Block | Time | Activity |
|-------|------|----------|
| Part 1 | 3 hr | Complete and polish the project |
| Part 2 | 1 hr | Assessment + reflection |

---

# Part 1 — Complete the Project (3 hr)

## 1.1 Project health check

Before adding anything new, verify the state of what you built on Day 6.

```bash
# From week-07/ directory
uv run pytest tests/ -v
```

All 30 tests should pass. If any fail, fix them before moving on. Common issues:
- `sys.path` not pointing to `src/` — check the `sys.path.insert` lines at the top of each test file.
- NumPy shape mismatches — check `(1, n_heads, d_head)` vs `(n_heads, d_head)` in `KVCache.append`.
- Floating point comparisons — use `pytest.approx` not `==` for floats.

## 1.2 Add a config module

Replace the hardcoded `MockModelConfig` dataclass in `mock_model.py` with a proper config module that also covers sampling parameters and logging.

Create `src/config.py`:

```python
# src/config.py
"""
Centralized configuration for the sampling toolkit project.

Uses Python dataclasses for simplicity. For production projects, use
pydantic-settings to load from .env and validate types automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture configuration for the mock transformer model."""
    n_vocab: int = 100
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 16
    d_model: int = 64   # must equal n_heads * d_head
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError(
                f"d_model ({self.d_model}) must equal "
                f"n_heads ({self.n_heads}) * d_head ({self.d_head})"
            )


@dataclass
class SamplingConfig:
    """Default sampling hyperparameters."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")


@dataclass
class AppConfig:
    """Top-level application config."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    log_level: str = "INFO"
```

## 1.3 Add a CLI entry point

Add `src/main.py` so the project can be invoked as:

```bash
uv run python src/main.py --strategy greedy --prompt "10 23 45" --max-tokens 20
```

```python
# src/main.py
"""
CLI entry point for the sampling toolkit demo.

Usage:
    uv run python src/main.py --strategy greedy --prompt "10 23 45 7" --max-tokens 20
    uv run python src/main.py --strategy top_k --top-k 10 --temperature 0.9
    uv run python src/main.py --strategy top_p --top-p 0.9 --temperature 1.0
    uv run python src/main.py --strategy combined --top-k 20 --top-p 0.95 --rep-penalty 1.3

Install: uv add numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import AppConfig, ModelConfig, SamplingConfig
from mock_model import MockModel
from demo import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoregressive decoding demo with configurable sampling strategies."
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "temperature", "top_k", "top_p", "combined"],
        default="combined",
        help="Sampling strategy to use.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="10 23 45 7",
        help="Space-separated token IDs for the prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Number of new tokens to generate.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def build_sampling_config(args: argparse.Namespace) -> SamplingConfig:
    """Map CLI args and strategy presets to a SamplingConfig."""
    strategy_defaults: dict[str, dict[str, float | int]] = {
        "greedy":      {"temperature": 0.0001, "top_k": 1},
        "temperature": {"temperature": args.temperature},
        "top_k":       {"temperature": args.temperature, "top_k": args.top_k or 10},
        "top_p":       {"temperature": args.temperature, "top_p": args.top_p or 0.9},
        "combined":    {
            "temperature": args.temperature,
            "top_k": args.top_k or 20,
            "top_p": args.top_p or 0.95,
            "repetition_penalty": args.rep_penalty or 1.3,
        },
    }
    defaults = strategy_defaults[args.strategy]
    return SamplingConfig(
        temperature=float(defaults.get("temperature", args.temperature)),
        top_k=int(defaults.get("top_k", args.top_k)),
        top_p=float(defaults.get("top_p", args.top_p)),
        repetition_penalty=float(defaults.get("repetition_penalty", args.rep_penalty)),
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    # Parse prompt
    try:
        prompt_ids = [int(t) for t in args.prompt.split()]
    except ValueError as exc:
        log.error("Invalid prompt token IDs: %s", exc)
        return 1

    # Build config
    cfg = AppConfig(sampling=build_sampling_config(args))
    log.info("Strategy: %s", args.strategy)
    log.info("Sampling config: %s", cfg.sampling)

    # Build model
    model_cfg = ModelConfig()
    model = MockModel(model_cfg)

    # Validate prompt token IDs
    invalid = [t for t in prompt_ids if t < 0 or t >= model_cfg.n_vocab]
    if invalid:
        log.error("Token IDs out of vocab range [0, %d): %s", model_cfg.n_vocab, invalid)
        return 1

    # Generate
    rng = np.random.default_rng(seed=cfg.sampling.seed)
    generated = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temperature,
        top_k=cfg.sampling.top_k,
        top_p=cfg.sampling.top_p,
        repetition_penalty=cfg.sampling.repetition_penalty,
        rng=rng,
    )

    log.info("Prompt  : %s", prompt_ids)
    log.info("Generated: %s", generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## 1.4 Add config tests

Add `tests/test_config.py`:

```python
# tests/test_config.py
"""Tests for src/config.py"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from config import ModelConfig, SamplingConfig, AppConfig


class TestModelConfig:
    def test_default_d_model_valid(self) -> None:
        cfg = ModelConfig()
        assert cfg.d_model == cfg.n_heads * cfg.d_head

    def test_invalid_d_model_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(n_heads=4, d_head=16, d_model=99)


class TestSamplingConfig:
    def test_defaults_valid(self) -> None:
        cfg = SamplingConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            SamplingConfig(temperature=0.0)

    def test_top_p_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            SamplingConfig(top_p=1.5)

    def test_rep_penalty_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            SamplingConfig(repetition_penalty=0.8)


class TestAppConfig:
    def test_default_app_config(self) -> None:
        cfg = AppConfig()
        assert cfg.model.n_vocab == 100
        assert cfg.sampling.temperature == 1.0
        assert cfg.log_level == "INFO"
```

## 1.5 Self-review checklist

Before marking the project complete, verify each item:

- [ ] `uv run pytest tests/ -v` passes — all tests green, no warnings about missing fixtures.
- [ ] `uv run python src/main.py --strategy greedy` runs without error and prints generated tokens.
- [ ] `uv run python src/main.py --strategy combined --top-k 20 --top-p 0.9 --rep-penalty 1.3` runs.
- [ ] `uv run python src/demo.py` prints the KV cache memory comparison table.
- [ ] No `print()` calls in `src/*.py` — only `log.info`, `log.debug`, etc.
- [ ] No hardcoded secrets or API keys anywhere.
- [ ] All public functions have type hints on arguments and return type.
- [ ] All public functions have docstrings explaining purpose, args, and return value.
- [ ] `config.py` validates inputs in `__post_init__` — bad values raise `ValueError`.
- [ ] `KVCache.append` validates layer index and tensor shape before mutating state.

## 1.6 Final project structure

```
week-07/
  src/
    config.py       -- ModelConfig, SamplingConfig, AppConfig
    sampling.py     -- greedy_decode, temperature_scale, top_k_filter, top_p_filter,
                        apply_repetition_penalty, sample_token
    kv_cache.py     -- KVCache dataclass
    mock_model.py   -- MockModel, MockModelConfig
    demo.py         -- generate(), compare_kv_cache_memory(), main()
    main.py         -- CLI entry point (argparse)
  tests/
    test_sampling.py    -- 18 tests
    test_kv_cache.py    -- 10 tests
    test_config.py      -- 5 tests
  day-5.md
  day-6.md
  day-7.md
```

---

# Part 2 — Week 7 Assessment (1 hr)

## Assessment

See `../assessment.md` (or create it in `week-07/` if not present) for the formal assessment questions.

### Reflection questions

Answer these in writing — either in a personal notes file or out loud. Aim for 2-3 sentences each.

1. **Tokenization trade-off**: BPE tokenizers use a fixed vocabulary learned from training data. A rare word like "microservices" might be split into 2-4 subword tokens while a common word like "the" is always 1 token. How does this affect the effective context length when processing technical documentation vs. natural language? What does this imply for domain-specific models?

2. **KV cache and context length**: You calculated in Day 6 that a 7B model with a 8192-token context uses about 4 GB for the KV cache (in FP16). A 70B model with the same context would use roughly 10x more memory. Given that GPU memory is fixed, what architectural choices can a model designer make to reduce KV cache size without reducing context length? Name at least two.

3. **Sampling in production**: You implemented five sampling strategies. A production chatbot needs to balance coherence (not generating nonsense) with creativity (not always giving the same answer). Describe a concrete strategy — which combination of temperature, top-k, top-p, and repetition penalty would you use for: (a) a factual Q&A assistant, (b) a creative story generator. Justify each parameter choice.

---

## Week 7 Summary

### What you covered this week

**Day 1 — Pretraining & the LLM Stack**
- How LLMs are trained from scratch on internet-scale text.
- The three-stage pipeline: pretraining -> SFT -> RLHF.
- Why instruction tuning is not the same as pretraining.

**Day 2 — Tokenization & Context Windows**
- BPE tokenization: how the vocabulary is built from merge rules.
- Tokens vs words vs characters: why tokenization matters for model behavior.
- Context windows: absolute limits, positional encoding, and why longer isn't always better.

**Day 3 — Autoregressive Decoding & KV Cache**
- The decode loop: one token per forward pass.
- Why KV cache exists: avoid recomputing past attention keys/values.
- Memory cost of KV cache: 2 * n_layers * n_heads * d_head * seq_len * dtype_bytes.

**Day 4 — Sampling Strategies**
- Temperature, top-k, top-p (nucleus), repetition penalty.
- The probability pipeline: logits -> temperature scale -> filter -> softmax -> categorical sample.
- How each strategy trades off determinism vs diversity.

**Day 5 — Latency, Throughput, Quantization & Scaling Laws**
- TTFT vs TBT: prefill is compute-bound, decode is memory-bandwidth-bound.
- Batching amortizes weight loading across sequences.
- INT8/INT4 quantization: 4x/8x memory savings with small accuracy loss.
- Chinchilla optimal: 20 tokens per parameter; total FLOPs ~ 6ND.

**Day 6 — Deep Practice: Sampling Toolkit + KV Cache**
- Full implementation of all sampling functions.
- KV cache as a dataclass with typed append/get/memory_bytes methods.
- Mock autoregressive model connecting both components.
- Pytest suite covering all edge cases.

**Day 7 — Project Completion + Assessment**
- Config module with validated dataclasses.
- CLI entry point with argparse.
- End-to-end test run.

### What to remember

Three things that will come up in every LLM engineering role:

1. **Batching is how you get throughput.** A single-user latency demo means nothing in production. Always think about how your system performs under load with concurrent requests sharing the same GPU.

2. **Quantization is the first tool to reach for when a model is too large.** INT8 costs you ~1% accuracy and halves memory. INT4 (AWQ/GPTQ) halves it again. Before adding GPUs, quantize.

3. **Sampling hyperparameters matter.** The same model with temperature=0.1 vs temperature=1.2 behaves like two different products. Understand what each parameter does mechanically — you will tune these in every deployment.

---

## Self-check

- [ ] I can explain the full pretraining -> SFT -> RLHF pipeline and when each stage is necessary.
- [ ] I can calculate KV cache memory for a given model architecture (n_layers, n_heads, d_head, seq_len, dtype).
- [ ] I can implement all five sampling strategies from memory (greedy, temperature, top-k, top-p, repetition penalty) and explain the pipeline order.
- [ ] I can state the Chinchilla optimal token-to-parameter ratio and calculate optimal N and D for a given compute budget.
- [ ] I can run the complete test suite and interpret failures — I understand what each test is checking and why.
