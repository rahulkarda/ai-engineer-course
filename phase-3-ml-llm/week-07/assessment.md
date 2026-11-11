# Week 07 Assessment — LLM Training & Inference Fundamentals

## Conceptual (interview-style)

Answer each question in 2–5 sentences. Precision matters more than length.

**1. Pretraining vs. instruction tuning**

Describe the difference between pretraining and instruction tuning. What changes between the two phases, and what does the training data look like for each?

**2. RLHF vs. SFT alone**

Why does RLHF (Reinforcement Learning from Human Feedback) tend to produce better-aligned models than supervised fine-tuning alone? What signal does SFT lack that RLHF provides?

**3. Context window overflow**

What happens when you feed a 2000-token sequence to a model trained with a 1024-token context window? Describe the behavior at the positional encoding level and the likely output quality degradation.

**4. KV cache: compute and memory tradeoffs**

Explain why a KV cache reduces compute during autoregressive inference. What memory cost does it introduce, and how does that cost scale with sequence length?

**5. KV cache memory calculation**

A GPT-2-style model has 12 layers, 12 attention heads, head dimension d_head = 64, and uses FP16 weights. Calculate the KV cache memory in megabytes for a sequence length of 1024 tokens. Show your work.

**6. Temperature and greedy decoding**

Mathematically, why does temperature T approaching 0 converge to greedy decoding? Write out the softmax with temperature and show what happens to the distribution as T → 0.

**7. Top-k vs. top-p sampling**

Explain the difference between top-k and top-p (nucleus) sampling. Under what conditions does top-p outperform top-k, and why?

**8. Scaling laws and Chinchilla**

What are neural scaling laws? State the key finding from the Chinchilla paper in one sentence. What did it change about how practitioners think about compute budgets?

**9. Quantization tradeoffs**

What is weight quantization in the context of LLM inference? Compare INT4 and FP16 quantization on three axes: model size, throughput, and output quality.

**10. Inference latency metrics**

Define Time to First Token (TTFT) and Time Between Tokens (TBT). Which metric is dominated by the prefill phase? Which by the decode phase? Explain why.

---

## Coding Challenge

### Task

Implement a `TextGenerator` class in Python 3.12+ that simulates autoregressive token generation with sampling controls.

You are given a mock logits function with the following signature:

```python
def get_logits(token_ids: list[int]) -> np.ndarray:
    ...
```

It accepts the full sequence of token IDs generated so far and returns a 1-D NumPy array of raw logits over the vocabulary.

### Requirements

Implement `TextGenerator` in a single file `text_generator.py`:

```python
import numpy as np
import logging
from typing import Callable

class TextGenerator:
    def __init__(
        self,
        vocab_size: int,
        get_logits: Callable[[list[int]], np.ndarray],
    ) -> None:
        ...

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.3,
    ) -> list[int]:
        ...
```

### Behavioral specification

- `generate()` must return exactly `max_new_tokens` new token IDs (not including the prompt).
- Sampling pipeline per step:
  1. Apply repetition penalty to logits for any token already in the sequence.
  2. Apply temperature scaling (divide logits by T before softmax).
  3. Apply top-p filtering: set all tokens outside the nucleus to `-inf` before sampling.
  4. Sample one token from the resulting distribution.
- When `temperature <= 1e-8`, use greedy decoding (argmax) instead of sampling.
- Maintain a conceptual KV cache as a list that grows by one entry per generation step. The list is not required to store actual tensors — it may store token IDs, step indices, or any sentinel that tracks sequence growth. It must be accessible as `self.kv_cache` after `generate()` returns.
- Use `logging` (not `print`) for any diagnostic output.
- All public methods and the class itself must have type hints and docstrings.

### Acceptance criteria

- [ ] Class and `generate()` have correct type hints and docstrings.
- [ ] `generate(prompt_ids, max_new_tokens=5, ...)` returns a list of exactly 5 token IDs.
- [ ] Calling `generate()` with `temperature=0.0` produces the same output on repeated calls with the same inputs (deterministic greedy).
- [ ] Calling `generate()` with `top_p=1.0` and `repetition_penalty=1.0` is equivalent to standard temperature sampling (no nucleus truncation, no penalty).
- [ ] `self.kv_cache` has length equal to `len(prompt_ids) + max_new_tokens` after `generate()` returns.
- [ ] At least 3 pytest tests pass (place them in `tests/test_text_generator.py`).
- [ ] No `print()` calls in implementation code.

### Starter test scaffold (do not modify the test signatures)

```python
import numpy as np
import pytest
from text_generator import TextGenerator

VOCAB_SIZE = 100

def deterministic_logits(token_ids: list[int]) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(VOCAB_SIZE)

def test_generate_returns_correct_length():
    gen = TextGenerator(VOCAB_SIZE, deterministic_logits)
    result = gen.generate([1, 2, 3], max_new_tokens=7)
    assert len(result) == 7

def test_greedy_is_deterministic():
    gen = TextGenerator(VOCAB_SIZE, deterministic_logits)
    r1 = gen.generate([1, 2, 3], max_new_tokens=5, temperature=0.0)
    r2 = gen.generate([1, 2, 3], max_new_tokens=5, temperature=0.0)
    assert r1 == r2

def test_kv_cache_length():
    gen = TextGenerator(VOCAB_SIZE, deterministic_logits)
    prompt = [1, 2, 3]
    gen.generate(prompt, max_new_tokens=4)
    assert len(gen.kv_cache) == len(prompt) + 4
```

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

### Scoring notes

**Independent implementation (40%):** Full credit for a working `generate()` that passes all acceptance criteria without referencing solution code. Partial credit (20%) for a correct structure with one failing criterion. No credit for copied solutions.

**Conceptual understanding (25%):** Graded on the 10 written answers. Each question is worth 2.5 points. Credit requires correct technical content; vague answers score 0.

**Debugging & problem-solving (20%):** Assessed during code review or oral follow-up. Evaluator will introduce one bug (e.g., off-by-one in top-p cumulative sum, missing softmax normalization). Full credit for identifying and fixing it. Partial credit for identifying without fixing.

**Code quality & testing (15%):** Full credit for type hints on all public interfaces, docstrings on class and `generate()`, at least 3 passing tests, no `print()` in implementation. Deduct 5 points per missing element.

Answer key: ../solutions/week-07/assessment.md
