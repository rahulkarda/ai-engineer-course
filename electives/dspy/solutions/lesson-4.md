# Solutions — Lesson 4: Optimizers

---

## Exercise 1 — Inspect demos after compilation

```python
compiled = compile_bootstrap(program, trainset)

predictor = compiled.classify
print(f"Demos injected: {len(predictor.demos)}")
for i, demo in enumerate(predictor.demos):
    print(f"\nDemo {i+1}:")
    print(f"  review:    {demo.get('review', '')[:60]}")
    print(f"  sentiment: {demo.get('sentiment', '')}")
    print(f"  reasoning: {demo.get('reasoning', '')[:80]}")
```

Demos are drawn from training examples where the **baseline program succeeded** (metric
returned True). BootstrapFewShot only records demonstrations from correct traces —
it assumes that a successful trace is a good example to show the model.

---

## Exercise 2 — Demo count sweep

```python
import logging
from dspy.teleprompt import BootstrapFewShot

logger = logging.getLogger(__name__)

def sweep_demo_count(
    program: dspy.Module,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    counts: list[int] = [1, 2, 4],
) -> dict[int, float]:
    results: dict[int, float] = {}
    for n in counts:
        opt = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=n)
        compiled = opt.compile(program.__class__(), trainset=trainset)
        score = evaluate(compiled, devset, f"demos={n}")
        results[n] = score
        logger.info("max_bootstrapped_demos=%d → %.1f%%", n, score)
    return results
```

On small datasets (8 examples), performance often plateaus at 2–3 demos. More demos
can hurt if training set is small (few diverse examples available).

---

## Exercise 3 — Save/load round-trip

```python
from pathlib import Path
import json

SAVE_PATH = Path("compiled_sentiment.json")

def test_save_load_roundtrip(
    compiled: dspy.Module,
    devset: list[dspy.Example],
) -> None:
    # Save
    compiled.save(str(SAVE_PATH))
    assert SAVE_PATH.exists()

    # Load into fresh instance
    reloaded = SentimentProgram()
    reloaded.load(str(SAVE_PATH))

    compiled_score = evaluate(compiled, devset, "compiled")
    reloaded_score = evaluate(reloaded, devset, "reloaded")

    assert abs(reloaded_score - compiled_score) < 0.01, (
        f"Score mismatch: compiled={compiled_score}, reloaded={reloaded_score}"
    )
```

---

## Exercise 4 — MIPRO stub + unit test

```python
"""pipeline_compile_mipro.py"""
from dspy.teleprompt import MIPROv2
import dspy


def compile_mipro(
    program: dspy.Module,
    trainset: list[dspy.Example],
) -> dspy.Module:
    optimizer = MIPROv2(
        metric=exact_match,
        auto="light",
        num_threads=4,
    )
    return optimizer.compile(
        program,
        trainset=trainset,
        num_trials=10,
        minibatch_size=5,
    )
```

Unit test:

```python
"""test_mipro.py"""
from unittest.mock import MagicMock, patch, call
import dspy
from pipeline_compile_mipro import compile_mipro


def test_compile_mipro_calls_optimizer() -> None:
    mock_compiled = MagicMock()

    with patch("pipeline_compile_mipro.MIPROv2") as MockMIPROv2:
        instance = MagicMock()
        instance.compile.return_value = mock_compiled
        MockMIPROv2.return_value = instance

        program = MagicMock(spec=dspy.Module)
        trainset = [MagicMock(spec=dspy.Example)]

        result = compile_mipro(program, trainset)

    MockMIPROv2.assert_called_once_with(
        metric=exact_match,
        auto="light",
        num_threads=4,
    )
    instance.compile.assert_called_once_with(
        program,
        trainset=trainset,
        num_trials=10,
        minibatch_size=5,
    )
    assert result is mock_compiled
```

---

## Exercise 5 — Why train/dev split matters

If the optimizer sees the same examples it is evaluated on, it can directly memorize
and encode them as demonstrations, producing an inflated score that does not reflect
performance on new inputs. A disjoint dev set ensures evaluation measures
generalization — the program's ability to handle unseen tickets — rather than
in-sample fit. Overlapping splits lead to over-optimistic compilation decisions: you
would choose a program with more demos than necessary, wasting tokens at inference.
