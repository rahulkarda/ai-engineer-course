# Lesson 4 — Optimizers: BootstrapFewShot & MIPRO

> Objective: Compile a DSPy program with BootstrapFewShot then MIPRO; understand what each optimizer changes.
> Time: ~60 min · Prereqs: lesson-3.md

---

## Concept

The optimizer — DSPy calls it a **teleprompter** — is what makes DSPy different from a
prompt library. Given:

- A program (set of Signatures + Modules)
- A training set of examples
- A metric

It searches for better prompts and demonstrations than you would write by hand.

Two optimizers cover most real-world use cases:

| Optimizer | What it tunes | Cost |
|-----------|--------------|------|
| `BootstrapFewShot` | Few-shot demonstrations appended to each `Predict` | Low (few LM calls) |
| `MIPRO` | Both demonstrations *and* the instruction text of each `Predict` | High (many LM calls) |

Start with `BootstrapFewShot`. Graduate to `MIPRO` when the baseline is already good
but you need to squeeze out more accuracy.

---

## How it works internally

### BootstrapFewShot

Algorithm:

1. Run the unoptimized program on a subset of the training set.
2. For each example where the program *succeeds* (metric ≥ threshold), record the full
   LM trace — inputs, intermediate reasoning, and outputs — as a demonstration.
3. Select the `k` best demonstrations per `Predict` node.
4. Inject those demonstrations into the program's `demos` list.

The resulting program is the original program but with few-shot examples prepended to
every LM call. The instruction text is unchanged.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,   # max demos injected per Predict
    max_labeled_demos=2,        # also allow ground-truth labeled demos
)
compiled = optimizer.compile(program, trainset=trainset)
```

### MIPRO (Multi-prompt Instruction PRoposal Optimizer)

MIPRO is a Bayesian optimization loop:

1. Generate candidate instruction texts for each `Predict` using an LM.
2. Sample combinations of (instruction, demos) proposals.
3. Evaluate each combination on a validation set using the metric.
4. Select the combination with the highest score.

This is expensive: it runs the program many times. Use `num_trials` to control budget.

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=my_metric,
    auto="light",       # "light" | "medium" | "heavy" — controls num_trials
    num_threads=4,
)
compiled = optimizer.compile(
    program,
    trainset=trainset,
    num_trials=20,
    minibatch_size=8,
)
```

### Save and load compiled programs

```python
compiled.save("compiled_program.json")

fresh = SentimentProgram()
fresh.load("compiled_program.json")
```

The saved JSON contains the optimized instructions and demonstrations — the LM
is not baked in.

### What compilation does NOT change

- Your Python code (forward logic, tool calls, control flow)
- The LM backend or model name
- The Signature field definitions

Compilation only mutates the `demos` and `instructions` attributes of each
internal `Predict` node.

---

## Worked example

```python
"""lesson4_optimizers.py — BootstrapFewShot and MIPRO compilation."""
import logging
import os
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_PATH = Path("compiled_sentiment.json")


# ---------------------------------------------------------------------------
# Signature and program (same as lesson 3)
# ---------------------------------------------------------------------------

class SentimentClassify(dspy.Signature):
    """Classify the sentiment of a product review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="exactly one of: positive, negative, neutral")


class SentimentProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(SentimentClassify)

    def forward(self, review: str) -> dspy.Prediction:
        return self.classify(review=review)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    raw = [
        ("This product is amazing, love it!", "positive"),
        ("Terrible quality, broke after one day.", "negative"),
        ("It works, nothing special.", "neutral"),
        ("Best purchase I've made this year!", "positive"),
        ("Completely disappointed, total waste of money.", "negative"),
        ("Average, does what it says.", "neutral"),
        ("Exceeded my expectations in every way.", "positive"),
        ("Would not recommend to anyone.", "negative"),
        ("Solid product, exactly as described.", "positive"),
        ("Just okay, not worth the price.", "neutral"),
    ]
    examples = [
        dspy.Example(review=r, sentiment=s).with_inputs("review")
        for r, s in raw
    ]
    return examples[:7], examples[7:]   # trainset, devset


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def exact_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    return pred.sentiment.strip().lower() == example.sentiment.strip().lower()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(program: dspy.Module, devset: list[dspy.Example], label: str) -> float:
    ev = Evaluate(
        devset=devset,
        metric=exact_match,
        num_threads=1,
        display_progress=False,
        display_table=False,
    )
    score = ev(program)
    logger.info("[%s] score: %.1f%%", label, score)
    return score


# ---------------------------------------------------------------------------
# Compile with BootstrapFewShot
# ---------------------------------------------------------------------------

def compile_bootstrap(
    program: dspy.Module,
    trainset: list[dspy.Example],
) -> dspy.Module:
    optimizer = BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
    )
    compiled = optimizer.compile(program, trainset=trainset)
    logger.info("Bootstrap compilation complete.")
    return compiled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lm = dspy.LM(
        model="anthropic/claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=128,
    )
    dspy.configure(lm=lm)

    trainset, devset = build_dataset()
    program = SentimentProgram()

    # Baseline
    baseline_score = evaluate(program, devset, "baseline")

    # Compiled
    compiled = compile_bootstrap(program, trainset)
    compiled_score = evaluate(compiled, devset, "bootstrap")

    logger.info(
        "Improvement: %.1f%% → %.1f%%",
        baseline_score,
        compiled_score,
    )

    # Persist the compiled program
    compiled.save(str(SAVE_PATH))
    logger.info("Saved to %s", SAVE_PATH)

    # Reload and verify
    reloaded = SentimentProgram()
    reloaded.load(str(SAVE_PATH))
    reload_score = evaluate(reloaded, devset, "reloaded")
    assert abs(reload_score - compiled_score) < 0.01, "Reload score mismatch"
```

### Inspecting what was learned

After compilation, print the injected demos:

```python
# Access the compiled Predict node
predictor = compiled.classify
print(f"Demos injected: {len(predictor.demos)}")
for i, demo in enumerate(predictor.demos):
    print(f"\nDemo {i+1}:")
    print(f"  review: {demo.get('review', '')[:60]}")
    print(f"  sentiment: {demo.get('sentiment', '')}")
```

---

## Exercises

1. **Inspect demos** — After `compile_bootstrap`, print all injected demos using the
   snippet above. How many were added? Are they drawn from training examples where the
   baseline program succeeded or failed?

2. **Demo count sweep** — Run `BootstrapFewShot` with `max_bootstrapped_demos=1`,
   `2`, `4`. Log the devset score for each. At what count does performance plateau?

3. **Save / load round-trip** — Save the compiled program, then load it into a new
   `SentimentProgram()` instance. Call `evaluate()` on the loaded version. Assert the
   score matches the compiled program within 0.1%.

4. **MIPRO stub** — Write a `compile_mipro` function analogous to `compile_bootstrap`
   but using `MIPROv2` with `auto="light"` and `num_trials=10`. Do **not** run it
   (it requires many LM calls). Instead, write a unit test that mocks
   `MIPROv2.compile` and asserts your function calls it with the right arguments.

5. **Why train/dev split matters** — In your own words (2–3 sentences): why should the
   trainset passed to `compile` and the devset passed to `Evaluate` be disjoint? What
   goes wrong if they overlap?

(Solutions: [solutions/lesson-4.md](solutions/lesson-4.md))

---

## Self-check

- [ ] I can compile a program with `BootstrapFewShot` and explain what it changes
- [ ] I understand the difference between `max_bootstrapped_demos` and `max_labeled_demos`
- [ ] I can save a compiled program to JSON and reload it
- [ ] I know when to use MIPRO over BootstrapFewShot
- [ ] I can inspect the `demos` attribute of a compiled predictor
