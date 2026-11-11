# Lesson 2 — Modules: Predict, ChainOfThought, ReAct

> Objective: Compose Predict, ChainOfThought, and ReAct into multi-step programs.
> Time: ~60 min · Prereqs: lesson-1.md

---

## Concept

A DSPy **Module** is the equivalent of a PyTorch `nn.Module` for LLM programs: it
encapsulates one or more LM calls, can be composed with other modules, and exposes
parameters (prompts, examples) that the optimizer can tune.

DSPy ships three foundational modules:

| Module | What it does |
|--------|-------------|
| `dspy.Predict` | Single forward pass — signature in, completion out |
| `dspy.ChainOfThought` | Adds a `reasoning` field before the answer; forces scratchpad thinking |
| `dspy.ReAct` | Implements Reason + Act loop; interleaves tool calls with reasoning |

---

## How it works internally

### dspy.Predict

`Predict` is the atomic unit. Internally it:

1. Serializes the Signature fields into a prompt template
2. Appends any few-shot demonstrations stored in `self.demos`
3. Calls the configured LM
4. Parses the response back into the declared output fields

```
[System / task instruction from Signature docstring]
[Few-shot demos, if any]
---
Field1: {value}
Field2: {value}
OutputField: ← model continues here
```

### dspy.ChainOfThought

`ChainOfThought` wraps `Predict` and injects a hidden `reasoning` field before every
declared output field. This implements chain-of-thought without you writing "think
step by step" in your prompt. The reasoning text is stored on the result but is
typically not surfaced to downstream modules.

Prompt structure:

```
...
Reasoning: Let's think step by step. ${reasoning}
Answer: ${answer}
```

The optimizer can later replace "Let's think step by step." with a more task-specific
rationale instruction.

### dspy.ReAct

`ReAct` implements the Reason-Act loop (Yao et al. 2022). You provide:

- A Signature (what the overall task expects)
- A list of `dspy.Tool` objects (functions the model can call)

Internally, ReAct generates `Thought → Action → Observation` triples until either
a finish action is produced or `max_iters` is reached. Each iteration is a separate
LM call.

```
Thought 1: I should look up the capital.
Action 1: lookup_country[France]
Observation 1: France's capital is Paris.
Thought 2: I have the answer.
Action 2: finish[Paris]
```

### Custom Modules

You subclass `dspy.Module` and implement `forward()`:

```python
class MyPipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.ChainOfThought(Step1Sig)
        self.step2 = dspy.Predict(Step2Sig)

    def forward(self, question: str) -> dspy.Prediction:
        r1 = self.step1(question=question)
        r2 = self.step2(context=r1.answer, question=question)
        return r2
```

All `dspy.Predict` / `dspy.ChainOfThought` instances inside a module are
automatically discovered and treated as optimizable parameters.

---

## Worked example

```python
"""lesson2_modules.py — Predict, ChainOfThought, and ReAct."""
import logging
import os
from typing import Any

import dspy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

class QuestionAnswer(dspy.Signature):
    """Answer the question accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ReasonedQA(dspy.Signature):
    """Answer using step-by-step reasoning."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class AgentTask(dspy.Signature):
    """Use available tools to answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# Tools for ReAct
# ---------------------------------------------------------------------------

def lookup_population(country: str) -> str:
    """Return approximate population of a country (stub)."""
    data: dict[str, str] = {
        "germany": "84 million",
        "france": "68 million",
        "japan": "125 million",
    }
    return data.get(country.lower(), "unknown")


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        # restrict to arithmetic only
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "unsafe expression"
        return str(eval(expression))  # noqa: S307 — restricted input
    except Exception as exc:
        return f"error: {exc}"


# ---------------------------------------------------------------------------
# Multi-step pipeline (custom Module)
# ---------------------------------------------------------------------------

class FactCheckPipeline(dspy.Module):
    """Extract a claim, reason about it, then produce a verdict."""

    class _ExtractClaim(dspy.Signature):
        """Extract the main factual claim from a statement."""
        statement: str = dspy.InputField()
        claim: str = dspy.OutputField(desc="a single, testable factual claim")

    class _VerifyClaim(dspy.Signature):
        """Assess whether the claim is likely true or false and explain why."""
        claim: str = dspy.InputField()
        verdict: str = dspy.OutputField(desc="'true', 'false', or 'uncertain'")
        explanation: str = dspy.OutputField(desc="one-sentence justification")

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.Predict(self._ExtractClaim)
        self.verify = dspy.ChainOfThought(self._VerifyClaim)

    def forward(self, statement: str) -> dspy.Prediction:
        extracted = self.extract(statement=statement)
        logger.info("Extracted claim: %s", extracted.claim)
        verified = self.verify(claim=extracted.claim)
        return verified


def demo_predict(lm: dspy.LM) -> None:
    dspy.configure(lm=lm)
    predict = dspy.Predict(QuestionAnswer)
    result = predict(question="What is the capital of Germany?")
    logger.info("[Predict] %s", result.answer)


def demo_cot(lm: dspy.LM) -> None:
    dspy.configure(lm=lm)
    cot = dspy.ChainOfThought(ReasonedQA)
    result = cot(question="If a train travels 120 km in 1.5 hours, what is its speed?")
    logger.info("[CoT reasoning] %s", result.reasoning)
    logger.info("[CoT answer] %s", result.answer)


def demo_react(lm: dspy.LM) -> None:
    dspy.configure(lm=lm)
    tools = [
        dspy.Tool(lookup_population, name="lookup_population",
                  desc="Get approximate population of a country"),
        dspy.Tool(calculate, name="calculate",
                  desc="Evaluate arithmetic expression"),
    ]
    react = dspy.ReAct(AgentTask, tools=tools, max_iters=4)
    result = react(question="What is the combined population of Germany and France?")
    logger.info("[ReAct] %s", result.answer)


def demo_pipeline(lm: dspy.LM) -> None:
    dspy.configure(lm=lm)
    pipe = FactCheckPipeline()
    result = pipe(statement="The Eiffel Tower is located in Berlin.")
    logger.info("[Pipeline verdict] %s — %s", result.verdict, result.explanation)


if __name__ == "__main__":
    lm = dspy.LM(
        model="anthropic/claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )
    demo_predict(lm)
    demo_cot(lm)
    demo_react(lm)
    demo_pipeline(lm)
```

---

## Exercises

1. **Predict vs CoT** — Run both `demo_predict` and `demo_cot` on the same arithmetic
   question. Compare accuracy. Add a `logger.info` that prints the `reasoning` field
   from CoT. What does the scratchpad contain?

2. **Custom tool** — Add a `lookup_capital(country: str) -> str` tool to the ReAct
   demo. Update `AgentTask`'s signature docstring to mention it can use this tool.
   Test it on "What is the capital of the country with a population of 84 million?"

3. **Pipeline composition** — Extend `FactCheckPipeline` with a third step: a
   `Summarize` module that combines the original statement and the explanation into a
   one-sentence correction if the verdict is "false". Return all three fields.

4. **Module introspection** — After instantiating `FactCheckPipeline`, call
   `pipe.named_predictors()`. What does it return? How does this relate to
   the optimizer's ability to tune the pipeline?

5. **Unit test** — Write a pytest test for `FactCheckPipeline.forward` that:
   - Mocks `dspy.Predict.__call__` to return a fixed `claim`
   - Mocks `dspy.ChainOfThought.__call__` to return fixed `verdict` and `explanation`
   - Asserts the pipeline returns both fields without making real LM calls

(Solutions: [solutions/lesson-2.md](solutions/lesson-2.md))

---

## Self-check

- [ ] I can explain when to use Predict vs ChainOfThought vs ReAct
- [ ] I can subclass `dspy.Module` and compose multiple predictors in `forward()`
- [ ] I can wire `dspy.Tool` objects into a ReAct agent
- [ ] I understand what `named_predictors()` exposes and why it matters for optimization
- [ ] I can write unit tests for a DSPy module without real LM calls
