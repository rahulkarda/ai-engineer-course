# Reasoning Models · Lesson 1 — What Are Reasoning Models?

> Objective: Understand extended thinking as a first-class API capability and inspect the thinking trace.
> Time: ~1 hr · Prereqs: Claude Messages API basics (claude-ecosystem lesson-1)

---

## Concept

### Chain-of-thought: from prompt trick to architecture

Early LLM users discovered that appending "Let's think step by step" to a prompt
improved accuracy on math and logic problems. This worked because it induced the model
to produce intermediate reasoning tokens before the final answer — the model's own
output became a scratchpad that influenced its next prediction.

For years this was a *prompt trick*. The model produced both its reasoning and its answer
in a single uncontrolled text stream. You could not separate them, budget them, or verify
them independently.

**Reasoning models** make chain-of-thought a **first-class capability**:

- The model has a dedicated **thinking phase** that runs before the visible response.
- Thinking tokens are **separate from output tokens** in the content block structure.
- You set an explicit **token budget** that controls how long the model is allowed to think.
- The thinking trace is returned as a `thinking` content block you can read and log.

### Extended thinking in Claude

Claude exposes this via the `thinking` parameter in `messages.create()`:

```python
response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "..."}],
)
```

When thinking is enabled:
- `response.content` contains **both** `thinking` blocks and `text` blocks.
- `thinking` blocks have `.thinking` attribute (the raw CoT text).
- `text` blocks contain the final answer.
- The model always produces the thinking block(s) first, then the answer.

### Why does this help?

Standard inference: the model predicts one token at a time, left to right.
Each token is one forward pass. Hard reasoning problems require many logical steps that
don't fit in a single forward pass — the attention window "sees" the problem statement
but has no workspace to derive intermediate conclusions.

Thinking tokens *are* that workspace. Each reasoning step becomes part of the context
for the next step. The model effectively performs multiple forward passes of reasoning
before committing to an answer. More budget = more workspace = harder problems solvable.

This is also why **output length alone does not help** for reasoning. Padding an answer
with verbose prose adds output tokens but not reasoning tokens. Thinking budget is the
lever, not `max_tokens`.

---

## How it works internally

### Content block structure

A response with extended thinking has this shape:

```
response.content = [
    ThinkingBlock(type="thinking", thinking="First I should consider..."),
    TextBlock(type="text", text="The answer is 42."),
]
```

Always iterate blocks and branch on `.type` — never assume position 0 is text:

```python
for block in response.content:
    if block.type == "thinking":
        log.debug("reasoning trace: %s", block.thinking[:200])
    elif block.type == "text":
        answer = block.text
```

### Token accounting

`response.usage` gains a new field when thinking is enabled:

```python
response.usage.input_tokens    # prompt tokens
response.usage.output_tokens   # text answer tokens
# thinking tokens are counted separately in the billing model
```

Thinking tokens count toward `max_tokens`. Set `max_tokens` large enough to cover both
the thinking budget AND the expected answer length:

```
max_tokens ≥ budget_tokens + expected_answer_tokens
```

If `max_tokens` is too small, the model will be cut off mid-thought.

### Minimum budget

The API enforces a minimum `budget_tokens` of **1024**. Below that the model cannot
complete even trivial reasoning chains. Typical useful values: 2000–10000 for hard
problems, up to 100000 for very complex tasks.

---

## Worked example

```python
"""lesson_1_demo.py — Inspect extended thinking output."""
import logging
import os

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def solve_with_thinking(problem: str, budget: int = 5000) -> tuple[str, str]:
    """
    Solve a problem using extended thinking.

    Returns:
        (thinking_trace, answer)
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,   # budget + headroom for the answer
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": problem}],
    )

    thinking_trace = ""
    answer = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_trace = block.thinking
        elif block.type == "text":
            answer = block.text

    log.info(
        "tokens — in: %d  out: %d  thinking_chars: %d",
        response.usage.input_tokens,
        response.usage.output_tokens,
        len(thinking_trace),
    )

    return thinking_trace, answer


def main() -> None:
    problem = (
        "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
        "Show your reasoning carefully."
    )

    trace, answer = solve_with_thinking(problem, budget=2000)

    log.info("=== Thinking trace (first 500 chars) ===")
    log.info("%s", trace[:500])
    log.info("=== Answer ===")
    log.info("%s", answer)


if __name__ == "__main__":
    main()
```

Run it:

```bash
ANTHROPIC_API_KEY=<your-key> python lesson_1_demo.py
```

Key observations:
- `budget + 2000` ensures answer tokens don't get cut off.
- The thinking trace is internal scratch — it may contain false starts and corrections.
- The final `text` block always contains the committed answer.

---

## Exercises

1. **Block inspector** — Write a function `parse_thinking_response(response)` that
   returns a `dict` with keys `"thinking"` (str or `None`) and `"answer"` (str).
   It should handle responses where no `thinking` block is present (thinking disabled).
   Write a pytest test with a mocked `response.content` covering both cases.

2. **Budget effect** — Write a function that calls `solve_with_thinking` with budgets
   `[1024, 3000, 8000]` on the same hard arithmetic problem and logs the thinking trace
   length vs. budget. Do not make real API calls — mock the client to return a stub
   `ThinkingBlock` with length proportional to budget.

3. **max_tokens guard** — Write a `safe_thinking_call(problem, budget)` function that
   raises `ValueError` if `budget < 1024` or if the combined `budget + 500` would
   exceed `max_tokens=16000`. Add a test for each guard condition.

4. **Thinking disabled** — Call the same model with `thinking={"type": "disabled"}` and
   observe whether thinking blocks appear. Log the difference in `output_tokens`.
   (Mock the client; assert that no `thinking` block appears in the mocked path.)

5. **Logging** — Extend `solve_with_thinking` to log a structured dict at `DEBUG` level:
   `{"budget": budget, "thinking_tokens": len(trace), "answer_tokens": len(answer)}`.
   Write a test using `caplog` to assert the log message appears.

(Solutions: [solutions/lesson-1.md](solutions/lesson-1.md))

---

## Self-check

- [ ] I can explain why thinking tokens are a workspace, not just verbose output
- [ ] I can write a `messages.create()` call with extended thinking enabled
- [ ] I can iterate `response.content` safely and extract both thinking and text blocks
- [ ] I know the minimum `budget_tokens` value and why `max_tokens` must exceed the budget
- [ ] I can write a pytest test that mocks the API response with thinking blocks

## Supplementary links

- [Extended thinking guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Thinking content block spec](https://docs.anthropic.com/en/api/messages#body-messages-content-thinking)
- [Chain-of-Thought Prompting (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
