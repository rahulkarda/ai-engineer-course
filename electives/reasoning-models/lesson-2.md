# Reasoning Models · Lesson 2 — Test-Time Compute Scaling

> Objective: Understand why larger thinking budgets improve accuracy and how to control them.
> Time: ~1 hr · Prereqs: [lesson-1.md](lesson-1.md)

---

## Concept

### Training-time vs. test-time compute

Two dials control model capability:

| Dial | What it buys | Cost paid |
|------|-------------|-----------|
| **Training-time compute** | Better weights, wider knowledge | One-time, at model training |
| **Test-time compute** | More reasoning steps per query | Per-request, at inference |

Training-time scaling hit a wall: doubling training FLOPs yields diminishing accuracy
gains, and larger models are expensive to deploy. Test-time scaling offers a new axis:
keep the model fixed and spend more tokens per hard query.

The empirical finding (established across OpenAI o1, DeepSeek-R1, and Claude extended
thinking) is a **consistent accuracy-vs-budget curve**: accuracy on hard benchmarks rises
as thinking budget grows, up to a saturation point specific to the problem difficulty.

### Why thinking tokens improve accuracy

Consider a multi-step math proof. The model must:
1. Identify the right sub-problems.
2. Solve each one.
3. Combine results.
4. Verify the answer.

With zero thinking budget, the model attempts all four steps in a single forward pass
with no explicit intermediate state. Hard problems fail not because the model "doesn't
know" the answer, but because the reasoning chain is too long to fit in one step.

With a thinking budget, the model can:
- Allocate tokens to steps 1–3 before committing to step 4.
- **Backtrack** — if a reasoning path leads nowhere, the model can abandon it and try
  another within the same thinking block.
- **Self-verify** — it can attempt a solution, check it, and correct it before the
  final answer.

This is why the thinking trace sometimes contains phrases like "Wait, that's wrong" or
"Let me reconsider". That backtracking is a feature, not a bug.

### The cost/accuracy curve

Empirically, for a problem of difficulty D:

- Budget too small → model forced to skip steps → low accuracy
- Budget optimal → accuracy peaks
- Budget oversized → marginal accuracy gain, wasted tokens

The optimal budget depends on problem difficulty, not problem length. A one-sentence
hard maths problem may need more budget than a 1000-word easy summarisation task.

---

## How it works internally

### Streaming and thinking

Extended thinking works with streaming, but the `thinking` block arrives in full before
the first text chunk — the model generates the complete reasoning trace, then starts
streaming the answer:

```python
with client.messages.stream(
    model="claude-sonnet-5-5",
    max_tokens=12000,
    thinking={"type": "enabled", "budget_tokens": 8000},
    messages=[{"role": "user", "content": problem}],
) as stream:
    for event in stream:
        if hasattr(event, "type"):
            if event.type == "content_block_start":
                log.debug("block started: %s", event.content_block.type)
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    # incremental thinking content
                    pass
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
```

### Budget as a soft ceiling

`budget_tokens` is a **soft ceiling**, not a hard stop. The model may use fewer tokens
if the problem is easy. It will not exceed the budget. If the problem genuinely requires
more tokens than the budget allows, the model truncates its reasoning and produces a
best-effort answer with the remaining context.

### Multi-turn conversations with thinking

When building a multi-turn conversation that includes thinking, you must pass the
`thinking` blocks back in the history exactly as received — the API requires them to be
present for the model to maintain reasoning consistency across turns:

```python
# After turn 1: response.content contains [ThinkingBlock, TextBlock]
history.append({"role": "assistant", "content": response.content})
# Do NOT strip the thinking block before the next turn
```

---

## Worked example

```python
"""lesson_2_budget_scaling.py — Measure accuracy vs. budget."""
import logging
import os
from dataclasses import dataclass

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Hard arithmetic problem with a known correct answer
PROBLEM = """
Three friends split a bill. Alice pays twice as much as Bob.
Bob pays $15 more than Carol. The total is $120.
How much does each person pay? Show exact amounts.
"""
CORRECT_ALICE = 62.0  # $62
CORRECT_BOB = 31.0    # $31
CORRECT_CAROL = 16.0  # $16 (but actually 120-62-31 = 27... see trace)


@dataclass
class TrialResult:
    budget: int
    thinking_chars: int
    answer: str
    input_tokens: int
    output_tokens: int


def run_trial(client: anthropic.Anthropic, budget: int) -> TrialResult:
    """Run one trial with a given thinking budget."""
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": PROBLEM}],
    )

    thinking_chars = 0
    answer = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_chars = len(block.thinking)
        elif block.type == "text":
            answer = block.text

    return TrialResult(
        budget=budget,
        thinking_chars=thinking_chars,
        answer=answer,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def main() -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    budgets = [1024, 3000, 8000]

    for budget in budgets:
        result = run_trial(client, budget)
        log.info(
            "budget=%d  thinking_chars=%d  in=%d  out=%d",
            result.budget,
            result.thinking_chars,
            result.input_tokens,
            result.output_tokens,
        )
        log.info("answer snippet: %s", result.answer[:150].replace("\n", " "))


if __name__ == "__main__":
    main()
```

Run it:
```bash
ANTHROPIC_API_KEY=<your-key> python lesson_2_budget_scaling.py
```

What to observe:
- `thinking_chars` grows with budget until the problem is "solved".
- Very small budgets produce shorter traces and sometimes wrong answers.
- Once the budget is sufficient, more budget adds no accuracy gain.

---

## Exercises

1. **Budget recommendation function** — Write `recommend_budget(problem: str) -> int`
   that classifies the problem into `easy / medium / hard` using simple heuristics
   (length, presence of maths keywords, number of logical steps stated) and returns
   `1024 / 4000 / 10000` respectively. Unit-test all three cases with no API calls.

2. **Streaming with budget** — Write a `stream_with_thinking(problem, budget)` function
   that streams a response, accumulates the `thinking_delta` text separately from
   `text_delta`, and returns `(thinking: str, answer: str)`. Mock the streaming iterator
   in your test to verify both streams are captured correctly.

3. **Multi-turn thinking** — Write a two-turn conversation function where turn 1 asks a
   hard question and turn 2 asks a follow-up. Verify the history builder includes the
   `thinking` block in the assistant turn. Use a mock to assert the second API call
   receives the thinking block in `messages`.

4. **Token budget arithmetic** — Write a function `max_tokens_needed(budget, max_answer)`
   that returns the minimum safe `max_tokens` value. Add parametrized pytest cases for
   `(budget=2000, max_answer=500)`, `(budget=10000, max_answer=1000)`.

5. **Cost estimator** — Given thinking at `$3/1M` tokens and output at `$15/1M` tokens
   (Sonnet 5.5 example rates), write `estimate_cost(budget_tokens, output_tokens) -> float`
   and test it. Rates should be injected as parameters, not hardcoded, to allow updates.

(Solutions: [solutions/lesson-2.md](solutions/lesson-2.md))

---

## Self-check

- [ ] I can explain the difference between training-time and test-time compute scaling
- [ ] I understand why the thinking trace may contain self-corrections and dead ends
- [ ] I know that `budget_tokens` is a soft ceiling, not a hard stop
- [ ] I can stream a thinking response and separate thinking deltas from text deltas
- [ ] I know to include thinking blocks in multi-turn conversation history

## Supplementary links

- [Extended thinking guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Scaling LLM test-time compute optimally (Snell et al. 2024)](https://arxiv.org/abs/2408.03314)
- [OpenAI o1 system card](https://openai.com/index/openai-o1-system-card/)
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
