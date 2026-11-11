# Elective: Claude Ecosystem · Lesson 2 — Model Family & Extended Thinking

> Objective: Choose the right Claude model for a workload and configure adaptive thinking correctly.
> Time: ~1 hr · Prereqs: [[lesson-1.md]]

## Concept

The Claude 5-family has three tiers. Picking the wrong tier is the fastest way to burn
budget on a simple task or produce mediocre output on a hard one.

| Model | ID | Input $/1M | Output $/1M | Best for |
|-------|----|-----------|------------|---------|
| Claude Opus 5.5 | `claude-opus-5-5` | $4.00 | $20.00 | Hard reasoning, long-horizon agentic, code review |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | $2.00 | $10.00 | Everyday coding, enterprise workflows, high-volume production |
| Claude Haiku 5.5 | `claude-haiku-5-5` | $0.10 | $0.50 | Classification, extraction, latency-sensitive routes |

**Default to `claude-opus-5-5` for new tasks.** Downgrade only after measuring that
Sonnet or Haiku matches quality requirements.

## How extended thinking works internally

All three 5.5 models think **adaptively by default** — you never set `budget_tokens`.
The model decides internally how much pre-response reasoning to do, controlled by the
`effort` level you supply.

**The `thinking` parameter:**
- Omit it (or send `{"type": "adaptive"}`) — thinking is on, depth controlled by `effort`.
- `{"type": "disabled"}` — **400 error on all 5.5 models**. You cannot turn it off.
- `budget_tokens` — **400 error on all 5.5 models**. The concept is retired.

**The `output_config.effort` parameter:**

| Level | Use case |
|-------|---------|
| `low` | Classification, simple Q&A, subagents |
| `medium` | Default for Opus 5.5 and Haiku 5.5 |
| `high` | Default for Sonnet 5.5; most coding |
| `xhigh` | Complex multi-step coding, hard math |
| `max` | Correctness-critical tasks (cost matters less) |

**Thinking display:** By default the model's `thinking` blocks have empty `.thinking`
text (`display` defaults to `"omitted"`). Set `display: "summarized"` to receive a
human-readable reasoning summary — useful for debugging and transparency.

**Why effort matters more than model for cost:** A smaller model at high effort often
costs more and produces worse output than a larger model at low effort.
Tune model first, then effort.

## Worked example

```python
# lesson_2_demo.py
import logging
import os
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_TIERS = {
    "fast": "claude-haiku-5-5",
    "balanced": "claude-sonnet-5-5",
    "best": "claude-opus-5-5",
}


def classify(text: str) -> str:
    """Fast, cheap classification — Haiku at low effort."""
    response = client.messages.create(
        model=MODEL_TIERS["fast"],
        max_tokens=64,
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": f"Classify as POSITIVE, NEGATIVE, or NEUTRAL (one word only): {text}",
        }],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def reason_hard(problem: str) -> tuple[str, str]:
    """Hard reasoning with extended thinking — Opus at xhigh effort."""
    response = client.messages.create(
        model=MODEL_TIERS["best"],
        max_tokens=8192,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "xhigh"},
        messages=[{"role": "user", "content": problem}],
    )

    thinking_summary = ""
    answer = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_summary = block.thinking  # non-empty because display="summarized"
        elif block.type == "text":
            answer = block.text

    log.info(
        "Tokens — in: %d  out: %d",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    return thinking_summary, answer


def route_by_complexity(task: str, complexity: str) -> str:
    """Model router: pick tier based on assessed complexity."""
    tier_map = {
        "simple": ("fast", "low"),
        "medium": ("balanced", "high"),
        "complex": ("best", "xhigh"),
    }
    model_key, effort = tier_map.get(complexity, ("best", "high"))

    response = client.messages.create(
        model=MODEL_TIERS[model_key],
        max_tokens=2048,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": task}],
    )
    return next(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    # Classification — fast lane
    sentiment = classify("This product exceeded all my expectations!")
    log.info("Sentiment: %s", sentiment)

    # Hard reasoning — deep lane
    problem = (
        "A farmer has 17 sheep. All but 9 die. "
        "How many sheep are left? Explain your reasoning."
    )
    summary, answer = reason_hard(problem)
    log.info("Thinking summary: %s", summary[:200])
    log.info("Answer: %s", answer)
```

Key observations:
- `thinking` block `.thinking` is empty string unless `display="summarized"`.
- Always pass thinking blocks back in multi-turn conversations unchanged — the API
  needs them to maintain reasoning continuity.
- Never construct model IDs with date suffixes (`claude-opus-5-5-20260401` is wrong).

## Exercises

1. **Cost calculator.** Write a function `estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float`
   using the price table above. Write parametrized tests covering all three models.

2. **Effort sweep.** Using a mocked client, test that `reason_hard()` sends
   `output_config={"effort": "xhigh"}` in the request payload.
   (Hint: capture the kwargs passed to `messages.create`.)

3. **Thinking block extraction.** Given a mocked response with both `thinking` and `text`
   blocks, write a function that returns `(thinking_text, answer_text)`.
   Assert it handles a response with no `thinking` block gracefully (returns `""`).

4. **Model router.** Extend `route_by_complexity` so it accepts a list of tasks and
   routes each independently. Track the total estimated cost across all tasks using
   the mocked usage values.

(Solutions: solutions/lesson-2.md)

## Self-check

- [ ] I can name the three 5.5 model IDs exactly and their per-token prices.
- [ ] I know why `budget_tokens` and `thinking: {type: "disabled"}` cause 400 errors on 5.5 models.
- [ ] I can explain what `output_config.effort` controls and choose the right level for a workload.
- [ ] I can extract `thinking` blocks from a response safely.
- [ ] I can implement a model router that picks tier by task complexity.

## Supplementary links

- [Model overview](https://docs.anthropic.com/en/docs/about-claude/models)
- [Extended thinking guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Effort levels](https://docs.anthropic.com/en/api/messages#body-output_config-effort)
