# Phase 3 · Week 08 · Day 1 — Prompt Engineering Fundamentals

> Objective: Master zero-shot, few-shot, chain-of-thought, and system/role prompting patterns using the Anthropic SDK.  ·  Time: ~1 hr  ·  Prereqs: [[../week-07/day-7]]

---

## Concept

Prompt engineering is the practice of structuring natural language inputs to reliably steer LLM behavior. The model has no persistent state between API calls — every response is conditioned entirely on the tokens in the current context window. That means the prompt *is* your program.

Four foundational techniques:

**Zero-shot prompting** — give the model a task with no examples. Relies entirely on the model's pre-trained knowledge. Works well for clear, unambiguous tasks that the model has seen during training.

**Few-shot prompting** — include 2–5 input/output example pairs before the real query. Demonstrates the desired format and reasoning style without requiring fine-tuning. Each example is an in-context "training sample." The model uses pattern matching across the examples to generalize.

**Chain-of-thought (CoT)** — instruct the model to reason step-by-step before giving a final answer. Standard CoT embeds a reasoning trace in each few-shot example. Zero-shot CoT appends "Think step by step." or "Let's think through this carefully." to the user turn — this alone substantially improves performance on multi-step reasoning tasks.

**System / role prompts** — the `system` parameter in the Claude API sets a persistent persona, behavioral constraints, and output format rules that apply for the entire conversation. Role prompts (e.g., "You are a senior Python engineer…") prime the model's output distribution toward domain-appropriate language and judgment.

---

## How it works internally

**Tokenization and attention.** The model processes prompt text as a sequence of tokens. The transformer's attention mechanism allows every output token to attend to every input token. Longer, more structured prompts give the model more "signal" to attend to when generating a response.

**Few-shot as implicit gradient.** During fine-tuning, gradient updates move model weights toward desired outputs. Few-shot prompting achieves a similar effect without weight updates: the examples shift the probability distribution over next tokens at inference time. The model treats the examples as a mini-distribution to match.

**CoT and scratchpad computation.** Language models generate tokens left-to-right. When you force intermediate reasoning steps, you are effectively providing additional compute budget. The model "thinks" by writing, and the written steps become new context it can attend to when generating the final answer. This is why CoT helps most on tasks where a single forward pass is insufficient (arithmetic, multi-hop reasoning, code planning).

**System prompt placement.** In the Claude API, the `system` string is prepended to the conversation before any `messages`. It has the highest positional weight in attention for most queries. Constraints placed there are harder for user-turn content to override — which is exactly what you want for persona and safety rules.

**Delimiters.** XML tags (`<input>`, `<examples>`, `<query>`), triple-quotes (`"""`), and markdown fences (` ``` `) help the model parse multi-part prompts by introducing clear structural boundaries. Tokens adjacent to a closing tag are strongly associated with the content inside it. This reduces hallucination from prompt-text bleed and improves instruction following.

---

## Worked example

```python
"""
week-08/day-1-prompt-engineering.py

Demonstrates zero-shot, few-shot, and chain-of-thought prompting
using the Anthropic Python SDK.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY="..."
"""

import logging
import os
from typing import Any

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 1024


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client, reading the key from the environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# 1. Zero-shot classification
# ---------------------------------------------------------------------------

def zero_shot_classify(client: anthropic.Anthropic, text: str) -> str:
    """
    Classify the sentiment of `text` without any examples.

    Returns one of: POSITIVE | NEGATIVE | NEUTRAL
    """
    logger.info("Running zero-shot classification.")

    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Classify the sentiment of the user's text. "
        "Reply with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=16,
        system=system_prompt,
        messages=[
            {"role": "user", "content": text},
        ],
    )

    result = message.content[0].text.strip()
    logger.info("Zero-shot result: %s", result)
    return result


# ---------------------------------------------------------------------------
# 2. Few-shot classification
# ---------------------------------------------------------------------------

def build_few_shot_messages(
    examples: list[dict[str, str]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Build a Claude `messages` list that includes labelled examples
    followed by the real query.

    Each example dict must have keys:
        "input"  — the text to classify
        "label"  — the correct classification label
    """
    messages: list[dict[str, Any]] = []

    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["label"]})

    # Append the actual query last
    messages.append({"role": "user", "content": query})
    return messages


def few_shot_classify(client: anthropic.Anthropic, text: str) -> str:
    """
    Classify sentiment using 3 labelled examples (few-shot).
    """
    logger.info("Running few-shot classification.")

    examples: list[dict[str, str]] = [
        {
            "input": "The product exceeded every expectation. Absolutely delighted.",
            "label": "POSITIVE",
        },
        {
            "input": "Delivery was two weeks late and the item arrived broken.",
            "label": "NEGATIVE",
        },
        {
            "input": "The package arrived on the expected date.",
            "label": "NEUTRAL",
        },
    ]

    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Classify the sentiment of the user's text. "
        "Reply with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."
    )

    messages = build_few_shot_messages(examples=examples, query=text)

    message = client.messages.create(
        model=MODEL,
        max_tokens=16,
        system=system_prompt,
        messages=messages,
    )

    result = message.content[0].text.strip()
    logger.info("Few-shot result: %s", result)
    return result


# ---------------------------------------------------------------------------
# 3. Chain-of-thought reasoning
# ---------------------------------------------------------------------------

def cot_reasoning(client: anthropic.Anthropic, problem: str) -> str:
    """
    Solve a multi-step reasoning problem using zero-shot CoT.

    The instruction "Think step by step." causes the model to emit
    intermediate reasoning before the final answer.
    """
    logger.info("Running chain-of-thought reasoning.")

    system_prompt = (
        "You are a careful reasoning assistant. "
        "When given a problem, think step by step before stating your final answer. "
        "Format your response as:\n"
        "<reasoning>\n"
        "  ... your step-by-step reasoning ...\n"
        "</reasoning>\n"
        "<answer>\n"
        "  ... your final answer ...\n"
        "</answer>"
    )

    # Zero-shot CoT: append the trigger phrase to the user message
    user_message = f"{problem}\n\nThink step by step."

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )

    result = message.content[0].text.strip()
    logger.info("CoT response received (%d chars).", len(result))
    return result


# ---------------------------------------------------------------------------
# Main: run all three techniques
# ---------------------------------------------------------------------------

def main() -> None:
    client = get_client()

    # --- Zero-shot ---
    sample_text = "I can't believe how smooth the onboarding was. Great experience!"
    zero_shot_result = zero_shot_classify(client, sample_text)
    print(f"Zero-shot classification: {zero_shot_result}")

    # --- Few-shot ---
    few_shot_result = few_shot_classify(client, sample_text)
    print(f"Few-shot classification:  {few_shot_result}")

    # --- Chain-of-thought ---
    math_problem = (
        "Alice has 3 times as many apples as Bob. "
        "Bob has 4 more apples than Carol. "
        "Carol has 6 apples. "
        "How many apples does Alice have?"
    )
    cot_result = cot_reasoning(client, math_problem)
    print("\nChain-of-thought reasoning:\n")
    print(cot_result)


if __name__ == "__main__":
    main()
```

### What the code demonstrates

| Section | Technique | Key API detail |
|---|---|---|
| `zero_shot_classify` | Zero-shot | Single user message, no examples |
| `build_few_shot_messages` | Few-shot construction | Alternating user/assistant turns |
| `few_shot_classify` | Few-shot | 3 examples injected as message history |
| `cot_reasoning` | Zero-shot CoT | "Think step by step." trigger + XML delimiter output format |

Note the `system` parameter in every call — it sets the persona and output contract independently of the conversation turns. This is the most reliable place to put hard constraints because it is outside the user-controlled turn.

---

## Exercises

1. Write a system prompt for a code-review assistant that enforces strict Python style rules: PEP 8 compliance, type annotations required, no bare `except` clauses, maximum function length of 30 lines. The prompt should also instruct the model to output findings as a numbered list with file line references.

2. Convert a zero-shot math reasoning prompt to a 3-shot CoT prompt. Start from this zero-shot version:
   ```
   User: "A train travels at 60 km/h for 2.5 hours. How far does it travel?"
   ```
   Add 3 worked examples that show step-by-step arithmetic reasoning before the answer. Use the `build_few_shot_messages` pattern from the worked example.

3. Take the following multi-part prompt and rewrite it using XML delimiters to separate the context, instructions, and query. Explain in a comment why each delimiter boundary helps the model:
   ```
   You are a legal assistant. Here is a contract clause: "The vendor shall deliver goods within 30 days of purchase order receipt." The user wants to know: does this clause allow for force majeure delays? Answer formally.
   ```

4. Implement a Python function `build_few_shot_prompt(examples: list[dict], query: str) -> list[dict]` that:
   - Accepts a list of `{"input": str, "label": str}` dicts
   - Returns a valid Claude `messages` list with examples interleaved as user/assistant turns
   - Has full type hints
   - Raises `ValueError` if any example is missing required keys
   - Is covered by at least 2 `pytest` test cases (use `unittest.mock` if needed)

   (Solutions: ../solutions/week-08/day-1.md)

---

## Self-check

- [ ] I can explain the difference between zero-shot and few-shot prompting and name a use case where few-shot is necessary
- [ ] I can explain why chain-of-thought improves performance on multi-step reasoning tasks at the token-generation level
- [ ] I can place a system prompt correctly in the Anthropic SDK `messages.create()` call and explain why it is separate from the `messages` list
- [ ] I can implement `build_few_shot_messages` without reference and describe how alternating user/assistant turns encode examples
- [ ] I can use XML tags as delimiters in a prompt and explain what structural problem they solve
