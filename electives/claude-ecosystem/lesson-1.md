# Elective: Claude Ecosystem · Lesson 1 — Messages API Mechanics

> Objective: Understand every structural element of a Messages API request and response.
> Time: ~1 hr · Prereqs: Python 3.12+, `anthropic` SDK installed

## Concept

The Claude Messages API is a **single endpoint** (`POST /v1/messages`) that handles
text, vision, tools, and streaming. Every other feature — caching, batches, files — layers
on top of this one primitive.

A request has five top-level fields you need to internalize:

| Field | Required | Purpose |
|-------|----------|---------|
| `model` | yes | Which Claude to call |
| `max_tokens` | yes | Hard ceiling on output tokens |
| `messages` | yes | Conversation turns (alternating `user`/`assistant`) |
| `system` | no | Operator-level instructions (processed before `messages`) |
| `tools` | no | Tool definitions Claude may invoke |

## How it works internally

**Render order:** `tools` → `system` → `messages`. The model sees them concatenated in this
order; a `cache_control` marker caches everything up to that point in the rendered stream.

**Stop reasons:** The model stops for one of five reasons:

| `stop_reason` | Meaning |
|---------------|---------|
| `end_turn` | Natural completion |
| `max_tokens` | Hit the ceiling you set |
| `stop_sequence` | A custom stop string matched |
| `tool_use` | Claude wants to call a tool — you must respond |
| `pause_turn` | Server paused a long tool turn; re-send to continue |
| `refusal` | Safety classifier declined — check `stop_details` |

**Always check `stop_reason` before reading content.** A `max_tokens` stop may give you
a truncated, syntactically broken response. A `refusal` stop may give you empty content.

**Content blocks:** `response.content` is a list of typed blocks, not a string.
Block types you'll encounter: `text`, `thinking`, `tool_use`, `image`, `tool_result`.
Iterate and branch on `.type` — never assume position 0 is text.

## Worked example

```python
# lesson_1_demo.py
import logging
import os
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def single_turn(user_text: str) -> str:
    """Minimal single-turn call; returns the text response."""
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        system="You are a concise technical assistant.",
        messages=[{"role": "user", "content": user_text}],
    )

    log.info(
        "stop_reason=%s  in=%d  out=%d",
        response.stop_reason,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Refusal: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        log.warning("Response truncated — increase max_tokens")

    return next(b.text for b in response.content if b.type == "text")


def multi_turn() -> list[dict]:
    """Stateless multi-turn: rebuild history each call."""
    history: list[dict] = []

    turns = [
        "My project is named 'Helios'.",
        "What's the project name I just told you?",
    ]

    for user_text in turns:
        history.append({"role": "user", "content": user_text})
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            messages=history,
        )
        assistant_text = next(b.text for b in response.content if b.type == "text")
        history.append({"role": "assistant", "content": assistant_text})
        log.info("assistant: %s", assistant_text[:80])

    return history


def streaming_example(prompt: str) -> str:
    """Streaming call — accumulate the full text."""
    collected: list[str] = []

    with client.messages.stream(
        model="claude-haiku-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text_chunk in stream.text_stream:
            collected.append(text_chunk)

    final = stream.get_final_message()
    log.info("final stop_reason=%s", final.stop_reason)
    return "".join(collected)


if __name__ == "__main__":
    answer = single_turn("Explain what an API is in one sentence.")
    print(answer)

    history = multi_turn()
    print(history[-1]["content"])

    streamed = streaming_example("Count to 5.")
    print(streamed)
```

Key observations:
- `response.content` is a list — always iterate with `b.type == "text"` guard.
- Pass `history` with each stateless call; the API has no server-side session.
- `stream.get_final_message()` gives the complete `Message` object after streaming.

## Exercises

1. **Stop reason guard.** Modify `single_turn()` so that if `stop_reason == "max_tokens"`,
   it automatically retries with `max_tokens` doubled (up to a cap of 4096).
   Write a unit test that injects a mock response with `stop_reason="max_tokens"` to verify
   the retry logic without calling the real API.

2. **Content block types.** Write a function `extract_blocks(response)` that returns a
   `dict[str, list]` keyed by block type (`"text"`, `"thinking"`, etc.).
   Test it with a mocked response containing mixed block types.

3. **Streaming token accounting.** Using `stream.get_final_message()`, log the
   `input_tokens` and `output_tokens` from the streaming response.
   Calculate the estimated cost at Haiku 5.5 rates ($0.10/$0.50 per 1M tokens).

4. **System prompt effect.** Call the API twice with the same user question but
   different system prompts ("reply in French" vs "reply in Spanish").
   Assert the two responses differ. Mock the client to avoid real API calls.

(Solutions: solutions/lesson-1.md)

## Self-check

- [ ] I can explain the five top-level request fields and when each is required.
- [ ] I can list all six stop reasons and describe the correct action for each.
- [ ] I can iterate `response.content` safely without assuming block type or count.
- [ ] I can implement a stateless multi-turn conversation loop.
- [ ] I can stream a response and get the complete `Message` from the stream.

## Supplementary links

- [Messages API reference](https://docs.anthropic.com/en/api/messages)
- [Streaming guide](https://docs.anthropic.com/en/docs/streaming)
- [Content blocks reference](https://docs.anthropic.com/en/api/messages#body-messages-content)
