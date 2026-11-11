# Elective: Claude Ecosystem · Lesson 4 — Message Batches API

> Objective: Process thousands of documents asynchronously at ~50% cost using the Batches API.
> Time: ~1 hr · Prereqs: [[lesson-1.md]], [[lesson-3.md]]

## Concept

The Message Batches API processes requests **asynchronously** at 50% of standard token prices.
You submit a batch, poll until it finishes (usually < 1 hour, max 24 hours), then stream results.

Use it when:
- Latency doesn't matter (nightly jobs, bulk enrichment, offline analysis).
- You have more than a handful of items to process.
- Cost matters more than speed.

Do **not** use it for:
- Real-time user-facing requests (use the regular Messages API).
- Tasks that require one result before computing the next.

**Key limits:**
- Up to 100,000 requests per batch, or 256 MB total.
- Results available for 29 days after batch creation.
- Results arrive in **any order** — key by `custom_id`, never by position.

## How it works internally

Each batch request is a `Request(custom_id, params)` object. The `custom_id` is your
correlation handle — it's what you use to match results to inputs.

**Batch lifecycle:**

```
create() → processing_status="in_progress"
    ↓  (poll with retrieve())
    ↓  processing_status="ended"
    ↓
results() → stream of BetaMessageBatchIndividualResponse objects
    each: .custom_id + .result.type ("succeeded"|"errored"|"canceled"|"expired")
```

**Error handling per result:**
- `succeeded`: read `.result.message.content`.
- `errored` with `invalid_request`: bad request params — fix and resubmit.
- `errored` with `overloaded`: transient — safe to resubmit.
- `expired`: batch hit the 24h limit — resubmit.

**Combining caching + batches:** Put a `cache_control` marker on a shared system prompt.
All requests in the batch referencing the same stable prefix will share the cache entry,
cutting your effective cost even further.

## Worked example

```python
# lesson_4_demo.py
import logging
import os
import time
from dataclasses import dataclass

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SHARED_SYSTEM = [
    {
        "type": "text",
        "text": "You are a precise sentiment analyser. Reply with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.",
        "cache_control": {"type": "ephemeral"},
    }
]


@dataclass
class BatchResult:
    custom_id: str
    sentiment: str | None
    error: str | None


def create_sentiment_batch(texts: list[str]) -> str:
    """Submit a batch; return the batch ID."""
    requests = [
        Request(
            custom_id=f"item-{i}",
            params=MessageCreateParamsNonStreaming(
                model="claude-haiku-5-5",
                max_tokens=16,
                output_config={"effort": "low"},
                system=SHARED_SYSTEM,
                messages=[{"role": "user", "content": text}],
            ),
        )
        for i, text in enumerate(texts)
    ]

    batch = client.messages.batches.create(requests=requests)
    log.info("Batch created: %s  status=%s", batch.id, batch.processing_status)
    return batch.id


def wait_for_batch(batch_id: str, poll_interval: int = 30) -> None:
    """Poll until the batch ends."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        log.info(
            "status=%s  processing=%d  succeeded=%d  errored=%d",
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)


def collect_results(batch_id: str) -> list[BatchResult]:
    """Stream results and return a list of BatchResult objects."""
    results: list[BatchResult] = []

    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            msg = item.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            results.append(BatchResult(custom_id=item.custom_id, sentiment=text.strip(), error=None))
        else:
            error_type = getattr(item.result, "error", item.result.type)
            log.warning("Failed item %s: %s", item.custom_id, error_type)
            results.append(BatchResult(custom_id=item.custom_id, sentiment=None, error=str(error_type)))

    # Sort by original order using the numeric suffix in custom_id
    results.sort(key=lambda r: int(r.custom_id.split("-")[1]))
    return results


def batch_pipeline(texts: list[str]) -> list[BatchResult]:
    """Full end-to-end pipeline."""
    batch_id = create_sentiment_batch(texts)
    wait_for_batch(batch_id)
    return collect_results(batch_id)


if __name__ == "__main__":
    sample_texts = [
        "This is the best purchase I've ever made!",
        "Completely useless, returning it tomorrow.",
        "It arrived on time and does what it says.",
        "I'm not sure how I feel about this product.",
    ]

    results = batch_pipeline(sample_texts)
    for r, original in zip(results, sample_texts):
        log.info("%s → %s", original[:50], r.sentiment or r.error)
```

Key observations:
- `custom_id` is your only link between inputs and outputs — make it meaningful.
- Results arrive out of order from the API; re-sort by your own key.
- Shared system prompts with `cache_control` work inside batches — extra savings.
- Poll interval of 30+ seconds is appropriate; most batches complete within minutes.

## Exercises

1. **Batch status tracker.** Write a function that takes a batch ID and returns a
   `dict` with keys `succeeded`, `errored`, `expired`, `processing`.
   Test it with a mocked `request_counts` object.

2. **Retry on error.** Extend `collect_results` to separate `BatchResult` objects into
   `succeeded` and `failed` lists. Write a `resubmit_failed` function that creates a new
   batch containing only the failed items. Mock the client to test without real API calls.

3. **Cost comparison.** Given 1000 texts with an average of 200 input tokens and 10 output
   tokens each on Haiku 5.5, calculate:
   (a) the cost without batching (standard prices).
   (b) the cost with batching (50% discount).
   (c) the additional savings if a 150-token system prompt is cached across all requests.
   Write this as a pure function with a pytest test.

4. **Large batch chunker.** The Batches API accepts up to 100,000 requests per batch.
   Write a `chunk_batch(items, max_size=100_000)` generator that splits a list of items
   into chunks. Test edge cases: empty list, exactly `max_size` items, `max_size + 1` items.

(Solutions: solutions/lesson-4.md)

## Self-check

- [ ] I can explain when to use batches vs. the real-time Messages API.
- [ ] I understand the batch lifecycle and the four result types.
- [ ] I can construct `Request` objects and correlate results by `custom_id`.
- [ ] I can combine prompt caching with batches for maximum savings.
- [ ] I can calculate the cost savings of batching for a given workload.

## Supplementary links

- [Message Batches guide](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
- [Batches API reference](https://docs.anthropic.com/en/api/creating-message-batches)
- [Python batches example](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/batch_processing.py)
