# Elective: The Claude / Anthropic Ecosystem In Depth

> Audience: engineers who already use the `anthropic` Python SDK and want production-grade mastery.
> Duration: 7 self-contained lessons + 1 capstone project · ~1 hr per lesson, ~4 hr project.

## Why this elective

The Claude API has more moving parts than a single `messages.create()` call suggests.
Understanding the full surface — caching mechanics, batch economics, the Files API, tool-use contracts,
and cross-provider deployment — separates engineers who "use Claude" from engineers who ship
cost-efficient, observable, production-grade pipelines.

## Prerequisites

- Python 3.12+
- `anthropic` SDK installed via `uv add anthropic` (or `pip install anthropic`)
- `ANTHROPIC_API_KEY` set in environment (never hardcoded)
- Pytest basics

## Module map

| # | Lesson | Key concept |
|---|--------|-------------|
| 1 | [Messages API mechanics](lesson-1.md) | system/messages/stop reasons/streaming |
| 2 | [Model family & extended thinking](lesson-2.md) | opus/sonnet/haiku tradeoffs, adaptive thinking |
| 3 | [Prompt caching](lesson-3.md) | cache_control, prefix stability, savings math |
| 4 | [Message Batches API](lesson-4.md) | async bulk processing, ~50% cost savings |
| 5 | [Files API](lesson-5.md) | vision, PDF, reuse across requests |
| 6 | [Token counting & usage tracking](lesson-6.md) | count_tokens, cost attribution, budgets |
| 7 | [Tool use & server-side tools](lesson-7.md) | tool loop, web search, code execution |
| — | [Capstone project](project.md) | cost-optimised batch doc-processing pipeline |

## Setup

```bash
uv add anthropic
# or: pip install anthropic
```

```python
import os
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

All tests in this module **mock** API calls — no real tokens spent running the test suite.

## Supplementary links

- [API reference](https://docs.anthropic.com/en/api)
- [Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Model overview](https://docs.anthropic.com/en/docs/about-claude/models)
- [Prompt caching guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Message Batches guide](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
- [Files API guide](https://docs.anthropic.com/en/docs/build-with-claude/files)
