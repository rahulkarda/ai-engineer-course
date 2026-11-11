# Elective: Reasoning Models & Test-Time Compute

> Audience: engineers who understand the Claude Messages API and want to use extended
> thinking and reasoning models effectively in production.
> Duration: 5 self-contained lessons + 1 capstone project · ~1 hr per lesson, ~4 hr project.

## Why this elective

Large language models improve at hard problems when given more time to "think" before
answering. Claude's extended thinking surface exposes this capability directly: you
allocate a token budget for chain-of-thought, and the model produces a visible reasoning
trace before its final answer.

Understanding *when* to pay for this compute, *how* to control the budget, and *how* to
verify the output quality separates engineers who blindly call the API from those who
ship accurate, cost-efficient systems.

## Prerequisites

- Python 3.12+
- `anthropic` SDK installed via `uv add anthropic`
- `ANTHROPIC_API_KEY` set in environment (never hardcoded)
- Pytest basics, `unittest.mock`

## Module map

| # | Lesson | Key concept |
|---|--------|-------------|
| 1 | [What are reasoning models?](lesson-1.md) | Extended thinking, chain-of-thought as first-class capability |
| 2 | [Test-time compute scaling](lesson-2.md) | Why more thinking tokens help, budget control |
| 3 | [When to use reasoning vs fast models](lesson-3.md) | Cost/latency tradeoffs, routing heuristics |
| 4 | [Prompting reasoning models](lesson-4.md) | Less is more, self-consistency, best-of-N |
| 5 | [Verifiers, evals, and agents](lesson-5.md) | Verifier models, evaluating reasoning quality, tool use |
| — | [Capstone project](project.md) | Routing layer: hard → reasoning + best-of-N + verifier; easy → fast |

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

## Model IDs used in this elective

| Role | Model ID |
|------|----------|
| Reasoning (high-capability) | `claude-opus-5-5` |
| Reasoning (balanced) | `claude-sonnet-5-5` |
| Fast / cheap | `claude-haiku-5-5` |

These are the exact strings to pass to `model=`. Never use `-4-x` variants.

## Supplementary links

- [Extended thinking guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Model overview](https://docs.anthropic.com/en/docs/about-claude/models)
- [API reference](https://docs.anthropic.com/en/api/messages)
- [Thinking block reference](https://docs.anthropic.com/en/api/messages#body-messages-content-thinking)
- [Test-time compute scaling (OpenAI o1 system card)](https://openai.com/index/openai-o1-system-card/)
- [Let's Verify Step by Step (Lightman et al. 2023)](https://arxiv.org/abs/2305.20050)
- [Self-Consistency improves CoT (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
