# Elective: Computer Use & Browser Agents

> Module overview · Self-contained · Estimated time: 10–12 hours

## What this module teaches

Modern AI agents can drive a keyboard and mouse, navigate web pages, and complete
multi-step tasks in real browsers — without scraping HTML or writing brittle CSS
selectors. This module covers the **computer-use paradigm** end-to-end: the
perception-reason-action loop, the Anthropic computer-use tool, browser automation
with Playwright, and — critically — the safety architecture needed before you let
an agent touch anything real.

After this module you will be able to:

- [ ] Explain the screenshot → reason → action loop and its failure modes
- [ ] Describe the Anthropic computer-use tool API: inputs, outputs, action types
- [ ] Build browser-automation tools with Playwright and expose them to an agent
- [ ] Compare DOM-based vs. vision-based grounding and choose the right approach
- [ ] Define an action space (click, type, scroll, navigate, screenshot) with typed schemas
- [ ] Ground vision-based actions to pixel coordinates and DOM-based actions to selectors
- [ ] Design a safety layer: action allowlist, confirmation gate, sandboxed browser
- [ ] Add verification steps (post-action screenshots, DOM assertions) and retry logic
- [ ] Evaluate a web agent: success rate, turn efficiency, safety violation rate
- [ ] Test the entire stack with mocked LLM and mocked browser — no real API calls

## Prerequisites

- Python 3.12+
- Phase 3 (ML/LLMs) complete, or equivalent LLM API experience
- Pydantic, pytest, logging familiarity
- `ANTHROPIC_API_KEY` set in environment (only needed for integration; all exercises mock)

## Dependencies

```bash
uv add anthropic
uv add playwright
uv add pydantic
uv add pytest pytest-asyncio
```

> Install Playwright browsers (once, outside tests):
> ```bash
> playwright install chromium
> ```
> Tests **never** call a real browser — they mock the Playwright API.

## Lesson index

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [lesson-1.md](lesson-1.md) | The computer-use paradigm & perception-reason-action loop | ~50 min |
| 2 | [lesson-2.md](lesson-2.md) | Anthropic computer-use tool: API, action types, agentic loop | ~60 min |
| 3 | [lesson-3.md](lesson-3.md) | Browser automation with Playwright as agent tools | ~60 min |
| 4 | [lesson-4.md](lesson-4.md) | DOM vs. vision grounding & action spaces | ~60 min |
| 5 | [lesson-5.md](lesson-5.md) | Safety, sandboxing & human-approval gates | ~70 min |
| 6 | [lesson-6.md](lesson-6.md) | Reliability: verification, retries & evaluating web agents | ~60 min |
| P | [project.md](project.md) | Capstone: sandboxed browser agent with allowlist + approval gate | ~4–5 hr |

## Safety notice

Computer-use agents can take **irreversible real-world actions**: submit forms, delete
files, make purchases, send messages. Every lesson in this module repeats this warning
in context. The project enforces it structurally. Never run a computer-use agent against
a production environment without an explicit confirmation gate and action allowlist.

## Solutions

All exercise answers live under [solutions/](solutions/). Open them only after attempting
the exercises yourself.

## Supplementary links

- [Anthropic computer use docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)
- [Anthropic computer use API reference](https://docs.anthropic.com/en/api/messages)
- [Playwright Python docs](https://playwright.dev/python/docs/intro)
- [Playwright async API](https://playwright.dev/python/docs/api/class-playwright)
- [WebArena benchmark](https://webarena.dev/)
- [VisualWebArena benchmark](https://jykoh.com/vwa)
- [Mind2Web dataset](https://osu-nlp-group.github.io/Mind2Web/)
- [SeeAct: GPT-4V web agent paper](https://arxiv.org/abs/2401.01614)
- [Anthropic safety overview](https://www.anthropic.com/safety)
