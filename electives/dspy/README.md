# Elective: DSPy — Programming (Not Prompting) LLMs

> Module overview · Self-contained · Estimated time: 6–8 hours

## What this module teaches

DSPy reframes LLM development: instead of hand-crafting prompt strings, you write
**typed programs** — Python classes with declared input/output signatures — and let an
**optimizer** find the best prompts and few-shot examples automatically.

After this module you will be able to:

- [ ] Explain the DSPy philosophy and when it outperforms manual prompting
- [ ] Define Signatures and compose Modules (Predict, ChainOfThought, ReAct)
- [ ] Write a metric function and run evaluation
- [ ] Compile a program with BootstrapFewShot and MIPRO
- [ ] Integrate a DSPy-compiled module into an existing LangGraph agent
- [ ] Decide confidently whether a task warrants DSPy or hand-written prompts

## Prerequisites

- Python 3.12+
- Phase 3 (ML/LLMs) complete, or equivalent LLM API experience
- Familiarity with Pydantic and pytest
- `ANTHROPIC_API_KEY` set in environment

## Dependency

```bash
uv add dspy-ai
```

> Note: the package name is `dspy-ai`; the import is `import dspy`.

## Lesson index

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [lesson-1.md](lesson-1.md) | Philosophy & Signatures | ~45 min |
| 2 | [lesson-2.md](lesson-2.md) | Modules: Predict, ChainOfThought, ReAct | ~60 min |
| 3 | [lesson-3.md](lesson-3.md) | Metrics & Evaluation | ~45 min |
| 4 | [lesson-4.md](lesson-4.md) | Optimizers: BootstrapFewShot & MIPRO | ~60 min |
| 5 | [lesson-5.md](lesson-5.md) | Teleprompters & advanced patterns | ~45 min |
| 6 | [lesson-6.md](lesson-6.md) | DSPy vs manual prompting · Agent integration | ~60 min |
| P | [project.md](project.md) | Capstone: optimize a classification pipeline | ~3 hr |

## Solutions

All exercise answers live under [solutions/](solutions/). Open them only after attempting
the exercises yourself.

## Supplementary links

- DSPy paper: <https://arxiv.org/abs/2310.03714>
- DSPy docs: <https://dspy.ai>
- DSPy GitHub: <https://github.com/stanfordnlp/dspy>
- Anthropic API docs: <https://docs.anthropic.com>
