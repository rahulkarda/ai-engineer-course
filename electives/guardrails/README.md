# Elective: LLM Guardrails — NeMo Guardrails & Guardrails AI

> Module overview · Self-contained · Estimated time: 8–10 hours

## What this module teaches

Production LLM applications fail in ways unit tests miss: jailbreaks, prompt injection,
toxic outputs, hallucinated facts, PII leakage, schema drift. **Guardrails** are
programmable safety layers that intercept, validate, and correct LLM inputs and outputs
before they reach users or downstream systems.

After this module you will be able to:

- [ ] Explain the LLM threat surface and map it to input vs. output guardrails
- [ ] Use Guardrails AI: RAIL specs, built-in validators, Pydantic-based output validation, reask loops
- [ ] Use NeMo Guardrails: Colang flows, rails config, dialogue and topical rails
- [ ] Write custom validators for domain-specific constraints
- [ ] Compose a multi-layer guardrail wrapper in front of a LangGraph agent
- [ ] Measure guardrail latency/cost and make informed trade-offs
- [ ] Test a guardrail layer adversarially with a suite of known-bad inputs

## Prerequisites

- Python 3.12+
- Phase 3 (ML/LLMs) complete, or equivalent LLM API experience
- Familiarity with Pydantic, pytest, and basic LangGraph
- `ANTHROPIC_API_KEY` set in environment

## Dependencies

```bash
uv add guardrails-ai
uv add nemoguardrails
uv add anthropic
uv add pydantic
uv add langchain-anthropic   # for NeMo integration
```

> `guardrails-ai` import: `import guardrails`  
> `nemoguardrails` import: `from nemoguardrails import RailsConfig, LLMRails`

## Lesson index

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [lesson-1.md](lesson-1.md) | Threat surface & guardrail taxonomy | ~45 min |
| 2 | [lesson-2.md](lesson-2.md) | Guardrails AI: RAIL specs & validators | ~60 min |
| 3 | [lesson-3.md](lesson-3.md) | Guardrails AI: Pydantic output validation & reask | ~60 min |
| 4 | [lesson-4.md](lesson-4.md) | NeMo Guardrails: Colang flows & rails config | ~60 min |
| 5 | [lesson-5.md](lesson-5.md) | Custom validators & composing a guardrail layer | ~60 min |
| 6 | [lesson-6.md](lesson-6.md) | Latency/cost trade-offs & adversarial testing | ~60 min |
| P | [project.md](project.md) | Capstone: composable guardrail layer for an agent | ~4 hr |

## Solutions

All exercise answers live under [solutions/](solutions/). Open them only after attempting
the exercises yourself.

## Supplementary links

- Guardrails AI docs: <https://www.guardrailsai.com/docs>
- Guardrails AI GitHub: <https://github.com/guardrails-ai/guardrails>
- NeMo Guardrails docs: <https://docs.nvidia.com/nemo/guardrails/>
- NeMo Guardrails GitHub: <https://github.com/NVIDIA/NeMo-Guardrails>
- OWASP LLM Top 10: <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- Anthropic safety overview: <https://www.anthropic.com/safety>
- Anthropic API docs: <https://docs.anthropic.com>
