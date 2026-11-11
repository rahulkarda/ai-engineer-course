# Phase 5 · Week 11 · Day 2 — Pydantic AI: Tools & Validation

> Objective: Register tools on agents, understand tool schemas, validate tool inputs/outputs, and handle tool errors gracefully.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Tools are functions the LLM can call during a run. In Pydantic AI:

- **Schema is inferred automatically** from type hints and docstrings — no manual JSON schema writing.
- **Tool functions receive `RunContext`** as the first argument, giving them access to deps and run metadata.
- **Return values are validated** — if a tool returns a Pydantic model, it's serialised and sent back to the model as structured JSON.
- **Tool errors can be retried** — raise `ModelRetry` to ask the model to try a different call.

There are two tool registration patterns:
1. `@agent.tool` — full tool with `RunContext` access (can read deps, usage, etc.)
2. `@agent.tool_plain` — stateless tool with no context (pure function, slightly simpler)

## How it works internally

When a tool is registered, Pydantic AI:

1. Inspects the function signature (excluding `RunContext` param).
2. Builds a JSON Schema from the remaining params using Pydantic's type inference.
3. Includes the docstring as the tool `description` sent to the model.
4. When the model emits a tool-call, deserialises the JSON args into the typed params (validation happens here — invalid args raise `ValidationError` which becomes a retry).
5. Calls the function, serialises the return value, appends it to the message list, and re-calls the model.

```
model → tool_call(name="search", args={...})
              ↓
     validate args against schema
              ↓
     call search(ctx, **validated_args)
              ↓
     serialise return → append to messages
              ↓
     model → final answer
```

## Worked example

```python
# src/agents/code_review_agent.py
"""Code review agent with tools for linting and complexity checking."""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

logger = logging.getLogger(__name__)


# ── Result & deps ─────────────────────────────────────────────────────────────
class ReviewResult(BaseModel):
    issues: list[str]
    severity: Annotated[str, Field(pattern=r"^(low|medium|high)$")]
    recommendation: str


@dataclass
class ReviewDeps:
    allowed_imports: list[str] = field(default_factory=list)
    max_complexity: int = 10


# ── Agent ─────────────────────────────────────────────────────────────────────
review_agent: Agent[ReviewDeps, ReviewResult] = Agent(
    model="anthropic:claude-sonnet-5-5",
    deps_type=ReviewDeps,
    result_type=ReviewResult,
    system_prompt=(
        "You are a senior Python code reviewer. "
        "Use tools to analyse the code before producing your review."
    ),
)


# ── Tool 1: parse and count nodes ─────────────────────────────────────────────
@review_agent.tool
def count_functions(ctx: RunContext[ReviewDeps], source_code: str) -> dict[str, int]:
    """Count functions, classes, and import statements in the source code.

    Args:
        source_code: Valid Python source code to analyse.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        # Tell the model the code is invalid — it can adjust its review
        raise ModelRetry(f"SyntaxError in code: {exc}") from exc

    counts = {
        "functions": sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)),
        "classes": sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef)),
        "imports": sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
    }
    logger.debug("count_functions result: %s", counts)
    return counts


# ── Tool 2: check imports against allowlist ───────────────────────────────────
@review_agent.tool
def check_imports(ctx: RunContext[ReviewDeps], source_code: str) -> list[str]:
    """Return a list of disallowed imports found in the source code.

    Args:
        source_code: Valid Python source code to analyse.
    """
    tree = ast.parse(source_code)
    violations: list[str] = []
    allowed = ctx.deps.allowed_imports

    if not allowed:
        return []  # no restriction configured

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in allowed:
                violations.append(node.module)

    return violations


# ── Tool 3: plain tool (no context needed) ────────────────────────────────────
@review_agent.tool_plain
def estimate_complexity(source_code: str) -> int:
    """Estimate cyclomatic complexity as the number of branching statements.

    Args:
        source_code: Valid Python source code.
    """
    tree = ast.parse(source_code)
    branch_nodes = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)
    return sum(1 for n in ast.walk(tree) if isinstance(n, branch_nodes))


# ── Run ───────────────────────────────────────────────────────────────────────
async def review_code(source: str, deps: ReviewDeps) -> ReviewResult:
    logger.info("Starting code review")
    result = await review_agent.run(
        f"Review this Python code:\n\n```python\n{source}\n```",
        deps=deps,
    )
    return result.data


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    sample = """
import os
import eval_bad

def process(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
        else:
            result.append(0)
    return result
"""

    deps = ReviewDeps(allowed_imports=["os", "sys", "pathlib"], max_complexity=5)
    out = asyncio.run(review_code(sample, deps))
    print(out.model_dump_json(indent=2))
```

**Key observations:**
- Docstrings become tool descriptions — write them clearly; the model uses them for tool selection.
- `ModelRetry` tells the framework to feed the error back to the model for a retry attempt.
- `@agent.tool_plain` for pure functions; `@agent.tool` when you need deps or run metadata.
- Tool return types can be primitives, dicts, or Pydantic models — all serialisable.

### Validation on tool inputs

Pydantic AI validates tool call arguments before calling the function. If the model sends a malformed argument:

```python
@review_agent.tool_plain
def analyse_line(
    line_number: Annotated[int, Field(ge=1)],
    code: str,
) -> str:
    """Analyse a single line of code by 1-based line number."""
    lines = code.splitlines()
    if line_number > len(lines):
        raise ModelRetry(f"Line {line_number} does not exist (file has {len(lines)} lines)")
    return lines[line_number - 1]
```

`Field(ge=1)` means the model can never pass `line_number=0` — validation fails and the model gets an error message automatically.

## Exercises

1. Add a `detect_security_issues(source_code: str) -> list[str]` plain tool that checks for `eval(`, `exec(`, `subprocess(shell=True`, and `pickle.load(`. Return a list of detected patterns. Register it on `review_agent`.

2. Modify `check_imports` to raise `ModelRetry` if the source code has a `SyntaxError`. Explain why this is better than returning an empty list.

3. Write a `@review_agent.tool` called `read_file(ctx, path: str) -> str` that reads a file from `ctx.deps.base_dir`. Add `base_dir: pathlib.Path` to `ReviewDeps`. Add input validation that rejects paths containing `..` (directory traversal).

4. Write a pytest test that registers a `TestModel` response and asserts `detect_security_issues` is called when the source contains `eval(`. Use `result.all_messages()` to inspect the tool call messages.

(Solutions: ../solutions/week-11/day-2.md)

## Self-check

- [ ] I can explain the difference between `@agent.tool` and `@agent.tool_plain`.
- [ ] I understand how Pydantic AI derives the tool JSON schema from type hints and docstrings.
- [ ] I can use `ModelRetry` to signal a recoverable tool error.
- [ ] I can add `Annotated[type, Field(...)]` constraints to tool parameters.
- [ ] I can write a test that inspects which tools were called during a run.

## Supplementary links

- [Pydantic AI docs — Tools](https://ai.pydantic.dev/tools/)
- [Pydantic AI docs — ModelRetry](https://ai.pydantic.dev/results/#retries)
- [Pydantic Field validators](https://docs.pydantic.dev/latest/concepts/fields/)
