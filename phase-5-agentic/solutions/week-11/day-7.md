# Solutions — Week 11 · Day 7 (Project Reference)

See `solutions/week-11/project.md` for the full reference implementation.

This file contains the day-7 integration tips for the week project.

## Integration checklist answers

### Why does the agent use `anthropic:claude-sonnet-5-5` in production but `TestModel` in tests?

`anthropic:claude-sonnet-5-5` is the production model — it calls the real Anthropic API, costs money, and is non-deterministic. `TestModel` is a fake implementation that returns a fixed response without any network calls. Tests must be deterministic and free, so all agent runs in tests use `TestModel` or `unittest.mock` to patch `agent.run`.

### How do I override the model for a single test?

```python
from pydantic_ai.models.test import TestModel

with my_agent.override(model=TestModel()):
    result = await my_agent.run("input", deps=deps)
```

The `override()` context manager is the idiomatic Pydantic AI pattern — it's reset after the `with` block.

### How do I inject a specific tool call result in tests?

`TestModel` doesn't call real tools. To test that a tool works correctly, test the tool function directly:

```python
def test_count_functions_counts_correctly():
    from src.agents.code_review_agent import count_functions
    # count_functions is a plain function — call it directly
    # Note: @agent.tool_plain wraps it; access via agent._function_tools or test the underlying fn
    code = "def f(): pass\ndef g(): pass\n"
    # If the tool is pure, import and test it directly
    result = count_functions_impl(code)  # the underlying function before decoration
    assert result["functions"] == 2
```

To test the underlying function body without Pydantic AI decoration overhead, consider defining the logic in a separate pure function and registering it:

```python
def _count_functions_impl(source_code: str) -> dict[str, int]:
    # ... logic
    pass

@review_agent.tool_plain
def count_functions(source_code: str) -> dict[str, int]:
    """..."""
    return _count_functions_impl(source_code)
```

Then `tests/unit/test_tools.py` imports and tests `_count_functions_impl` directly.

### How should I structure the pyproject.toml?

```toml
[project]
name = "week-11-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic-ai>=0.0.14",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

Note: `pydantic-ai` is declared but not installed — the learner runs `uv sync` to install.
