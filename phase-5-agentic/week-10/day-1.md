# Phase 5 · Week 10 · Day 1 — LangChain Core: LCEL & Runnables

> Objective: Understand the LangChain Expression Language (LCEL) protocol, compose typed chains with the pipe operator, and wire LangSmith tracing.  ·  Time: ~1 hr  ·  Prereqs: [[week-09/day-7]]

---

## Concept

**LangChain Expression Language (LCEL)** is a declarative composition protocol built into LangChain. It provides a unified interface for building chains of LLM calls, prompt formatting, output parsing, and arbitrary Python logic — all composable, streamable, and introspectable without subclassing or boilerplate.

Why LCEL exists:
- Before LCEL (pre-0.1), LangChain used imperative `Chain` subclasses (`LLMChain`, `SequentialChain`). Every chain had a different calling convention, partial batching was manual, and async required separate implementations.
- LCEL standardises everything behind a single **Runnable protocol**. Composing two runnables with `|` produces a new runnable that automatically handles streaming, batching, async, retries, and fallbacks.
- Type safety: chains validate input/output schemas at construction time, not at runtime.

Key benefits over the imperative pattern:
- Automatic first-class streaming (token-level for LLMs).
- Built-in `batch()` with configurable concurrency.
- Async everywhere — no separate async subclass.
- Native LangSmith tracing without instrumentation code.

---

## How It Works Internally

### The Runnable Protocol

Every LCEL component implements the `Runnable[Input, Output]` interface:

```python
class Runnable(ABC):
    def invoke(self, input: Input, config: RunnableConfig | None = None) -> Output: ...
    def stream(self, input: Input, ...) -> Iterator[Output]: ...
    def batch(self, inputs: list[Input], ...) -> list[Output]: ...
    async def ainvoke(self, input: Input, ...) -> Output: ...
    async def astream(self, input: Input, ...) -> AsyncIterator[Output]: ...
    async def abatch(self, inputs: list[Input], ...) -> list[Output]: ...
```

Every built-in component (`ChatPromptTemplate`, `ChatAnthropic`, `StrOutputParser`, etc.) implements this protocol.

### Pipe Operator `|`

The `|` operator is syntactic sugar for `RunnableSequence`:

```python
chain = prompt | llm | parser
# equivalent to:
chain = RunnableSequence(first=prompt, middle=[llm], last=parser)
```

At construction time, `RunnableSequence` verifies that the output type of step N is compatible with the input type of step N+1.

### Utility Runnables

| Class | Purpose |
|---|---|
| `RunnablePassthrough` | Passes input unchanged; useful for injecting raw input alongside transformed data |
| `RunnableParallel` | Runs multiple runnables on the same input concurrently, returns `dict` of results |
| `RunnableLambda` | Wraps any Python callable as a Runnable |

**`RunnableParallel` example:**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

parallel = RunnableParallel(
    original=RunnablePassthrough(),
    uppercased=RunnableLambda(lambda x: x.upper()),
)
parallel.invoke("hello")
# {"original": "hello", "uppercased": "HELLO"}
```

### Type Checking at Construction Time

LCEL uses Python generics. `ChatPromptTemplate` returns `ChatPromptValue`; `ChatAnthropic` accepts `PromptValue` and returns `AIMessage`. Mismatched types surface as type errors in IDEs and type checkers (mypy/pyright) before any code runs.

### LangSmith Tracing

LangSmith hooks in via the `CallbackManager` embedded in every `RunnableConfig`. When the following env vars are set, every `invoke` / `stream` / `batch` call is automatically traced — no code changes required:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=my-project
```

Each span in LangSmith corresponds to one runnable invocation. Nested chains produce nested spans. Streaming traces show token-level deltas.

---

## Worked Example

Build a typed LCEL chain: user question → prompt → Claude haiku → parsed string output.

**Dependencies (add via uv):**
```
uv add langchain langchain-anthropic
```

**`chain_example.py`**

```python
import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_chain() -> Any:
    """Construct a multi-step LCEL chain with tracing-ready components."""
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise technical assistant. Answer in 2-3 sentences max.",
            ),
            ("human", "{question}"),
        ]
    )

    parser = StrOutputParser()

    # Inject metadata alongside the question using RunnableParallel
    enrich_input = RunnableParallel(
        question=RunnablePassthrough(),
        char_count=RunnableLambda(lambda q: len(q)),
    )

    # Log enriched input before the prompt step
    def log_and_extract(data: dict[str, Any]) -> dict[str, str]:
        logger.info("Input char count: %d", data["char_count"])
        return {"question": data["question"]}

    chain = enrich_input | RunnableLambda(log_and_extract) | prompt | llm | parser
    return chain


def main() -> None:
    chain = build_chain()
    question = "What is the difference between LCEL and the old LLMChain?"
    logger.info("Invoking chain...")

    # Sync invoke
    result: str = chain.invoke(question)
    logger.info("Result: %s", result)

    # Streaming
    logger.info("Streaming result:")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()

    # Batch — two questions concurrently
    questions = [
        "What is RunnablePassthrough?",
        "How does LCEL handle async?",
    ]
    results: list[str] = chain.batch(questions, config={"max_concurrency": 2})
    for q, r in zip(questions, results, strict=True):
        logger.info("Q: %s | A: %s", q, r[:80])


if __name__ == "__main__":
    main()
```

**Key points in the example:**
- `ChatAnthropic` model ID is `claude-haiku-5-5`.
- API key comes from `os.getenv("ANTHROPIC_API_KEY")` — never hardcoded.
- `RunnableParallel` enriches the input with metadata (char count) before the prompt.
- `RunnableLambda` wraps a plain Python function for logging/extraction.
- `chain.stream()` yields string chunks; `chain.batch()` takes a list and returns a list.
- `logging` used throughout — no `print` in production paths.

---

## Exercises

_(Solutions: ../solutions/week-10/day-1.md)_

1. **Extend the chain with a fallback.** Use `.with_fallbacks([backup_llm])` on the `ChatAnthropic` step. Define `backup_llm` as a second `ChatAnthropic` instance using `claude-sonnet-5-5`. Write a test that confirms the fallback is invoked when the primary raises a `RateLimitError` (mock it).

2. **Add retry logic and timeout.** Wrap the LLM step with `.with_retry(stop_after_attempt=3)` and `.with_config({"timeout": 10})`. Explain in a code comment why retry is placed on the LLM runnable rather than the full chain.

3. **Async streaming.** Rewrite `main()` to use `astream` with `asyncio.run()`. Collect all chunks into a list, then join and log. Ensure the function signature is `async def main() -> None` and all awaits are explicit.

4. **Custom RunnableLambda with schema.** Create a `RunnableLambda` that uppercases the final string output and annotates its input/output types using `RunnableLambda[str, str]`. Append it to the chain. Write a unit test asserting the output is fully uppercased regardless of LLM response casing (mock the LLM to return a known string).

---

## Self-check

- Can you name all six methods of the Runnable protocol?
- What does `RunnableParallel` return, and when would you use it over `RunnableSequence`?
- How does LangSmith tracing activate without modifying chain code?
- What Python operator composes two runnables, and what class does it produce?
- Where do type mismatches between LCEL steps surface — construction or runtime?

---

## Supplementary Links

- [LCEL conceptual guide](https://python.langchain.com/docs/concepts/lcel/)
- [Runnable interface reference](https://python.langchain.com/docs/concepts/runnables/)
- [RunnableParallel docs](https://python.langchain.com/docs/how_to/parallel/)
- [LangSmith tracing quickstart](https://docs.smith.langchain.com/observability/quickstart)
- [LangChain LCEL streaming deep-dive](https://python.langchain.com/docs/how_to/streaming/)
