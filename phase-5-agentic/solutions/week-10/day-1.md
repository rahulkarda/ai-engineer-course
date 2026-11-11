# Solutions — Phase 5 · Week 10 · Day 1

> LangChain Core: LCEL & Runnables

---

## Exercise 1 — Fallback with `.with_fallbacks()`

```python
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_chain_with_fallback() -> Any:
    primary_llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    backup_llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    # Attach fallback BEFORE composing with prompt/parser.
    # with_fallbacks() wraps only the LLM step, so the fallback runs a
    # complete LLM call with the same formatted prompt — not the whole chain.
    llm_with_fallback = primary_llm.with_fallbacks([backup_llm])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant. Answer in 2-3 sentences."),
            ("human", "{question}"),
        ]
    )
    parser = StrOutputParser()

    return prompt | llm_with_fallback | parser


# --- Tests ---

def test_fallback_invoked_on_rate_limit_error() -> None:
    """Primary raises RateLimitError; fallback should be called and succeed."""
    from anthropic import RateLimitError as AnthropicRateLimitError

    chain = build_chain_with_fallback()

    # The primary model is the first runnable inside llm_with_fallback.
    # We patch ChatAnthropic.invoke to raise on the first call and succeed on
    # the second (which is the backup model's invoke).
    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate a rate limit error from the primary
            raise AnthropicRateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={},
            )
        # Second call — backup model succeeds
        from langchain_core.messages import AIMessage
        return AIMessage(content="Fallback response from sonnet.")

    with patch.object(ChatAnthropic, "invoke", side_effect=side_effect):
        result: str = chain.invoke({"question": "What is LCEL?"})

    assert "Fallback" in result
    assert call_count == 2, f"Expected 2 invoke calls, got {call_count}"
    logger.info("Fallback test passed: %s", result)
```

**Why this works:**
`with_fallbacks()` wraps the primary runnable in a `RunnableWithFallbacks`. On any exception from the primary, LangChain retries each fallback in order, passing the same input. Placing the fallback on the LLM step (not the whole chain) means the prompt has already been formatted — only the LLM call is retried with the backup model.

**Pitfalls:**
- Do not call `.with_fallbacks()` on the full chain — the prompt would be re-formatted unnecessarily and you lose visibility into which step failed.
- The exception classes that trigger fallback are configurable via `exceptions_to_handle`; by default any `Exception` is caught. Be explicit in production to avoid masking logic errors.

---

## Exercise 2 — Retry and timeout

```python
import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_chain_with_retry() -> Any:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    # Retry is placed on the LLM runnable, not the full chain.
    #
    # Why: retrying the prompt step is wasteful — it is a pure Python
    # transformation with no network I/O and will not fail transiently.
    # Retrying only the LLM step isolates the retry to the single component
    # that can fail due to transient network errors, rate limits, or timeouts.
    # Retrying the full chain would also re-run the prompt and any pre-LLM
    # lambdas, which wastes time and can produce side effects.
    llm_with_retry = (
        llm
        .with_retry(stop_after_attempt=3)
        .with_config({"run_name": "haiku_with_retry", "timeout": 10})
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant. Answer in 2-3 sentences."),
            ("human", "{question}"),
        ]
    )
    parser = StrOutputParser()

    chain = prompt | llm_with_retry | parser
    return chain


def main() -> None:
    chain = build_chain_with_retry()
    result: str = chain.invoke({"question": "What is RunnablePassthrough?"})
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
```

**Why this works:**
`.with_retry()` wraps the runnable in a `RunnableRetry` that catches specified exceptions (default: all) and re-invokes up to `stop_after_attempt` times with exponential backoff. `.with_config()` merges a `RunnableConfig` onto the runnable so `run_name` appears in LangSmith traces and `timeout` (in seconds) limits the wall-clock time of each attempt.

**Pitfalls:**
- `timeout` in `with_config` is a soft hint passed through `RunnableConfig` — the underlying HTTP client (httpx) must also respect it. If using LangChain's Anthropic integration, pass `timeout` to `ChatAnthropic(timeout=10)` as well for the hard timeout.
- `stop_after_attempt=3` means 1 initial attempt + 2 retries = 3 total. Do not set a high number for token-expensive models — retries multiply cost.

---

## Exercise 3 — Async streaming with `astream`

```python
import asyncio
import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_chain() -> object:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant. Answer in 2-3 sentences."),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


async def main() -> None:
    chain = build_chain()
    question = "How does LCEL handle async operations internally?"

    # ainvoke — single async call, waits for the full response
    logger.info("--- ainvoke ---")
    result: str = await chain.ainvoke({"question": question})
    logger.info("ainvoke result: %s", result)

    # astream — yields string chunks as they arrive; print each token live
    logger.info("--- astream ---")
    chunks: list[str] = []
    async for chunk in chain.astream({"question": question}):
        # Each chunk is a partial string from StrOutputParser
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    print()  # newline after streaming output
    full_response: str = "".join(chunks)
    logger.info("Collected %d chunks, total length: %d", len(chunks), len(full_response))


if __name__ == "__main__":
    asyncio.run(main())
```

**Why this works:**
Every LCEL step implements `astream()` natively. `StrOutputParser.astream()` passes through the `AIMessageChunk.content` strings emitted by `ChatAnthropic.astream()` token by token. `asyncio.run()` boots the event loop and runs `main()` to completion, collecting all chunks into a list before joining. The `async for` loop yields control to the event loop between chunks, allowing other coroutines to run if needed.

**Pitfalls:**
- Never use `asyncio.run()` inside an already-running event loop (e.g., Jupyter). Use `await main()` directly or `loop.run_until_complete()`.
- `astream()` on a `RunnableParallel` streams from the first branch to complete; branches with slower runnables still complete in parallel but their chunks arrive later.

---

## Exercise 4 — Input-validating `RunnableLambda`

```python
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_question(question: str) -> str:
    """Raise ValueError if the question is empty or whitespace-only."""
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    logger.info("Question validated: %r (length=%d)", question[:40], len(question))
    return question


def uppercase_output(text: str) -> str:
    """Convert final string output to uppercase."""
    return text.upper()


def build_chain() -> Any:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant. Answer in 2-3 sentences."),
            ("human", "{question}"),
        ]
    )
    parser = StrOutputParser()

    # RunnableLambda[str, str] — input validator at the front
    validator = RunnableLambda(validate_question)

    # RunnableLambda[str, str] — uppercaser at the back
    uppercaser = RunnableLambda(uppercase_output)

    # validator runs first, then the prompt needs {"question": ...}
    # so we wrap the prompt step to accept the raw string
    def to_prompt_input(question: str) -> dict[str, str]:
        return {"question": question}

    chain = (
        validator
        | RunnableLambda(to_prompt_input)
        | prompt
        | llm
        | parser
        | uppercaser
    )
    return chain


# --- Tests ---

def test_empty_question_raises() -> None:
    chain = build_chain()
    try:
        chain.invoke("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "non-empty" in str(exc)
    logger.info("Empty question test passed.")


def test_output_is_uppercased() -> None:
    """Mock LLM to return a known lowercase string; assert output is uppercased."""
    chain = build_chain()

    mock_response = AIMessage(content="lcel stands for langchain expression language.")

    with patch.object(ChatAnthropic, "invoke", return_value=mock_response):
        result: str = chain.invoke("What is LCEL?")

    assert result == result.upper(), f"Expected uppercase output, got: {result!r}"
    assert "LCEL" in result or "LANGCHAIN" in result
    logger.info("Uppercase test passed: %s", result)


def test_whitespace_only_raises() -> None:
    chain = build_chain()
    try:
        chain.invoke("   ")
        assert False, "Expected ValueError"
    except ValueError:
        pass
    logger.info("Whitespace-only question test passed.")


if __name__ == "__main__":
    test_empty_question_raises()
    test_output_is_uppercased()
    test_whitespace_only_raises()
    logger.info("All tests passed.")
```

**Why this works:**
`RunnableLambda` wraps any Python callable behind the Runnable protocol, making it composable with `|` like any other LCEL step. Placing the validator first ensures bad input is rejected before any downstream steps (prompt formatting, LLM call) execute. The uppercaser at the end is a post-processing step that transforms the final string — both are pure functions, easily testable in isolation. Mocking `ChatAnthropic.invoke` returns a known `AIMessage`, isolating the test from network calls.

**Pitfalls:**
- `RunnableLambda` does not validate types at construction time — only at runtime. Use `mypy`/`pyright` with explicit type annotations for static validation.
- If the lambda raises any exception, LCEL will not retry it unless wrapped in `.with_retry()`. Distinguish user-input errors (like `ValueError` for empty strings) from transient failures (network errors) — only the latter warrant retry.
