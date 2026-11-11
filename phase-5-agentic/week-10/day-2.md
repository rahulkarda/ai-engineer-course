# Phase 5 · Week 10 · Day 2 — LangChain Prompts, Output Parsers & Memory

> Objective: Build structured, reusable prompt templates and wire typed output parsers into LCEL chains; understand modern state-based memory vs deprecated Memory objects.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

LangChain separates three concerns that are often conflated in naive LLM code:

1. **Prompt construction** — assembling messages from templates and runtime variables.
2. **Output parsing** — converting raw LLM text into typed Python objects.
3. **Memory / conversation history** — deciding which prior turns to include in the next prompt.

Keeping these separate lets you swap any layer independently (e.g., change the model without touching the parser, or change the parser without changing the prompt) and makes each layer independently testable.

---

## How It Works Internally

### Prompt Templates

`ChatPromptTemplate` is the primary abstraction for chat models. It stores a list of `MessageTemplate` objects and formats them at invoke time.

```
ChatPromptTemplate
  └── list[BaseMessagePromptTemplate]
        ├── SystemMessagePromptTemplate   (role=system)
        ├── HumanMessagePromptTemplate    (role=human)
        ├── AIMessagePromptTemplate       (role=ai)
        └── MessagesPlaceholder           (role=any — injects a list of BaseMessages)
```

**Key classes:**

`SystemMessagePromptTemplate.from_template(template: str)` — formats a system message with `{variable}` interpolation.

`HumanMessagePromptTemplate.from_template(template: str)` — formats a human message.

`MessagesPlaceholder(variable_name: str)` — placeholder that accepts a `list[BaseMessage]` at invoke time. This is how conversation history is injected.

**Construction shorthand:**

```python
ChatPromptTemplate.from_messages([
    ("system", "You are {persona}."),
    ("human", "{question}"),
])
```

Tuple shorthand `("role", "template")` is equivalent to the explicit `*MessagePromptTemplate` classes.

**Dynamic variables** — any `{variable_name}` in a template string becomes a required key in the input dict. `ChatPromptTemplate.input_variables` lists them. Missing keys raise a `KeyError` at format time, not silently.

### Output Parsers

Output parsers implement `BaseOutputParser[T]` and expose:

| Method | Purpose |
|---|---|
| `parse(text: str) -> T` | Convert raw string to typed output |
| `get_format_instructions() -> str` | Returns text to append to the prompt telling the LLM how to format its response |

**`StrOutputParser`** — identity parser; strips `AIMessage` wrapper and returns raw string. Use for freeform text.

**`JsonOutputParser`** — parses JSON from the LLM response. Handles markdown code fences (` ```json ... ``` `). Returns `dict[str, Any]`.

**`PydanticOutputParser[T]`** — wraps `JsonOutputParser`, then validates and coerces into a Pydantic model `T`. Provides strongly-typed output with automatic validation errors.

**Integrating format instructions into the prompt:**

```python
parser = PydanticOutputParser(pydantic_object=MyModel)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured data.\n\n{format_instructions}"),
    ("human", "{text}"),
]).partial(format_instructions=parser.get_format_instructions())
```

Calling `.partial()` pre-fills the `format_instructions` variable so it does not need to be supplied at invoke time. The chain becomes `prompt | llm | parser` with only `{"text": ...}` as runtime input.

### Memory: Deprecated Objects vs Modern Patterns

**Deprecated (avoid in new code):**
- `ConversationBufferMemory` — stores all turns verbatim; retrieved via `.load_memory_variables()`.
- `ConversationSummaryMemory` — stores a rolling LLM-generated summary of older turns.
- Both rely on a `memory_key` string to inject into prompts, creating invisible coupling.

**Modern pattern — explicit state management:**

Manage history as a plain `list[BaseMessage]` in your application state. Pass it explicitly to the prompt via `MessagesPlaceholder`. This is the approach used in LangGraph and recommended for all new chains:

```python
history: list[BaseMessage] = []

def chat(question: str) -> str:
    response = chain.invoke({"history": history, "question": question})
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response))
    return response
```

Benefits: no hidden side effects, trivially serialisable, testable with any list fixture.

---

## Worked Example

Extract structured product data from unstructured text using `PydanticOutputParser`.

**Dependencies (add via uv):**
```
uv add langchain langchain-anthropic pydantic
```

**`structured_extraction.py`**

```python
import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic schema for structured output ---

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Numeric rating 1-5", ge=1, le=5)
    pros: list[str] = Field(description="List of positive aspects mentioned")
    cons: list[str] = Field(description="List of negative aspects mentioned")
    sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")


# --- Chain construction ---

def build_extraction_chain() -> Any:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    parser: PydanticOutputParser[ProductReview] = PydanticOutputParser(
        pydantic_object=ProductReview
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a product review analyst. Extract structured data "
                    "from the review text below.\n\n{format_instructions}"
                ),
            ),
            ("human", "Review text:\n\n{review_text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain


# --- Conversation chain with explicit history ---

def build_conversation_chain() -> Any:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


# --- Demo ---

def demo_extraction() -> None:
    chain = build_extraction_chain()
    review = (
        "I've been using the SoundMax Pro headphones for three months. "
        "The noise cancellation is incredible and battery life hits 30 hours easily. "
        "Build quality feels premium. However, the ear cups get warm after an hour "
        "and the companion app crashes on Android. I'd give it a 4 out of 5."
    )
    result: ProductReview = chain.invoke({"review_text": review})
    logger.info("Product: %s", result.product_name)
    logger.info("Rating: %d/5", result.rating)
    logger.info("Pros: %s", result.pros)
    logger.info("Cons: %s", result.cons)
    logger.info("Sentiment: %s", result.sentiment)


def demo_conversation() -> None:
    chain = build_conversation_chain()
    history: list[BaseMessage] = []

    turns: list[str] = [
        "What is the capital of France?",
        "What is its population?",
        "Name one famous landmark there.",
    ]

    for question in turns:
        logger.info("Human: %s", question)
        response: str = chain.invoke({"history": history, "question": question})
        logger.info("AI: %s", response)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response))


def main() -> None:
    logger.info("=== Extraction demo ===")
    demo_extraction()
    logger.info("=== Conversation demo ===")
    demo_conversation()


if __name__ == "__main__":
    main()
```

**Key points:**
- `ProductReview` is a Pydantic v2 `BaseModel` with `Field` descriptions — these flow into `get_format_instructions()`.
- `.partial(format_instructions=...)` pre-fills the system prompt variable so callers only need `{"review_text": ...}`.
- Conversation history is a plain `list[BaseMessage]` — no `Memory` object, no hidden state.
- `MessagesPlaceholder(variable_name="history")` injects the list directly into the prompt.

---

## Exercises

_(Solutions: ../solutions/week-10/day-2.md)_

1. **Nested Pydantic model.** Extend `ProductReview` to include a nested `ReviewerProfile` model (fields: `username: str`, `verified_purchase: bool`, `helpful_votes: int`). Update the prompt and test that the parser correctly populates nested fields from a sample review that includes reviewer metadata.

2. **Retry on parse failure.** `PydanticOutputParser` raises `OutputParserException` when the LLM returns malformed JSON. Wrap the parser step with `parser.with_retry(retry_if_exception_type=(OutputParserException,), stop_after_attempt=2)`. Write a unit test where the LLM mock returns invalid JSON on the first call and valid JSON on the second.

3. **Summary memory simulation.** Implement a `summarise_history` function that accepts `list[BaseMessage]` and returns a single `SystemMessage` containing an LLM-generated summary (use `claude-haiku-5-5`). Integrate it so that when `len(history) > 6`, older messages are replaced by the summary message. Write a test confirming the history list length is bounded.

4. **Multi-output chain.** Build a chain that runs two parsers in parallel on the same LLM response: `StrOutputParser` for the raw text and `PydanticOutputParser` for structured data. Use `RunnableParallel` and a single LLM call. The output type should be `dict[str, str | ProductReview]`. Write a typed wrapper function that returns this dict.

---

## Self-check

- What is the difference between `MessagesPlaceholder` and `HumanMessagePromptTemplate`?
- How does `PydanticOutputParser.get_format_instructions()` affect the LLM's response?
- Why is managing history as `list[BaseMessage]` preferred over `ConversationBufferMemory`?
- What exception does `PydanticOutputParser` raise on malformed output, and how do you handle it in LCEL?
- What does `.partial()` on a `ChatPromptTemplate` do at construction time vs invoke time?

---

## Supplementary Links

- [ChatPromptTemplate reference](https://python.langchain.com/docs/concepts/prompt_templates/)
- [Output parsers overview](https://python.langchain.com/docs/concepts/output_parsers/)
- [PydanticOutputParser how-to](https://python.langchain.com/docs/how_to/output_parser_structured/)
- [MessagesPlaceholder and chat history](https://python.langchain.com/docs/how_to/message_history/)
- [Why deprecate Memory objects](https://python.langchain.com/docs/versions/migrating_memory/)
