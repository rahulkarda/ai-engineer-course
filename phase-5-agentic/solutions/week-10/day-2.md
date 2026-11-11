# Solutions — Phase 5 · Week 10 · Day 2

> LangChain Prompts, Output Parsers & Memory

---

## Exercise 1 — Nested Pydantic model

```python
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Nested schema ---

class ReviewerProfile(BaseModel):
    username: str = Field(description="Username or display name of the reviewer")
    verified_purchase: bool = Field(description="Whether the reviewer purchased the product on the platform")
    helpful_votes: int = Field(description="Number of other users who found this review helpful", ge=0)


class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Numeric rating 1-5", ge=1, le=5)
    pros: list[str] = Field(description="List of positive aspects mentioned")
    cons: list[str] = Field(description="List of negative aspects mentioned")
    sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")
    reviewer: ReviewerProfile = Field(description="Profile information about the reviewer")


# --- Chain ---

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
                    "You are a product review analyst. Extract ALL structured fields "
                    "including nested reviewer profile from the review text.\n\n"
                    "{format_instructions}"
                ),
            ),
            ("human", "Review text:\n\n{review_text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


# --- Tests ---

def test_nested_reviewer_fields_populated() -> None:
    """Parser should populate nested ReviewerProfile from a mock LLM response."""
    chain = build_extraction_chain()

    # Construct valid JSON that the parser expects
    mock_json = """
    {
      "product_name": "SoundMax Pro",
      "rating": 4,
      "pros": ["Great noise cancellation", "30-hour battery"],
      "cons": ["Ear cups get warm"],
      "sentiment": "positive",
      "reviewer": {
        "username": "audiophile_jane",
        "verified_purchase": true,
        "helpful_votes": 42
      }
    }
    """
    mock_response = AIMessage(content=mock_json)

    with patch.object(ChatAnthropic, "invoke", return_value=mock_response):
        result: ProductReview = chain.invoke(
            {"review_text": "Great headphones. Bought them last month. 42 people found this helpful."}
        )

    assert result.reviewer.username == "audiophile_jane"
    assert result.reviewer.verified_purchase is True
    assert result.reviewer.helpful_votes == 42
    assert result.rating == 4
    logger.info("Nested fields test passed: reviewer=%s", result.reviewer)
```

**Why this works:**
`PydanticOutputParser` calls `get_format_instructions()` which serialises the full JSON schema of `ProductReview` including the nested `ReviewerProfile` object. The LLM receives this schema in the system prompt and returns a JSON blob that Pydantic validates recursively — nested models are instantiated automatically. No additional parsing code is needed.

**Pitfalls:**
- Deeply nested models with many optional fields increase prompt length. Keep field `description` strings concise.
- If `helpful_votes` is omitted by the LLM, Pydantic v2 raises `ValidationError`. Add `default=0` to make it optional.

---

## Exercise 2 — Retry on parse failure

```python
import json
import logging
import os
from typing import Any
from unittest.mock import MagicMock, call, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product")
    rating: int = Field(description="Numeric rating 1-5", ge=1, le=5)
    sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")


def build_chain_with_retry_parser() -> Any:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    parser: PydanticOutputParser[ProductReview] = PydanticOutputParser(
        pydantic_object=ProductReview
    )

    # Wrap the parser with retry so it re-invokes the LLM on parse failure.
    # stop_after_attempt=2 means: 1 initial attempt + 1 retry = 2 total LLM calls.
    # retry_if_exception_type limits retry to parse failures only — other exceptions
    # (network errors, auth failures) are not swallowed.
    parser_with_retry = parser.with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract product review data.\n\n{format_instructions}",
            ),
            ("human", "Review: {review_text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser_with_retry


# --- Tests ---

def test_retry_on_malformed_json() -> None:
    """LLM returns invalid JSON on first call; valid JSON on second. Parser should succeed."""
    chain = build_chain_with_retry_parser()

    valid_json = json.dumps({
        "product_name": "Widget X",
        "rating": 5,
        "sentiment": "positive",
    })

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: return unparseable output
            return AIMessage(content="Sorry, I cannot provide that information.")
        # Second call: valid JSON
        return AIMessage(content=valid_json)

    with patch.object(ChatAnthropic, "invoke", side_effect=side_effect):
        result: ProductReview = chain.invoke({"review_text": "Fantastic widget!"})

    assert result.product_name == "Widget X"
    assert result.rating == 5
    assert call_count == 2, f"Expected 2 LLM calls, got {call_count}"
    logger.info("Retry parser test passed: %s", result)
```

**Why this works:**
`parser.with_retry()` wraps the parser in a `RunnableRetry`. When the LLM output cannot be parsed into the Pydantic model, `PydanticOutputParser` raises `OutputParserException`. The retry wrapper catches this specific exception and re-invokes the full chain from that step, passing the same input — which means the LLM is called again with the same prompt, giving it another chance to produce valid JSON.

**Pitfalls:**
- `with_retry()` on the parser alone re-invokes the parser with the same (broken) input — it does NOT re-call the LLM by itself. To re-call the LLM, apply `with_retry()` to the `llm | parser` sub-chain or use `OutputFixingParser` which explicitly sends the broken output back to the LLM for correction.
- Two total attempts is usually sufficient; more retries multiply cost without improving correctness if the model consistently misformats.

---

## Exercise 3 — Summary memory simulation

```python
import logging
import os
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Summarise when history exceeds this many messages
SUMMARY_THRESHOLD = 6


def summarise_history(
    history: list[BaseMessage],
    model: ChatAnthropic,
) -> str:
    """Use the LLM to compress a list of messages into a single summary string."""
    if not history:
        return ""

    # Build a minimal prompt: show the transcript, ask for a summary
    messages_text = "\n".join(
        f"{msg.__class__.__name__}: {msg.content}" for msg in history
    )
    summary_prompt = (
        "Summarise the following conversation concisely in 2-4 sentences, "
        "preserving key facts and context:\n\n"
        + messages_text
    )

    response: AIMessage = model.invoke([HumanMessage(content=summary_prompt)])
    summary: str = str(response.content)
    logger.info("Generated summary (%d chars) from %d messages", len(summary), len(history))
    return summary


def compress_history_if_needed(
    history: list[BaseMessage],
    model: ChatAnthropic,
    threshold: int = SUMMARY_THRESHOLD,
) -> list[BaseMessage]:
    """
    If history exceeds threshold, replace older messages with a SystemMessage
    containing an LLM-generated summary. Always keeps the most recent 2 messages
    intact to preserve immediate conversational context.
    """
    if len(history) <= threshold:
        return history

    # Summarise everything except the last 2 messages
    to_summarise = history[:-2]
    recent = history[-2:]

    summary_text = summarise_history(to_summarise, model)
    summary_message = SystemMessage(
        content=f"[Conversation summary]: {summary_text}"
    )

    compressed = [summary_message] + recent
    logger.info(
        "Compressed history: %d → %d messages", len(history), len(compressed)
    )
    return compressed


def build_conversation_chain() -> object:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser(), llm


# --- Tests ---

def test_history_length_bounded_after_summary() -> None:
    """After exceeding threshold, compressed history must be <= threshold."""
    mock_llm = MagicMock(spec=ChatAnthropic)
    mock_llm.invoke.return_value = AIMessage(content="Summary of the conversation.")

    # Build a history that exceeds the threshold
    history: list[BaseMessage] = []
    for i in range(8):
        history.append(HumanMessage(content=f"Question {i}"))
        history.append(AIMessage(content=f"Answer {i}"))

    assert len(history) == 16  # 8 turns

    compressed = compress_history_if_needed(history, mock_llm, threshold=6)

    # Should be: 1 SystemMessage summary + last 2 messages = 3
    assert len(compressed) <= 6, f"Expected <=6 messages, got {len(compressed)}"
    assert isinstance(compressed[0], SystemMessage)
    assert "[Conversation summary]" in compressed[0].content
    logger.info("Bounded history test passed: %d messages", len(compressed))


def test_no_compression_below_threshold() -> None:
    """History below threshold should be returned unchanged."""
    mock_llm = MagicMock(spec=ChatAnthropic)

    history: list[BaseMessage] = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
    ]

    result = compress_history_if_needed(history, mock_llm, threshold=6)

    assert result == history
    mock_llm.invoke.assert_not_called()
    logger.info("No-compression test passed.")
```

**Why this works:**
Rather than using the deprecated `ConversationSummaryMemory`, this approach keeps history as a plain `list[BaseMessage]` in the caller's scope. `compress_history_if_needed` is a pure function — it takes the list and the model, produces a compressed list, and never stores state. The caller replaces its `history` reference with the return value. `MessagesPlaceholder` injects the compressed list directly into the prompt, preserving the structural clarity of explicit state management.

**Pitfalls:**
- The 2-message tail kept in `recent` ensures the immediately prior exchange is preserved verbatim, which prevents the model from losing track of the last question it answered.
- `summarise_history` itself makes an LLM call — mock it in tests that focus on the compression logic (not the summary quality) to avoid real API calls and keep tests deterministic.

---

## Exercise 4 — Multi-output chain with `RunnableParallel`

```python
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Schemas ---

class SentimentResult(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")
    confidence: float = Field(description="Confidence score 0.0–1.0", ge=0.0, le=1.0)
    dominant_emotion: str = Field(description="Dominant emotion expressed, e.g. joy, frustration, excitement")


class KeyFacts(BaseModel):
    product_name: str = Field(description="Name of the product mentioned")
    key_features: list[str] = Field(description="Up to 5 key features or facts mentioned about the product")
    price_mentioned: bool = Field(description="Whether a price or cost was mentioned in the text")


# --- Chain factory ---

def build_parallel_extraction_chain() -> Any:
    llm = ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    # Parser 1: sentiment
    sentiment_parser: PydanticOutputParser[SentimentResult] = PydanticOutputParser(
        pydantic_object=SentimentResult
    )

    # Parser 2: key facts
    facts_parser: PydanticOutputParser[KeyFacts] = PydanticOutputParser(
        pydantic_object=KeyFacts
    )

    # Prompt 1: targeted at sentiment extraction
    sentiment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract sentiment data from the review text. Respond with ONLY the JSON object.\n\n"
                "{format_instructions}",
            ),
            ("human", "{review_text}"),
        ]
    ).partial(format_instructions=sentiment_parser.get_format_instructions())

    # Prompt 2: targeted at key facts extraction
    facts_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract key product facts from the review text. Respond with ONLY the JSON object.\n\n"
                "{format_instructions}",
            ),
            ("human", "{review_text}"),
        ]
    ).partial(format_instructions=facts_parser.get_format_instructions())

    # Two independent sub-chains, each making its own LLM call
    sentiment_chain = sentiment_prompt | llm | sentiment_parser
    facts_chain = facts_prompt | llm | facts_parser

    # Run both in parallel on the same input dict {"review_text": ...}
    parallel_chain: RunnableParallel = RunnableParallel(
        sentiment=sentiment_chain,
        key_facts=facts_chain,
    )

    return parallel_chain


def extract_review_data(review_text: str) -> dict[str, SentimentResult | KeyFacts]:
    """
    Run both extraction chains in parallel and return a typed dict.
    The two LLM calls happen concurrently via RunnableParallel.
    """
    chain = build_parallel_extraction_chain()
    result: dict[str, SentimentResult | KeyFacts] = chain.invoke(
        {"review_text": review_text}
    )
    return result


# --- Tests ---

def test_parallel_extraction_returns_both_parsers() -> None:
    """Both parsers should return correctly typed objects."""
    import json

    sentiment_json = json.dumps({
        "sentiment": "positive",
        "confidence": 0.92,
        "dominant_emotion": "satisfaction",
    })
    facts_json = json.dumps({
        "product_name": "AcmePro Blender",
        "key_features": ["1000W motor", "BPA-free jug", "10-speed settings"],
        "price_mentioned": False,
    })

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        # First call is for sentiment prompt, second for facts prompt.
        # RunnableParallel runs them concurrently but in our mock they're
        # sequential. Return different JSON on each call.
        if call_count == 1:
            return AIMessage(content=sentiment_json)
        return AIMessage(content=facts_json)

    with patch.object(ChatAnthropic, "invoke", side_effect=side_effect):
        result = extract_review_data("Great blender, very powerful!")

    assert isinstance(result["sentiment"], SentimentResult)
    assert isinstance(result["key_facts"], KeyFacts)
    assert result["sentiment"].sentiment == "positive"
    assert result["key_facts"].product_name == "AcmePro Blender"
    assert call_count == 2
    logger.info(
        "Parallel extraction test passed: sentiment=%s, product=%s",
        result["sentiment"].sentiment,
        result["key_facts"].product_name,
    )
```

**Why this works:**
`RunnableParallel` fans the same input dict out to all branches and executes them concurrently (using `asyncio` in async mode, or threads in sync mode). Each branch is an independent `prompt | llm | parser` chain, so both LLM calls and both parsing steps are isolated. The result is a `dict` keyed by the branch names passed to `RunnableParallel`. Separate prompts per branch produce cleaner, more focused LLM responses than asking a single prompt to produce both outputs at once.

**Pitfalls:**
- `RunnableParallel` in sync `invoke()` uses a `ThreadPoolExecutor` internally — branches run in separate threads. Shared mutable state inside a branch lambda is not thread-safe.
- Using two LLM calls doubles token cost versus a single call with a combined output schema. If cost is critical, design a single Pydantic model combining both outputs and parse once.
