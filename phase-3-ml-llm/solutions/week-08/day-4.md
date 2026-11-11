# Day 4 Solutions — Reliability, Cost Estimation, and Budget Guards

## Exercise 1: Repair Loop Implementation

```python
# reliability/repair_loop.py
"""Retry / repair loop for structured LLM output extraction."""

from __future__ import annotations

import logging
import os
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from parser import LLMParseError, parse_llm_json

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_REPAIR_ATTEMPTS = 3


class ExtractionFailedError(Exception):
    """Raised when all repair attempts are exhausted without a valid response."""

    def __init__(self, message: str, attempts: int, last_raw: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw


def _build_repair_prompt(
    original_prompt: str,
    bad_response: str,
    error_message: str,
) -> str:
    """Build a follow-up prompt asking the model to fix a malformed response."""
    return f"""Your previous response could not be parsed. Please fix it.

<original_request>
{original_prompt}
</original_request>

<your_previous_response>
{bad_response}
</your_previous_response>

<parse_error>
{error_message}
</parse_error>

Return ONLY a valid JSON object with no markdown fences, no prose, and no explanation."""


def extract_with_repair(
    prompt: str,
    output_model: type[T],
    system_prompt: str = "You are a data extraction assistant. Return valid JSON only.",
    model: str = "claude-opus-5-5",
    max_tokens: int = 1024,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> tuple[T, int]:
    """Call Claude and retry up to max_attempts times if parsing fails.

    On each failed attempt, sends a repair prompt showing the model its
    previous bad output and the parse error, then asks it to try again.

    Args:
        prompt: The initial user prompt.
        output_model: Pydantic model to parse the response into.
        system_prompt: System-level instruction for the LLM.
        model: Claude model ID.
        max_tokens: Max tokens per API call.
        max_attempts: Total attempts before raising ExtractionFailedError (default 3).

    Returns:
        Tuple of (validated_instance, number_of_attempts_used).

    Raises:
        ExtractionFailedError: If all attempts fail.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    current_prompt = prompt
    last_raw: str = ""

    for attempt in range(1, max_attempts + 1):
        logger.info(
            "extract_with_repair: attempt %d/%d for model %s",
            attempt,
            max_attempts,
            output_model.__name__,
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": current_prompt}],
        )

        last_raw = response.content[0].text
        logger.debug("Raw response (attempt %d): %r", attempt, last_raw[:300])

        try:
            result = parse_llm_json(last_raw, output_model)
            logger.info("Parsed successfully on attempt %d", attempt)
            return result, attempt
        except LLMParseError as exc:
            logger.warning(
                "Parse failed on attempt %d: %s", attempt, exc
            )
            if attempt < max_attempts:
                current_prompt = _build_repair_prompt(
                    original_prompt=prompt,
                    bad_response=last_raw,
                    error_message=str(exc),
                )

    raise ExtractionFailedError(
        f"Failed to extract {output_model.__name__} after {max_attempts} attempt(s).",
        attempts=max_attempts,
        last_raw=last_raw,
    )
```

---

## Exercise 2: Cost Estimator for claude-opus-5-5

```python
# reliability/cost.py
"""Token-based cost estimation for Anthropic API calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pricing for claude-opus-5-5 as of the time of writing
# https://www.anthropic.com/pricing
OPUS_PRICE_INPUT_PER_M_TOKENS: float = 15.00   # USD per 1 million input tokens
OPUS_PRICE_OUTPUT_PER_M_TOKENS: float = 75.00  # USD per 1 million output tokens


@dataclass
class TokenUsage:
    """Accumulated token usage across one or more API calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate usage from a single API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        logger.debug(
            "Added usage: +%d input, +%d output (totals: %d / %d)",
            input_tokens,
            output_tokens,
            self.input_tokens,
            self.output_tokens,
        )


@dataclass
class CostEstimate:
    """Breakdown of estimated API cost."""

    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float

    def __str__(self) -> str:
        return (
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
            f"Cost: ${self.input_cost_usd:.4f} in + ${self.output_cost_usd:.4f} out "
            f"= ${self.total_cost_usd:.4f} total"
        )


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_m: float = OPUS_PRICE_INPUT_PER_M_TOKENS,
    output_price_per_m: float = OPUS_PRICE_OUTPUT_PER_M_TOKENS,
) -> CostEstimate:
    """Estimate API cost for a given token count.

    Args:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        input_price_per_m: Price per million input tokens (default: opus-4-5 rate).
        output_price_per_m: Price per million output tokens (default: opus-4-5 rate).

    Returns:
        CostEstimate with input, output, and total costs.
    """
    input_cost = (input_tokens / 1_000_000) * input_price_per_m
    output_cost = (output_tokens / 1_000_000) * output_price_per_m
    total = input_cost + output_cost

    logger.debug(
        "Cost estimate: %d in tokens @ $%.2f/M = $%.6f, "
        "%d out tokens @ $%.2f/M = $%.6f, total = $%.6f",
        input_tokens,
        input_price_per_m,
        input_cost,
        output_tokens,
        output_price_per_m,
        output_cost,
        total,
    )

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total,
    )


def estimate_cost_from_usage(usage: TokenUsage) -> CostEstimate:
    """Convenience wrapper that estimates cost from a TokenUsage accumulator."""
    return estimate_cost(usage.input_tokens, usage.output_tokens)
```

---

## Exercise 3: `BudgetExceededError` and `max_budget_usd` Guard

```python
# reliability/budget.py
"""Budget guard: track cumulative cost and raise when a cap is exceeded."""

from __future__ import annotations

import logging
import os
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from parser import LLMParseError, parse_llm_json
from reliability.cost import TokenUsage, estimate_cost_from_usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BudgetExceededError(Exception):
    """Raised when the cumulative API cost would exceed the configured budget cap."""

    def __init__(
        self,
        message: str,
        spent_usd: float,
        budget_usd: float,
    ) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd


class BudgetedClient:
    """Thin wrapper around anthropic.Anthropic that enforces a USD spend cap.

    Tracks cumulative token usage across all calls made through this instance.
    Before each call, checks whether the projected total would exceed the budget.

    Usage:
        client = BudgetedClient(max_budget_usd=0.50)
        response = client.create_message(model="claude-opus-5-5", ...)
    """

    def __init__(
        self,
        max_budget_usd: float,
        model: str = "claude-opus-5-5",
    ) -> None:
        if max_budget_usd <= 0:
            raise ValueError(f"max_budget_usd must be positive; got {max_budget_usd}")
        self.max_budget_usd = max_budget_usd
        self.model = model
        self._usage = TokenUsage()
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info(
            "BudgetedClient created: max_budget_usd=%.4f model=%s",
            max_budget_usd,
            model,
        )

    @property
    def spent_usd(self) -> float:
        """Current cumulative spend in USD."""
        return estimate_cost_from_usage(self._usage).total_cost_usd

    @property
    def remaining_usd(self) -> float:
        """Remaining budget in USD."""
        return self.max_budget_usd - self.spent_usd

    def _check_budget_before_call(self) -> None:
        """Raise BudgetExceededError if the budget is already exhausted."""
        current_spend = self.spent_usd
        if current_spend >= self.max_budget_usd:
            raise BudgetExceededError(
                f"Budget exhausted: spent ${current_spend:.4f} of ${self.max_budget_usd:.4f}",
                spent_usd=current_spend,
                budget_usd=self.max_budget_usd,
            )

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> anthropic.types.Message:
        """Create a message, track usage, and enforce the budget cap.

        Raises:
            BudgetExceededError: If the budget cap is already reached before the call.
        """
        self._check_budget_before_call()

        logger.info(
            "BudgetedClient: calling API (spent=%.4f / budget=%.4f)",
            self.spent_usd,
            self.max_budget_usd,
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

        self._usage.add(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        logger.info(
            "API call complete: +%d in / +%d out tokens. "
            "Total spend: $%.4f (%.1f%% of budget)",
            response.usage.input_tokens,
            response.usage.output_tokens,
            self.spent_usd,
            100.0 * self.spent_usd / self.max_budget_usd,
        )

        return response
```

---

## Exercise 4: Complete Test Suite for Repair Loop

```python
# tests/test_repair_loop.py
"""Unit tests for the repair loop covering all three cases."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest
from pydantic import BaseModel

from reliability.repair_loop import ExtractionFailedError, extract_with_repair
from reliability.cost import estimate_cost, CostEstimate, TokenUsage, estimate_cost_from_usage
from reliability.budget import BudgetedClient, BudgetExceededError


# ---------------------------------------------------------------------------
# Minimal test model
# ---------------------------------------------------------------------------

class SimpleOutput(BaseModel):
    name: str
    value: int


VALID_JSON = json.dumps({"name": "test", "value": 42})
INVALID_JSON = "This is not JSON."
INVALID_SCHEMA_JSON = json.dumps({"name": "test"})  # missing 'value'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(text: str, input_tokens: int = 100, output_tokens: int = 50) -> mock.MagicMock:
    """Build a mock anthropic.types.Message."""
    resp = mock.MagicMock()
    resp.content = [mock.MagicMock(text=text)]
    resp.stop_reason = "end_turn"
    resp.usage = mock.MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


# ---------------------------------------------------------------------------
# Case 1: Success on first attempt
# ---------------------------------------------------------------------------

class TestSuccessFirstAttempt:
    def test_returns_parsed_model_and_attempt_count_1(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _mock_response(VALID_JSON)

            result, attempts = extract_with_repair("extract this", SimpleOutput)

        assert isinstance(result, SimpleOutput)
        assert result.name == "test"
        assert result.value == 42
        assert attempts == 1
        mock_client.messages.create.assert_called_once()

    def test_calls_api_with_original_prompt(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _mock_response(VALID_JSON)

            extract_with_repair("my original prompt", SimpleOutput)

        call_messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert call_messages[0]["content"] == "my original prompt"


# ---------------------------------------------------------------------------
# Case 2: Fails once, succeeds on second attempt (repair)
# ---------------------------------------------------------------------------

class TestSuccessAfterRepair:
    def test_returns_result_after_one_repair(self) -> None:
        responses = [
            _mock_response(INVALID_JSON),   # attempt 1: bad JSON
            _mock_response(VALID_JSON),      # attempt 2: correct
        ]
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = responses

            result, attempts = extract_with_repair("extract this", SimpleOutput)

        assert result.value == 42
        assert attempts == 2
        assert mock_client.messages.create.call_count == 2

    def test_repair_prompt_includes_original_and_error(self) -> None:
        """The second API call's prompt should reference the bad response and error."""
        responses = [
            _mock_response(INVALID_JSON),
            _mock_response(VALID_JSON),
        ]
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = responses

            extract_with_repair("my original prompt", SimpleOutput)

        repair_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
        repair_prompt = repair_call_messages[0]["content"]
        assert "my original prompt" in repair_prompt
        assert INVALID_JSON in repair_prompt or "parse" in repair_prompt.lower()

    def test_fails_schema_then_succeeds(self) -> None:
        """Schema validation failure triggers repair; second attempt gives valid JSON."""
        responses = [
            _mock_response(INVALID_SCHEMA_JSON),  # missing 'value'
            _mock_response(VALID_JSON),
        ]
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = responses

            result, attempts = extract_with_repair("extract this", SimpleOutput)

        assert result.value == 42
        assert attempts == 2


# ---------------------------------------------------------------------------
# Case 3: Fails all max_attempts — raises ExtractionFailedError
# ---------------------------------------------------------------------------

class TestFailureAfterMaxRetries:
    def test_raises_extraction_failed_after_3_attempts(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _mock_response(INVALID_JSON)

            with pytest.raises(ExtractionFailedError) as exc_info:
                extract_with_repair("extract this", SimpleOutput, max_attempts=3)

        assert exc_info.value.attempts == 3
        assert mock_client.messages.create.call_count == 3

    def test_exception_carries_last_raw_response(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _mock_response("garbage output")

            with pytest.raises(ExtractionFailedError) as exc_info:
                extract_with_repair("extract this", SimpleOutput, max_attempts=2)

        assert "garbage output" in exc_info.value.last_raw

    def test_custom_max_attempts_respected(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _mock_response(INVALID_JSON)

            with pytest.raises(ExtractionFailedError):
                extract_with_repair("extract this", SimpleOutput, max_attempts=1)

        # Only 1 attempt made
        assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Cost estimator tests
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_zero_tokens_zero_cost(self) -> None:
        est = estimate_cost(0, 0)
        assert est.total_cost_usd == 0.0

    def test_one_million_input_tokens(self) -> None:
        est = estimate_cost(1_000_000, 0)
        assert est.input_cost_usd == pytest.approx(15.00)
        assert est.output_cost_usd == 0.0
        assert est.total_cost_usd == pytest.approx(15.00)

    def test_one_million_output_tokens(self) -> None:
        est = estimate_cost(0, 1_000_000)
        assert est.output_cost_usd == pytest.approx(75.00)

    def test_mixed_usage(self) -> None:
        # 500k input + 100k output
        est = estimate_cost(500_000, 100_000)
        assert est.input_cost_usd == pytest.approx(7.50)
        assert est.output_cost_usd == pytest.approx(7.50)
        assert est.total_cost_usd == pytest.approx(15.00)

    def test_token_usage_accumulator(self) -> None:
        usage = TokenUsage()
        usage.add(1000, 500)
        usage.add(2000, 300)
        assert usage.input_tokens == 3000
        assert usage.output_tokens == 800

    def test_estimate_from_usage(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        est = estimate_cost_from_usage(usage)
        assert est.total_cost_usd == pytest.approx(90.00)


# ---------------------------------------------------------------------------
# Budget guard tests
# ---------------------------------------------------------------------------

class TestBudgetedClient:
    def _make_client(self, budget: float) -> BudgetedClient:
        with mock.patch("anthropic.Anthropic"):
            client = BudgetedClient(max_budget_usd=budget)
        return client

    def test_raises_on_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BudgetedClient(max_budget_usd=0.0)

    def test_budget_exceeded_when_already_spent(self) -> None:
        """After spending >= budget, next call raises BudgetExceededError."""
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client_instance = mock_cls.return_value

            # First call: uses 1M input tokens => $15 spend; budget is $0.01
            mock_client_instance.messages.create.return_value = _mock_response(
                "result", input_tokens=1_000_000, output_tokens=0
            )

            bc = BudgetedClient(max_budget_usd=0.01)

            # First call should succeed (budget not yet checked mid-call)
            bc.create_message(
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
            )
            # Now spent $15 >> $0.01

            with pytest.raises(BudgetExceededError) as exc_info:
                bc.create_message(
                    system="sys",
                    messages=[{"role": "user", "content": "hi again"}],
                )

        assert exc_info.value.budget_usd == pytest.approx(0.01)
        assert exc_info.value.spent_usd > 0.01

    def test_remaining_usd_decreases_after_call(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client_instance = mock_cls.return_value
            mock_client_instance.messages.create.return_value = _mock_response(
                "ok", input_tokens=100_000, output_tokens=10_000
            )

            bc = BudgetedClient(max_budget_usd=10.00)
            initial_remaining = bc.remaining_usd
            bc.create_message(system="s", messages=[{"role": "user", "content": "q"}])
            assert bc.remaining_usd < initial_remaining

    def test_spent_usd_is_zero_initially(self) -> None:
        with mock.patch("anthropic.Anthropic"):
            bc = BudgetedClient(max_budget_usd=5.00)
        assert bc.spent_usd == pytest.approx(0.0)

    def test_budget_exceeded_error_attributes(self) -> None:
        exc = BudgetExceededError("over budget", spent_usd=1.50, budget_usd=1.00)
        assert exc.spent_usd == 1.50
        assert exc.budget_usd == 1.00
        assert "over budget" in str(exc)
```
