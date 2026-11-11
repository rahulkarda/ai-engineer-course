# Solutions — Phase 5 · Week 2 · Day 1 — Tool & Function Calling Fundamentals

---

## Exercise 1 — `search_web` Tool Schema

Define a `search_web` tool schema as a manual JSON schema dict. Takes `query: str` and optional `max_results: int` (default 5).

```python
import os
import anthropic

# Manual JSON schema dict for the search_web tool.
# The Anthropic API expects: name, description, input_schema (JSON Schema object).
search_web_tool: dict = {
    "name": "search_web",
    "description": (
        "Search the web for current information on a given topic. "
        "Returns a list of relevant results with titles, URLs, and snippets. "
        "Use this when you need up-to-date information not in your training data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string. Be specific for better results.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],  # max_results is optional
    },
}


def demonstrate_tool_use() -> None:
    """Send a message to Claude with the search_web tool defined."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        tools=[search_web_tool],
        messages=[{"role": "user", "content": "What are the latest Python 3.13 features?"}],
    )

    logger.info("Stop reason: %s", response.stop_reason)
    logger.info("Content: %s", response.content)


if __name__ == "__main__":
    demonstrate_tool_use()
```

**Key points:**
- `input_schema` uses JSON Schema draft-7 syntax — `type`, `properties`, `required`.
- Only fields in `required` are mandatory; omit optional fields from that list.
- `description` at the tool level guides the model on *when* to call the tool.
- `description` on each property guides the model on *what to pass*.

---

## Exercise 2 — `create_calendar_event` Tool: JSON Schema + Pydantic Model

```python
import os
from datetime import datetime
from typing import Any

import anthropic
from pydantic import BaseModel, field_validator


# --- Pydantic model for input validation ---

class CreateCalendarEventInput(BaseModel):
    """Validated input for creating a calendar event."""

    title: str
    date: str  # ISO 8601 format, e.g. "2025-06-15T14:00:00"
    duration_minutes: int
    attendees: list[str]

    @field_validator("date")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Ensure date is valid ISO 8601."""
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"date must be ISO 8601, got: {v!r}") from exc
        return v

    @field_validator("duration_minutes")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("duration_minutes must be positive")
        return v

    @field_validator("attendees")
    @classmethod
    def validate_attendees(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("attendees list must not be empty")
        return v


# --- JSON schema for the Anthropic tools API ---

create_calendar_event_tool: dict[str, Any] = {
    "name": "create_calendar_event",
    "description": (
        "Create a new calendar event and invite attendees. "
        "Use this when the user asks to schedule a meeting, appointment, or event. "
        "All times must be in ISO 8601 format (e.g. '2025-06-15T14:00:00')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title or subject of the calendar event.",
            },
            "date": {
                "type": "string",
                "description": "Start date and time in ISO 8601 format, e.g. '2025-06-15T14:00:00'.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Duration of the event in minutes. Must be a positive integer.",
                "minimum": 1,
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee email addresses or names.",
                "minItems": 1,
            },
        },
        "required": ["title", "date", "duration_minutes", "attendees"],
    },
}


def handle_calendar_tool_call(raw_input: dict[str, Any]) -> dict[str, Any]:
    """Validate raw tool input with Pydantic before processing."""
    import logging

    logger = logging.getLogger(__name__)

    validated = CreateCalendarEventInput(**raw_input)
    logger.info("Creating event: %s on %s for %d min", validated.title, validated.date, validated.duration_minutes)

    # Actual calendar API call would go here
    return {"status": "created", "event_id": "evt_abc123", "title": validated.title}


# Example usage
if __name__ == "__main__":
    sample_input = {
        "title": "Sprint Planning",
        "date": "2025-06-16T10:00:00",
        "duration_minutes": 60,
        "attendees": ["alice@example.com", "bob@example.com"],
    }
    result = handle_calendar_tool_call(sample_input)
    print(result)
```

**Key points:**
- JSON schema goes to the API; Pydantic model validates at runtime before your business logic.
- `field_validator` with `@classmethod` is Pydantic v2 syntax (replaces `@validator`).
- `minItems: 1` in JSON schema is a *hint* to the model; the Pydantic validator enforces it server-side.

---

## Exercise 3 — Predicting the Model's `tool_use` Response

**Scenario:** You send the message `"What's the weather like in Paris?"` with a `get_weather(city: str)` tool defined.

**Predicted model response** — the full response object would have `stop_reason: "tool_use"` and the content list would contain a `tool_use` block:

```python
from typing import Any

# What the Anthropic SDK returns as response.content when the model decides to call get_weather
predicted_response_content: list[dict[str, Any]] = [
    {
        "type": "tool_use",
        "id": "toolu_01XFDUDYJgAACTvnkyqtzM5k",   # unique ID generated by the API
        "name": "get_weather",
        "input": {
            "city": "Paris"
            # max_results / other optional params are omitted when the model uses defaults
        },
    }
]

# Full predicted response shape (simplified)
predicted_response: dict[str, Any] = {
    "id": "msg_01XFDUDYJgAACTvnkyqtzM5k",
    "type": "message",
    "role": "assistant",
    "content": predicted_response_content,
    "model": "claude-sonnet-5-5",
    "stop_reason": "tool_use",   # <-- key signal: model wants to call a tool
    "stop_sequence": None,
    "usage": {"input_tokens": 52, "output_tokens": 31},
}


def get_weather(city: str) -> dict[str, Any]:
    """Stub implementation — real version calls a weather API."""
    return {"city": city, "temperature_c": 18, "condition": "Partly cloudy"}


def process_response(response_content: list[dict[str, Any]]) -> None:
    """Demonstrate how to detect and handle a tool_use block."""
    import logging

    logger = logging.getLogger(__name__)

    for block in response_content:
        if block["type"] == "tool_use":
            tool_name: str = block["name"]
            tool_input: dict[str, Any] = block["input"]
            tool_use_id: str = block["id"]

            logger.info("Model requested tool: %s with input: %s", tool_name, tool_input)

            if tool_name == "get_weather":
                result = get_weather(**tool_input)
                logger.info("Tool result: %s", result)

                # This is the tool_result message you send back in the next turn
                tool_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result),
                        }
                    ],
                }
                logger.info("Prepared tool_result: %s", tool_result_message)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    process_response(predicted_response_content)
```

**Why `stop_reason == "tool_use"`?**
- The model signals it wants to call a tool by stopping with `"tool_use"` rather than `"end_turn"`.
- Your code must detect this, execute the function, and return a `tool_result` user message to continue the conversation.
- The `id` field in the `tool_use` block must match the `tool_use_id` in your `tool_result` reply — this is how the model correlates results with calls.
