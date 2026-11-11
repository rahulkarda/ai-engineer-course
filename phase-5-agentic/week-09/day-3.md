# Phase 5 · Week 09 · Day 3 — Agent Cards & Capability Advertisement

> Objective: Build agents that publish structured capability manifests and implement a registry for discovery.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

An **agent card** is the machine-readable identity of an A2A server — the JSON document that answers: "What can you do, how do I call you, and what do you need from me?"

Without a card, every integration requires manual documentation lookup, schema negotiation, and bespoke client code. With cards, a client can discover and call any compliant agent automatically.

**Agent card fields (A2A spec):**

| Field | Purpose |
|---|---|
| `name` | Stable identifier (slug, not display name) |
| `description` | Human-readable summary for LLM routing |
| `version` | Semver; clients can require minimum versions |
| `url` | Base URL where the agent serves A2A endpoints |
| `capabilities` | Object listing supported A2A features (streaming, push notifications) |
| `skills` | List of `Skill` objects — the actual task types this agent handles |
| `defaultInputModes` | Accepted content types (`text/plain`, `application/json`, etc.) |
| `defaultOutputModes` | Produced content types |
| `authentication` | Auth schemes required (none, API key, OAuth2, etc.) |

**Skill vs. capability:** A *capability* is a protocol feature (e.g., "can stream responses"). A *skill* is a domain function (e.g., "translates text between languages"). Clients use skills to route tasks; they use capabilities to choose communication patterns.

## How it works internally

Agent cards are served as static JSON at `/.well-known/agent.json`. This convention (borrowed from RFC 8615) means any HTTP client can discover an agent with one well-known GET request — no DNS special records, no service registry query needed.

A **registry** is an optional layer on top: a service that indexes known agent cards, supports search by skill, and caches cards with TTLs so clients don't have to probe every known URL on every request.

**Routing algorithm in an orchestrator:**

```
1. Receive user task
2. Extract required skills (NLP or rule-based)
3. Query registry for agents with matching skills
4. Score matches (exact > partial; reliability score; latency)
5. Pick winner; fall back to next if first fails
```

Cards should be versioned. When an agent upgrades its skill set, it bumps the card version. Clients can cache cards with ETags and revalidate when they get a `412 Precondition Failed` on a task submission — the server rejects tasks that reference stale card versions.

## Worked example

```python
"""
day_3_agent_cards.py
Pydantic models for A2A agent cards, a local registry, and capability-based routing.
"""

import logging
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Agent card schema (A2A-compatible)
# ──────────────────────────────────────────────

class AgentSkill(BaseModel):
    id: str                           # machine-readable slug, e.g. "text-summarization"
    name: str                         # human label
    description: str
    input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    examples: list[str] = Field(default_factory=list)  # sample prompts

    @field_validator("id")
    @classmethod
    def id_is_slug(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Skill id must be a slug, got: {v!r}")
        return v


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False


class AuthScheme(BaseModel):
    scheme: Literal["none", "api_key", "oauth2", "bearer"]
    description: str = ""


class AgentCard(BaseModel):
    """Served at /.well-known/agent.json"""
    name: str
    description: str
    version: str                      # semver
    url: str                          # base URL
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    authentication: AuthScheme = Field(default_factory=lambda: AuthScheme(scheme="none"))
    tags: list[str] = Field(default_factory=list)  # free-form for search

    def has_skill(self, skill_id: str) -> bool:
        return any(s.id == skill_id for s in self.skills)

    def skill_ids(self) -> list[str]:
        return [s.id for s in self.skills]


# ──────────────────────────────────────────────
# Sample agent cards
# ──────────────────────────────────────────────

def make_summarizer_card() -> AgentCard:
    return AgentCard(
        name="summarization-agent",
        description="Summarizes long documents into concise, structured output",
        version="1.2.0",
        url="https://agents.example.com/summarizer",
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="text-summarization",
                name="Text Summarization",
                description="Produces a concise summary with configurable length",
                examples=["Summarize this 50-page report in 5 bullet points"],
            ),
            AgentSkill(
                id="key-point-extraction",
                name="Key Point Extraction",
                description="Extracts the N most important claims from a document",
                examples=["Extract 10 key points from this meeting transcript"],
            ),
        ],
        tags=["nlp", "text-processing"],
    )


def make_sql_agent_card() -> AgentCard:
    return AgentCard(
        name="sql-generation-agent",
        description="Translates natural-language questions into SQL queries",
        version="2.0.1",
        url="https://agents.example.com/sql",
        capabilities=AgentCapabilities(push_notifications=True),
        skills=[
            AgentSkill(
                id="nl-to-sql",
                name="Natural Language to SQL",
                description="Generates safe, read-only SQL from plain-English questions",
                input_modes=["text/plain"],
                output_modes=["text/plain", "application/sql"],
                examples=["How many users signed up last month?"],
            ),
            AgentSkill(
                id="query-explanation",
                name="SQL Query Explanation",
                description="Explains an existing SQL query in plain English",
            ),
        ],
        authentication=AuthScheme(scheme="api_key", description="Pass X-API-Key header"),
        tags=["sql", "data", "analytics"],
    )


def make_translation_agent_card() -> AgentCard:
    return AgentCard(
        name="translation-agent",
        description="Translates text between 50+ language pairs",
        version="3.1.0",
        url="https://agents.example.com/translate",
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=[
            AgentSkill(
                id="text-translation",
                name="Text Translation",
                description="Translates text to a target language, preserving formatting",
                input_modes=["text/plain", "text/markdown"],
                output_modes=["text/plain", "text/markdown"],
                examples=["Translate to French: Hello, how are you?"],
            ),
        ],
        tags=["translation", "multilingual", "nlp"],
    )


# ──────────────────────────────────────────────
# Agent registry
# ──────────────────────────────────────────────

class AgentRegistry:
    """In-memory registry. Production would back this with Redis or a DB."""

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}   # name → card

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card
        logger.info("Registered agent '%s' with skills: %s", card.name, card.skill_ids())

    def find_by_skill(self, skill_id: str) -> list[AgentCard]:
        """Return all agents that have the given skill."""
        return [c for c in self._cards.values() if c.has_skill(skill_id)]

    def find_by_tag(self, tag: str) -> list[AgentCard]:
        return [c for c in self._cards.values() if tag in c.tags]

    def get(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def all_skills(self) -> dict[str, list[str]]:
        """Map skill_id → list of agent names that provide it."""
        index: dict[str, list[str]] = {}
        for card in self._cards.values():
            for skill in card.skills:
                index.setdefault(skill.id, []).append(card.name)
        return index


# ──────────────────────────────────────────────
# Routing logic
# ──────────────────────────────────────────────

def route_task(registry: AgentRegistry, required_skill: str) -> AgentCard | None:
    """
    Simple routing: return first agent with required skill.
    Production: score by latency, reliability, auth cost, etc.
    """
    candidates = registry.find_by_skill(required_skill)
    if not candidates:
        logger.warning("No agent found for skill '%s'", required_skill)
        return None
    chosen = candidates[0]
    logger.info("Routing '%s' task to '%s' at %s", required_skill, chosen.name, chosen.url)
    return chosen


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    registry = AgentRegistry()
    registry.register(make_summarizer_card())
    registry.register(make_sql_agent_card())
    registry.register(make_translation_agent_card())

    logger.info("All skills in registry: %s", registry.all_skills())

    # Route different task types
    for skill in ["text-summarization", "nl-to-sql", "text-translation", "unknown-skill"]:
        route_task(registry, skill)

    # NLP agents
    nlp_agents = registry.find_by_tag("nlp")
    logger.info("NLP-tagged agents: %s", [a.name for a in nlp_agents])

    # Serialise a card to JSON (for /.well-known/agent.json endpoint)
    card_json = make_summarizer_card().model_dump_json(indent=2)
    logger.info("Summarizer card JSON (first 300 chars):\n%s", card_json[:300])


if __name__ == "__main__":
    main()
```

Run: `pip install pydantic && python day_3_agent_cards.py`

## Exercises

1. Add a `reliability_score: float` field (0.0–1.0) to `AgentCard`. Update `route_task` to sort candidates by score descending and return the top-scoring agent. Write a pytest that verifies the highest-score agent is chosen.

2. Implement `AgentRegistry.find_by_capabilities(streaming=True, push_notifications=True)` — returns only agents that have all specified capabilities active. Test with at least three cards.

3. Build a `CardCache` that wraps `AgentRegistry`, stores cards with a `ttl_seconds` expiry, and re-fetches (mock the fetch) when TTL expires. Use `time.monotonic()` for expiry. Test the cache miss path.

4. Add `min_version: str` to `route_task`. Use Python's `packaging.version` (or a manual semver compare) to filter out agents below the minimum version. Test with versions `1.0.0`, `2.0.1`, and `3.1.0` and a minimum of `2.0.0`.

(Solutions: ../solutions/week-09/day-3.md)

## Self-check
- [ ] I can write a complete, valid `AgentCard` in Pydantic from memory
- [ ] I understand the difference between a skill and a capability
- [ ] I can implement a registry with skill-based and tag-based lookup
- [ ] I can explain why `/.well-known/agent.json` is the discovery endpoint

## Supplementary links
- [A2A Agent Card specification](https://a2aproject.github.io/A2A/specification/#agent-card)
- [RFC 8615 — Well-Known URI scheme](https://datatracker.ietf.org/doc/html/rfc8615)
- [Pydantic v2 validators](https://docs.pydantic.dev/latest/concepts/validators/)
