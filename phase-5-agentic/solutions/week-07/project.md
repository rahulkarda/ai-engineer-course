# Solutions — Week 07 Project: Multi-Agent Research Assistant

This file provides the reference implementation. It is a complete, runnable project matching the requirements in `week-07/project.md`.

---

## Project structure (complete)

```
week07_project/
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── blackboard.py
│   ├── supervisor.py
│   └── agents/
│       ├── __init__.py
│       ├── query_analyser.py
│       ├── researcher.py
│       ├── synthesiser.py
│       ├── fact_checker.py
│       └── publisher.py
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_blackboard.py
│   └── test_pipeline.py
├── main.py
├── pyproject.toml
└── .env.example
```

---

## src/models.py (complete)

```python
"""Shared Pydantic models for the research pipeline."""
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    REQUEST   = "REQUEST"
    RESPONSE  = "RESPONSE"
    ERROR     = "ERROR"
    BROADCAST = "BROADCAST"


class AgentMessage(BaseModel):
    id:             str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    sender:         str
    receiver:       str
    message_type:   MessageType
    payload:        dict[str, Any]
    timestamp:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SubQuery(BaseModel):
    index: int
    text:  str


class ResearchFinding(BaseModel):
    sub_query:  str
    answer:     str
    confidence: float = 0.8


class SynthesisOutput(BaseModel):
    draft:  str
    claims: list[str]


class FactCheckResult(BaseModel):
    claim:   str
    verdict: str   # "VERIFIED" | "DISPUTED" | "UNVERIFIABLE"
    reason:  str


class FinalReport(BaseModel):
    question:        str
    summary:         str
    findings:        list[ResearchFinding]
    verified_claims: list[FactCheckResult]
    word_count:      int
```

---

## src/blackboard.py (complete)

```python
"""Thread-safe blackboard with namespaces, versioning, and watchers."""
import logging
import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BlackboardEntry(BaseModel):
    key:     str
    value:   Any
    version: int = 1
    author:  str = "unknown"


class Blackboard:
    def __init__(self) -> None:
        self._store:    dict[str, BlackboardEntry] = {}
        self._lock:     threading.RLock = threading.RLock()
        self._watchers: dict[str, list[Callable[[BlackboardEntry], None]]] = {}

    def read(self, key: str) -> BlackboardEntry | None:
        with self._lock:
            return self._store.get(key)

    def read_section(self, section: str) -> dict[str, Any]:
        prefix = f"{section}/"
        with self._lock:
            return {k[len(prefix):]: v.value for k, v in self._store.items() if k.startswith(prefix)}

    def write(self, key: str, value: Any, author: str,
              expected_version: int | None = None) -> bool:
        with self._lock:
            existing = self._store.get(key)
            if expected_version is not None:
                stored = existing.version if existing else 0
                if stored != expected_version:
                    logger.warning("blackboard | CAS conflict on %s", key)
                    return False
            new_v = (existing.version + 1) if existing else 1
            entry = BlackboardEntry(key=key, value=value, version=new_v, author=author)
            self._store[key] = entry

        self._notify(key, entry)
        return True

    def watch(self, prefix: str, callback: Callable[[BlackboardEntry], None]) -> None:
        with self._lock:
            self._watchers.setdefault(prefix, []).append(callback)

    def _notify(self, key: str, entry: BlackboardEntry) -> None:
        handlers: list[Callable[[BlackboardEntry], None]] = []
        with self._lock:
            for prefix, cbs in self._watchers.items():
                if key.startswith(prefix):
                    handlers.extend(cbs)
        for cb in handlers:
            threading.Thread(target=cb, args=(entry,), daemon=True).start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: v.value for k, v in self._store.items()}
```

---

## src/supervisor.py (complete)

```python
"""PipelineSupervisor — orchestrates the full research pipeline."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from .agents.fact_checker import FactCheckerAgent
from .agents.publisher import PublisherAgent
from .agents.query_analyser import QueryAnalyserAgent
from .agents.researcher import ResearcherAgent
from .agents.synthesiser import SynthesiserAgent
from .blackboard import Blackboard
from .models import FinalReport, ResearchFinding

logger = logging.getLogger(__name__)

_DEFAULT_POOL_SIZE = int(os.getenv("RESEARCHER_POOL_SIZE", "2"))


class PipelineSupervisor:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
        self._board  = Blackboard()

    def run(self, question: str) -> FinalReport:
        logger.info("supervisor | start: %s", question[:60])
        self._board.write("queries/original", question, author="supervisor")

        # Stage 1: decompose
        sub_queries = QueryAnalyserAgent(self._client).analyse(question)
        for sq in sub_queries:
            self._board.write(f"queries/sub{sq.index}", sq.text, author="analyser")
        pool_size = min(len(sub_queries), _DEFAULT_POOL_SIZE)
        logger.info("supervisor | %d sub-queries, pool size %d", len(sub_queries), pool_size)

        # Stage 2: concurrent research
        findings: list[ResearchFinding] = []
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            futures = {
                pool.submit(
                    ResearcherAgent(f"researcher-{sq.index}", self._client).research,
                    sq.text,
                ): sq.index
                for sq in sub_queries
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    finding = fut.result()
                    self._board.write(f"research/{idx}", finding.model_dump(), author=f"researcher-{idx}")
                    findings.append(finding)
                except Exception as exc:
                    logger.exception("researcher-%d failed: %s", idx, exc)

        if not findings:
            raise RuntimeError("All researchers failed — cannot synthesise")

        # Stage 3: synthesise
        synthesis = SynthesiserAgent(self._client).synthesise(findings)
        self._board.write("drafts/synthesis", synthesis.model_dump(), author="synthesiser")

        # Stage 4: fact-check
        fact_results = FactCheckerAgent(self._client).check(synthesis.claims)

        # Stage 5: publish
        report = PublisherAgent().publish(question, synthesis.draft, findings, fact_results)
        self._board.write("results/final", report.model_dump(), author="publisher")
        logger.info("supervisor | done: %d words", report.word_count)
        return report
```

---

## tests/test_pipeline.py (complete)

```python
"""Integration tests — all LLM calls mocked."""
from unittest.mock import MagicMock
import pytest
from src.supervisor import PipelineSupervisor
from src.models import FinalReport


def _mock_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    state = {"n": 0}

    def _create(**kwargs):
        resp = MagicMock()
        resp.content = [MagicMock(text=responses[state["n"] % len(responses)])]
        state["n"] += 1
        return resp

    client.messages.create.side_effect = _create
    return client


STANDARD_RESPONSES = [
    '{"sub_queries": ["What is asyncio?", "What are coroutines?"]}',
    "asyncio is Python's async I/O framework.",
    "Coroutines are suspendable functions.",
    "DRAFT: Python asyncio uses coroutines.\nCLAIMS:\n- asyncio uses event loop\n- coroutines use await",
    "CLAIM: asyncio uses event loop\nVERDICT: VERIFIED\nREASON: Core design.\n\nCLAIM: coroutines use await\nVERDICT: VERIFIED\nREASON: Language spec.",
]


def test_pipeline_returns_final_report() -> None:
    report = PipelineSupervisor(_mock_client(STANDARD_RESPONSES)).run("How does asyncio work?")
    assert isinstance(report, FinalReport)
    assert report.word_count > 0


def test_pipeline_question_preserved() -> None:
    question = "What is a coroutine?"
    report = PipelineSupervisor(_mock_client(STANDARD_RESPONSES)).run(question)
    assert report.question == question


def test_pipeline_verified_claims_only_verified() -> None:
    report = PipelineSupervisor(_mock_client(STANDARD_RESPONSES)).run("Test")
    assert all(c.verdict == "VERIFIED" for c in report.verified_claims)


def test_blackboard_has_required_keys() -> None:
    sup = PipelineSupervisor(_mock_client(STANDARD_RESPONSES))
    sup.run("Test question")
    snap = sup._board.snapshot()
    assert "queries/original" in snap
    assert "drafts/synthesis" in snap
    assert "results/final" in snap


def test_pipeline_resilient_to_researcher_failure() -> None:
    responses = [
        '{"sub_queries": ["q1", "q2"]}',
        "Finding for q2.",   # researcher 0 will raise, researcher 1 succeeds
        "DRAFT: Partial.\nCLAIMS:\n- one claim",
        "CLAIM: one claim\nVERDICT: VERIFIED\nREASON: OK.",
    ]
    client = _mock_client(responses)
    call_count = {"n": 0}
    original_side_effect = client.messages.create.side_effect

    def _patched(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:  # second call = researcher 0
            raise Exception("timeout")
        return original_side_effect(**kwargs)

    client.messages.create.side_effect = _patched
    report = PipelineSupervisor(client).run("Resilience test")
    assert report is not None
    assert len(report.findings) >= 1
```

---

## pyproject.toml

```toml
[project]
name = "week07-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff"]

[tool.pytest.ini_options]
testpaths = ["tests"]
log_cli = true
log_cli_level = "INFO"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "ANN", "N"]
ignore = ["ANN101", "ANN102"]
```

---

## .env.example

```
ANTHROPIC_API_KEY=your_key_here
RESEARCHER_POOL_SIZE=2
```

---

## Running the project

```bash
# Install
pip install -e ".[dev]"

# Run tests (no API key needed)
pytest tests/ -v

# Run integration (real API)
export ANTHROPIC_API_KEY=<your_key>
python main.py "What are the trade-offs between PostgreSQL and MongoDB?"

# Dry run (no API key needed)
python main.py "Test question" --dry-run
```
