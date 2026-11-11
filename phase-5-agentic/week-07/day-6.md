# Phase 5 · Week 07 · Day 6 — Saturday Deep Practice (4 hr): Build a Multi-Agent Research Pipeline

> Objective: Build a complete 5-agent research pipeline integrating architectures, messaging, blackboard, specialization, and debate from days 1–5.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] [[day-2]] [[day-3]] [[day-4]] [[day-5]]

## Overview

Today you build a **multi-agent research pipeline** end-to-end. Given a research question, the system:

1. **QueryAnalyser** — decomposes the question into sub-queries
2. **Researcher × 2** — concurrent agents each answer one sub-query (scatter-gather)
3. **Synthesiser** — integrates research outputs into a coherent draft
4. **FactChecker** — debates the draft's claims with the Synthesiser
5. **Publisher** — formats the verified output

All agents communicate via a typed message bus. Intermediate state lives on a blackboard. The full run is orchestrated by a `PipelineSupervisor`.

## Architecture diagram

```
User question
      │
      ▼
 QueryAnalyser
      │  sub-queries[]
      ├──────────────────┐
      ▼                  ▼
 Researcher-A      Researcher-B  (concurrent)
      │                  │
      └────────┬─────────┘
               │ findings[]
               ▼
          Synthesiser ◄──debate──► FactChecker
               │
               ▼
           Publisher
               │
               ▼
          Final report
```

**State lives on the Blackboard:**
- `queries/original` — user question
- `queries/sub[0..N]` — decomposed sub-queries
- `research/[0..N]` — per-researcher findings
- `drafts/synthesis` — synthesiser output
- `results/final` — publisher output

## Part 1: Scaffold (45 min)

Set up the project structure and shared infrastructure.

```
week07_project/
├── src/
│   ├── __init__.py
│   ├── models.py        # Pydantic models
│   ├── bus.py           # MessageBus (from day 2)
│   ├── blackboard.py    # Blackboard (from day 3)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py      # BaseAgent
│   │   ├── query_analyser.py
│   │   ├── researcher.py
│   │   ├── synthesiser.py
│   │   ├── fact_checker.py
│   │   └── publisher.py
│   └── supervisor.py    # PipelineSupervisor
├── tests/
│   ├── test_bus.py
│   ├── test_blackboard.py
│   └── test_pipeline.py
└── pyproject.toml
```

```python
# src/models.py
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
    sub_query: str
    answer:    str
    confidence: float = 0.8


class SynthesisOutput(BaseModel):
    draft:  str
    claims: list[str]  # extracted factual claims


class FactCheckResult(BaseModel):
    claim:    str
    verdict:  str   # "VERIFIED" | "DISPUTED" | "UNVERIFIABLE"
    reason:   str


class FinalReport(BaseModel):
    question:     str
    summary:      str
    findings:     list[ResearchFinding]
    verified_claims: list[FactCheckResult]
    word_count:   int
```

## Part 2: Message bus and blackboard (30 min)

Copy/adapt your implementations from day-2 and day-3 into `src/bus.py` and `src/blackboard.py`. Ensure they are importable from `src`.

Key additions for this project:
- `MessageBus.send_and_wait(msg, timeout)` — synchronous RPC helper
- `Blackboard.wait_for_key(key, timeout)` — blocks until a key is written

```python
# Add to bus.py
import threading

def send_and_wait(
    self,
    msg: AgentMessage,
    inbox_name: str,
    timeout: float = 10.0,
) -> AgentMessage | None:
    """Send a REQUEST and block until matching RESPONSE arrives."""
    event = threading.Event()
    result: list[AgentMessage] = []

    original_receive = self.receive

    def _poll() -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            reply = original_receive(inbox_name, timeout=0.1)
            if reply and reply.correlation_id == msg.id:
                result.append(reply)
                event.set()
                return

    t = threading.Thread(target=_poll, daemon=True)
    self.send(msg)
    t.start()
    event.wait(timeout=timeout)
    return result[0] if result else None
```

## Part 3: Specialist agents (60 min)

### QueryAnalyser

```python
# src/agents/query_analyser.py
import json
import logging
from typing import Any
import anthropic
from ..models import AgentMessage, MessageType, SubQuery

logger = logging.getLogger(__name__)

SYSTEM = """\
You decompose research questions into 2–3 focused sub-queries.
Respond ONLY with JSON: {"sub_queries": ["...", "...", "..."]}
"""


class QueryAnalyserAgent:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def analyse(self, question: str) -> list[SubQuery]:
        resp = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        raw = resp.content[0].text
        try:
            data = json.loads(raw)
            subs = data.get("sub_queries", [question])
        except json.JSONDecodeError:
            logger.warning("QueryAnalyser: JSON parse error, using original question")
            subs = [question]
        return [SubQuery(index=i, text=t) for i, t in enumerate(subs)]
```

### Researcher

```python
# src/agents/researcher.py
import logging
import anthropic
from ..models import ResearchFinding

logger = logging.getLogger(__name__)

SYSTEM = "You are a research agent. Answer the question with accurate, concise information. 3–5 sentences."


class ResearcherAgent:
    def __init__(self, name: str, client: anthropic.Anthropic) -> None:
        self.name = name
        self._client = client

    def research(self, sub_query: str) -> ResearchFinding:
        resp = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=SYSTEM,
            messages=[{"role": "user", "content": sub_query}],
        )
        answer = resp.content[0].text
        logger.info("%s | answered: %s...", self.name, answer[:40])
        return ResearchFinding(sub_query=sub_query, answer=answer)
```

### Synthesiser

```python
# src/agents/synthesiser.py
import logging
import re
import anthropic
from ..models import ResearchFinding, SynthesisOutput

logger = logging.getLogger(__name__)

SYSTEM = """\
You synthesise research findings into a coherent paragraph.
Then list the key factual claims you made.
Format:
DRAFT: <paragraph>
CLAIMS:
- <claim 1>
- <claim 2>
"""


class SynthesiserAgent:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def synthesise(self, findings: list[ResearchFinding]) -> SynthesisOutput:
        content = "\n\n".join(
            f"Finding {f.sub_query}:\n{f.answer}" for f in findings
        )
        resp = self._client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=512,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text
        draft_m  = re.search(r"DRAFT:\s*(.+?)(?=CLAIMS:|$)", raw, re.DOTALL)
        claim_ms = re.findall(r"^-\s*(.+)$", raw, re.MULTILINE)
        return SynthesisOutput(
            draft=draft_m.group(1).strip() if draft_m else raw,
            claims=claim_ms,
        )
```

### FactChecker

```python
# src/agents/fact_checker.py
import logging
import re
import anthropic
from ..models import FactCheckResult

logger = logging.getLogger(__name__)

SYSTEM = """\
You fact-check claims. For each claim respond:
CLAIM: <claim>
VERDICT: VERIFIED | DISPUTED | UNVERIFIABLE
REASON: <one sentence>
"""


class FactCheckerAgent:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def check(self, claims: list[str]) -> list[FactCheckResult]:
        content = "\n".join(f"- {c}" for c in claims)
        resp = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Check these claims:\n{content}"}],
        )
        raw = resp.content[0].text
        results = []
        blocks = re.split(r"\nCLAIM:", raw)
        for block in blocks:
            if not block.strip():
                continue
            c_m = re.search(r"^(.+)", block.strip())
            v_m = re.search(r"VERDICT:\s*(VERIFIED|DISPUTED|UNVERIFIABLE)", block)
            r_m = re.search(r"REASON:\s*(.+)", block)
            results.append(FactCheckResult(
                claim=c_m.group(1).strip() if c_m else block[:50],
                verdict=v_m.group(1) if v_m else "UNVERIFIABLE",
                reason=r_m.group(1).strip() if r_m else "",
            ))
        return results
```

### Publisher

```python
# src/agents/publisher.py
import logging
from ..models import FinalReport, ResearchFinding, FactCheckResult

logger = logging.getLogger(__name__)


class PublisherAgent:
    def publish(
        self,
        question: str,
        summary: str,
        findings: list[ResearchFinding],
        fact_results: list[FactCheckResult],
    ) -> FinalReport:
        verified = [fr for fr in fact_results if fr.verdict == "VERIFIED"]
        report = FinalReport(
            question=question,
            summary=summary,
            findings=findings,
            verified_claims=verified,
            word_count=len(summary.split()),
        )
        logger.info(
            "publisher | report ready: %d words, %d verified claims",
            report.word_count, len(report.verified_claims),
        )
        return report
```

## Part 4: Supervisor (45 min)

```python
# src/supervisor.py
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
from .models import FinalReport

logger = logging.getLogger(__name__)


class PipelineSupervisor:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
        self._board  = Blackboard()

    def run(self, question: str) -> FinalReport:
        logger.info("supervisor | starting pipeline for: %s", question[:60])
        self._board.write("queries/original", question, author="supervisor")

        # Stage 1: decompose
        analyser   = QueryAnalyserAgent(self._client)
        sub_queries = analyser.analyse(question)
        for sq in sub_queries:
            self._board.write(f"queries/sub{sq.index}", sq.text, author="analyser")
        logger.info("supervisor | %d sub-queries generated", len(sub_queries))

        # Stage 2: concurrent research
        findings = []
        with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
            futures = {
                pool.submit(
                    ResearcherAgent(f"researcher-{sq.index}", self._client).research,
                    sq.text,
                ): sq.index
                for sq in sub_queries
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                finding = fut.result()
                self._board.write(f"research/{idx}", finding.model_dump(), author=f"researcher-{idx}")
                findings.append(finding)
        logger.info("supervisor | %d findings collected", len(findings))

        # Stage 3: synthesise
        synth   = SynthesiserAgent(self._client)
        synthesis = synth.synthesise(findings)
        self._board.write("drafts/synthesis", synthesis.model_dump(), author="synthesiser")

        # Stage 4: fact-check
        checker = FactCheckerAgent(self._client)
        fact_results = checker.check(synthesis.claims)

        # Stage 5: publish
        publisher = PublisherAgent()
        report = publisher.publish(question, synthesis.draft, findings, fact_results)
        self._board.write("results/final", report.model_dump(), author="publisher")

        return report
```

## Part 5: Tests (45 min)

```python
# tests/test_pipeline.py
"""Integration tests for the research pipeline — LLM fully mocked."""
from unittest.mock import MagicMock, patch
import pytest
from src.supervisor import PipelineSupervisor
from src.models import FinalReport


def _mock_client(responses: list[str]) -> MagicMock:
    """Return a mock Anthropic client that cycles through responses."""
    client = MagicMock()
    call_count = {"n": 0}

    def _create(**kwargs):  # noqa: ANN001
        resp = MagicMock()
        idx = call_count["n"] % len(responses)
        resp.content = [MagicMock(text=responses[idx])]
        call_count["n"] += 1
        return resp

    client.messages.create.side_effect = _create
    return client


MOCK_RESPONSES = [
    # QueryAnalyser
    '{"sub_queries": ["What is asyncio?", "What are coroutines?"]}',
    # Researcher 0
    "asyncio is Python's async I/O framework based on an event loop.",
    # Researcher 1
    "Coroutines are functions that can suspend execution with await.",
    # Synthesiser
    "DRAFT: Python asyncio uses coroutines for async I/O.\nCLAIMS:\n- asyncio uses an event loop\n- coroutines use await",
    # FactChecker
    "CLAIM: asyncio uses an event loop\nVERDICT: VERIFIED\nREASON: Core design.\n\nCLAIM: coroutines use await\nVERDICT: VERIFIED\nREASON: Language spec.",
]


def test_pipeline_returns_final_report() -> None:
    client = _mock_client(MOCK_RESPONSES)
    supervisor = PipelineSupervisor(client)
    report = supervisor.run("How does Python asyncio work?")
    assert isinstance(report, FinalReport)
    assert report.question == "How does Python asyncio work?"
    assert len(report.findings) >= 1
    assert report.word_count > 0


def test_pipeline_verified_claims() -> None:
    client = _mock_client(MOCK_RESPONSES)
    supervisor = PipelineSupervisor(client)
    report = supervisor.run("How does Python asyncio work?")
    assert all(vc.verdict == "VERIFIED" for vc in report.verified_claims)


def test_blackboard_populated() -> None:
    client = _mock_client(MOCK_RESPONSES)
    supervisor = PipelineSupervisor(client)
    supervisor.run("Test question")
    snap = supervisor._board.snapshot()
    assert "queries/original" in snap
    assert "results/final" in snap
```

## Part 6: Integration run (15 min)

Wire up a `main.py` entry point:

```python
# main.py
import logging
import os
import anthropic
from src.supervisor import PipelineSupervisor

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

if __name__ == "__main__":
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    supervisor = PipelineSupervisor(client)
    report = supervisor.run(
        "What are the key differences between threads, processes, and async IO in Python?"
    )
    print(f"\n{'='*60}")
    print(f"QUESTION: {report.question}")
    print(f"SUMMARY ({report.word_count} words):\n{report.summary}")
    print(f"\nVERIFIED CLAIMS ({len(report.verified_claims)}):")
    for vc in report.verified_claims:
        print(f"  [{vc.verdict}] {vc.claim}")
```

## Deliverables checklist

- [ ] All 5 specialist agents implemented with correct input/output types
- [ ] `PipelineSupervisor` orchestrates end-to-end pipeline
- [ ] Blackboard used for intermediate state (not just local variables)
- [ ] Concurrent research using `ThreadPoolExecutor`
- [ ] `test_pipeline.py` passes with mocked LLM — no real API calls
- [ ] `test_blackboard.py` covers write/read/CAS/watch
- [ ] `test_bus.py` covers send/receive/routing
- [ ] No hardcoded API keys; all via `os.getenv("ANTHROPIC_API_KEY")`

(Solutions: ../solutions/week-07/day-6.md)

## Self-check

- [ ] I can build a multi-agent pipeline from scratch without scaffolding
- [ ] I understand how to coordinate concurrent agents with scatter-gather
- [ ] My tests mock all LLM calls and run without network
- [ ] I can trace a message from user input to final report through every agent
