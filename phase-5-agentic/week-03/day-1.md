# Phase 5 · Week 03 · Day 1 — ReAct Pattern: Reasoning + Acting from Scratch

> Objective: Implement the ReAct (Reason+Act) loop in pure Python and understand why interleaved thought/action/observation outperforms pure chain-of-thought or pure tool use.  ·  Time: ~1 hr  ·  Prereqs: [[week-02/day-4]]

## Concept

**ReAct** (Yao et al., 2022) is a prompting and control-flow pattern where the LLM alternates between:

1. **Thought** — internal reasoning about what to do next
2. **Action** — a tool call with structured arguments
3. **Observation** — the tool result appended back to context

This cycle repeats until the LLM emits a `Finish` action with the final answer.

### Why not pure chain-of-thought?

Chain-of-thought (CoT) reasons in one shot. It can hallucinate facts because it never goes back to the world. ReAct grounds each reasoning step in real tool output, catching errors before they compound.

### Why not pure tool use?

Calling tools without explicit reasoning produces brittle pipelines: the model has no scratchpad to resolve ambiguity, reconsider a failed tool call, or choose between alternative tools. Thought steps make the decision process legible and steerable.

### Why interleaving works

Each thought has access to all prior observations. The model can write:

```
Thought: The search returned a population figure for 2020, but the question asks for 2023.
         I should search again with a more specific query.
Action: search("world population 2023 estimate")
```

This is impossible in a single-pass architecture.

### Stopping conditions

- LLM emits `Action: Finish[<answer>]`
- `max_steps` exceeded — return best partial answer with a warning
- Tool raises an unrecoverable error

## How it works internally

The loop maintains a **trace** — an ordered list of `(thought, action, observation)` triples. The entire trace is serialised into the context window on every LLM call:

```
[System prompt with tool descriptions]
Goal: <user goal>
Thought: <t1>
Action: <a1>
Observation: <o1>
Thought: <t2>
...
```

The LLM sees its own prior reasoning as text. It does not have memory in any other sense. This means:

- Traces grow linearly — cost increases with depth
- Truncating early turns can lose critical observations (prefer summarising over hard truncation)
- The format is the protocol — any deviation (wrong prefix, missing newline) breaks parsing

### Tool dispatch

The host program owns dispatch. The LLM produces a string like `search("climate change 2024")`. The host:

1. Parses the action string with a regex or structured output
2. Looks up the tool name in a registry (`dict[str, Callable]`)
3. Calls the function, captures stdout/return value
4. Appends `Observation: <result>` to the trace

The LLM never executes code. You can mock every tool in tests without touching the model.

## Worked example

```python
"""
day_1_react.py — ReAct agent in pure Python.

Run:
    python day_1_react.py

No real LLM required: mock_llm() simulates responses deterministically.
Swap in the Anthropic client by replacing _call_llm() as shown at the bottom.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    thought: str
    action: str
    observation: str


@dataclass
class ReactTrace:
    goal: str
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str | None = None

    def to_prompt(self) -> str:
        lines: list[str] = [f"Goal: {self.goal}"]
        for s in self.steps:
            lines.append(f"Thought: {s.thought}")
            lines.append(f"Action: {s.action}")
            lines.append(f"Observation: {s.observation}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Toy tools
# ---------------------------------------------------------------------------

_FACTS: dict[str, str] = {
    "eiffel tower height": "The Eiffel Tower is 330 metres tall including the antenna.",
    "python creator":       "Python was created by Guido van Rossum, first released in 1991.",
    "speed of light":       "The speed of light in a vacuum is approximately 299,792,458 m/s.",
}


def search(query: str) -> str:
    """Return a hardcoded fact for demo purposes."""
    key = query.strip().lower()
    for fact_key, fact_val in _FACTS.items():
        if fact_key in key or key in fact_key:
            return fact_val
    return "No result found for that query."


def calculate(expr: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    # Allow only digits, operators, spaces, and parentheses.
    sanitised = expr.strip()
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", sanitised):
        return f"Error: expression '{expr}' contains disallowed characters."
    try:
        result = eval(sanitised, {"__builtins__": {}})  # noqa: S307 — input validated above
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[[str], str]] = {
    "search":    search,
    "calculate": calculate,
}

SYSTEM_PROMPT = """\
You are a reasoning agent. On each turn output exactly two lines:
  Thought: <your reasoning>
  Action: <tool>("<argument>")

Available tools:
  search("<query>")    — look up a fact
  calculate("<expr>")  — evaluate arithmetic

When you have the final answer, output:
  Thought: I now have the answer.
  Action: Finish("<answer>")
"""

# Action parsing: matches  tool_name("argument")  or  Finish("answer")
_ACTION_RE = re.compile(r'(\w+)\("([^"]*)"\)')


class ReactAgent:
    def __init__(
        self,
        tools: dict[str, Callable[[str], str]] | None = None,
        model: str = "claude-haiku-5-5",
    ) -> None:
        self.tools = tools or TOOL_REGISTRY
        self.model = model
        # Anthropic client — initialised lazily so tests run without a key
        self._client: object | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, goal: str, max_steps: int = 10) -> str:
        trace = ReactTrace(goal=goal)
        log.info("Starting ReAct loop  goal=%r  max_steps=%d", goal, max_steps)

        for step_num in range(1, max_steps + 1):
            log.info("--- Step %d ---", step_num)

            thought, action_str = self.think(trace)
            log.info("Thought: %s", thought)
            log.info("Action:  %s", action_str)

            if action_str.startswith("Finish("):
                m = _ACTION_RE.match(action_str)
                answer = m.group(2) if m else action_str
                trace.final_answer = answer
                log.info("Final answer: %s", answer)
                return answer

            observation = self.act(action_str)
            log.info("Observation: %s", observation)

            trace.steps.append(TraceStep(thought, action_str, observation))

        log.warning("max_steps=%d reached without Finish — returning partial trace.", max_steps)
        return f"[Incomplete after {max_steps} steps] Last observation: {trace.steps[-1].observation}"

    def think(self, trace: ReactTrace) -> tuple[str, str]:
        """Call the LLM and parse out (thought, action)."""
        prompt = trace.to_prompt()
        raw = self._call_llm(prompt)
        return self._parse_thought_action(raw)

    def act(self, action_str: str) -> str:
        """Dispatch an action string to the matching tool."""
        m = _ACTION_RE.match(action_str.strip())
        if not m:
            return f"Parse error: could not parse action '{action_str}'"
        tool_name, argument = m.group(1), m.group(2)
        tool_fn = self.tools.get(tool_name)
        if tool_fn is None:
            return f"Unknown tool '{tool_name}'. Available: {list(self.tools)}"
        try:
            return tool_fn(argument)
        except Exception as exc:  # noqa: BLE001
            return f"Tool error: {exc}"

    # ------------------------------------------------------------------
    # LLM layer — swap mock for real client here
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM. Override to use a real model."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return self._call_anthropic(prompt)
        return self._mock_llm(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        """Real Anthropic call — only runs when ANTHROPIC_API_KEY is set."""
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        message = self._client.messages.create(  # type: ignore[union-attr]
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    @staticmethod
    def _mock_llm(prompt: str) -> str:
        """Deterministic mock for offline testing."""
        if "Goal: how tall" in prompt and "Observation:" not in prompt:
            return (
                "Thought: I should look up the height of the Eiffel Tower.\n"
                'Action: search("eiffel tower height")'
            )
        if "330 metres" in prompt:
            return (
                "Thought: I now have the height from the search result.\n"
                'Action: Finish("The Eiffel Tower is 330 metres tall.")'
            )
        if "Goal: what is 123" in prompt and "Observation:" not in prompt:
            return (
                "Thought: This is arithmetic — I will use the calculator.\n"
                'Action: calculate("123 * 456")'
            )
        if "56088" in prompt:
            return (
                "Thought: The calculation returned the result.\n"
                'Action: Finish("123 * 456 = 56088")'
            )
        return (
            "Thought: I do not have enough information yet.\n"
            'Action: Finish("I could not find the answer.")'
        )

    @staticmethod
    def _parse_thought_action(raw: str) -> tuple[str, str]:
        thought, action = "", ""
        for line in raw.splitlines():
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
        return thought, action


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = ReactAgent()

    result1 = agent.run("how tall is the Eiffel Tower?")
    print(f"\nAnswer 1: {result1}\n")

    result2 = agent.run("what is 123 * 456?")
    print(f"Answer 2: {result2}\n")
```

### How to swap in a real LLM

The `_call_llm` method checks for `ANTHROPIC_API_KEY`. If found, it routes to `_call_anthropic`. Set the key and run:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python day_1_react.py
```

No other changes needed.

## Exercises

1. Add a third tool `lookup_date(event: str) -> str` that returns hardcoded dates for 3 historical events. Integrate it into `TOOL_REGISTRY` and extend the mock LLM to exercise it.

2. The current loop stops at `max_steps` and returns a partial answer. Implement a `SoftStopPolicy` dataclass with `max_steps: int` and `max_observations_without_progress: int`. Stop early if the last N observations are identical (the agent is looping).

3. Rewrite `act()` to log structured JSON per tool call: `{"tool": ..., "args": ..., "result": ..., "duration_ms": ...}`. Use `logging` only — no `print`.

4. Extend `ReactTrace` with a `to_markdown() -> str` method that formats the trace as a readable Markdown table for debugging.

5. Write a `pytest` suite covering: (a) `calculate` with a disallowed character, (b) `act` with an unknown tool name, (c) `run` reaching `max_steps` without `Finish`.

(Solutions: ../solutions/week-03/day-1.md)

## Self-check

- [ ] I can explain why interleaved thought/action/observation outperforms pure CoT or pure tool use.
- [ ] I can trace through a 3-step ReAct loop by hand and predict the next action.
- [ ] I can implement `ReactAgent.run()` from memory, including the stopping conditions.
- [ ] I can describe where tool dispatch happens and why the LLM never executes code.
- [ ] I can swap the mock LLM for a real Anthropic client without changing the loop logic.

---

## Supplementary links

- Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models" — https://arxiv.org/abs/2210.11610
- Anthropic Messages API reference — https://docs.anthropic.com/en/api/messages
- LangChain ReAct agent source (reference implementation) — https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022) — https://arxiv.org/abs/2201.11903
