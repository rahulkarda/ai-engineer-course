# Solutions — Week 07 · Day 5: Debate, Collaboration & Conflict Resolution

## Exercise 1: ASCII confidence bar chart

```python
def print_confidence_chart(transcript: list[Argument]) -> None:
    print("\nConfidence per round:")
    # Group by round
    rounds: dict[int, list[Argument]] = {}
    for arg in transcript:
        rounds.setdefault(arg.round_num, []).append(arg)

    for r in sorted(rounds):
        print(f"Round {r}:")
        for arg in rounds[r]:
            bar_len = int(arg.confidence * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {arg.agent_name:15s} [{bar}] {arg.confidence:.2f}")
```

## Exercise 2: VoteResolver

```python
from collections import Counter

def vote_resolver(votes: list[str]) -> tuple[str, int, int]:
    """
    Returns (winner, winner_count, total_votes).
    Normalises to uppercase. Ties go to first alphabetically.
    """
    normalised = [v.strip().upper() for v in votes]
    counts = Counter(normalised)
    winner = counts.most_common(1)[0][0]
    return winner, counts[winner], len(votes)


# Test
def test_vote_resolver() -> None:
    verdict, count, total = vote_resolver(["YES", "NO", "YES", "YES", "NO"])
    assert verdict == "YES"
    assert count == 3
    assert total == 5
```

## Exercise 3: Three-way debate

```python
def _format_history_3way(transcript: list[Argument]) -> str:
    """Format history for 3 agents; same structure."""
    if not transcript:
        return "(none)"
    lines = []
    for arg in transcript:
        lines.append(
            f"[{arg.agent_name} round {arg.round_num}]\n"
            f"POSITION: {arg.position}\n"
            f"EVIDENCE: {'; '.join(arg.evidence)}\n"
            f"CONFIDENCE: {arg.confidence:.2f}"
        )
    return "\n\n".join(lines)


def run_3way_debate(
    config: "DebateConfig",
    agents: list[tuple[str, str]],  # list of (name, role)
    client: anthropic.Anthropic,
) -> "DebateResult":
    runner = DebateRunner(client)
    transcript: list[Argument] = []

    for r in range(1, config.max_rounds + 1):
        history = _format_history_3way(transcript)
        for name, role in agents:
            arg = runner._argue(name, role, config.question, history, r)
            transcript.append(arg)

    # Judge sees 3-way transcript
    full = _format_history_3way(transcript)
    judge_prompt = JUDGE_PROMPT.format(question=config.question, transcript=full)
    verdict_raw = runner._call(system="Impartial 3-way judge.", user=judge_prompt)
    # parse same as 2-way
    import re
    v_m = re.search(r"VERDICT:\s*(.+)", verdict_raw)
    w_m = re.search(r"WINNER:\s*(.+)", verdict_raw)
    from debate import DebateResult
    result = DebateResult(question=config.question, transcript=transcript)
    result.verdict = v_m.group(1).strip() if v_m else verdict_raw[:100]
    result.winner  = w_m.group(1).strip() if w_m else "unknown"
    return result
```

## Exercise 4: Early termination on high confidence

```python
for r in range(1, config.max_rounds + 1):
    # ... argue both agents ...

    # Early termination
    if arg_a.confidence >= 0.95 and arg_b.confidence >= 0.95 and r > 1:
        logger.info("early termination: both agents confidence >= 0.95 at round %d", r)
        break
```

## Exercise 5: Collaborative critique

```python
def collaborative_critique(
    topic: str,
    client: anthropic.Anthropic,
) -> str:
    """
    Agent A drafts → Agent B suggests → Agent C integrates.
    Returns final integrated text.
    """
    def call(system: str, user: str) -> str:
        resp = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    # Stage 1: Draft
    draft = call(
        "You are a technical writer. Write a clear 3-sentence explanation.",
        f"Explain: {topic}",
    )
    logger.info("collaborative | draft: %s...", draft[:60])

    # Stage 2: Suggestions
    suggestions = call(
        "You are an editor. Given a draft, list 3 specific improvements as bullet points. "
        "Be constructive, not destructive.",
        f"Draft:\n{draft}\n\nProvide 3 improvement suggestions:",
    )
    logger.info("collaborative | suggestions: %s...", suggestions[:60])

    # Stage 3: Integrate
    final = call(
        "You are a technical writer. Integrate the editor's suggestions into the draft. "
        "Return only the improved final version.",
        f"Original draft:\n{draft}\n\nSuggestions:\n{suggestions}\n\nIntegrated version:",
    )
    logger.info("collaborative | final: %s...", final[:60])
    return final
```

## Mock test for debate

```python
def test_debate_runner_mock() -> None:
    from unittest.mock import MagicMock
    import json
    from debate import DebateRunner, DebateConfig

    client = MagicMock()
    call_count = {"n": 0}
    responses = [
        "POSITION: Yes mandatory\nEVIDENCE:\n- readability\n- tooling\n- safety\nCOUNTER: slows novices\nCONFIDENCE: 0.7",
        "POSITION: No mandatory\nEVIDENCE:\n- dynamic typing\n- flexibility\n- duck typing\nCOUNTER: large codebases suffer\nCONFIDENCE: 0.6",
        "VERDICT: Optional but recommended\nREASONING: Both sides valid.\nWINNER: draw",
    ]

    def _create(**kwargs):
        r = MagicMock()
        r.content = [MagicMock(text=responses[call_count["n"] % len(responses)])]
        call_count["n"] += 1
        return r

    client.messages.create.side_effect = _create

    runner = DebateRunner(client)
    config = DebateConfig(question="Mandatory types?", max_rounds=1)
    result = runner.run(
        config,
        agent_a=("pro",  "Pro mandatory types."),
        agent_b=("anti", "Anti mandatory types."),
    )
    assert result.verdict != ""
    assert len(result.transcript) == 2  # one round: 2 arguments
```
