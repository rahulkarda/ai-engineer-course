## Exercises

1. Add a **confidence convergence plot**: after the debate, print each agent's confidence per round as an ASCII bar chart.
2. Implement a `VoteResolver` that takes N agent outputs (each a string "YES" or "NO") and returns the majority verdict with the vote count.
3. Modify `DebateRunner` to support 3-way debate (three agents + judge). How does `_format_history` change?
4. Implement early termination: if both agents reach `confidence >= 0.95` in the same round, skip remaining rounds.
5. Add a `collaborative_critique` function: Agent A drafts a short paragraph, Agent B suggests improvements as bullet points, Agent C integrates them into a final version.

(Solutions: ../solutions/week-07/day-5.md)

## Self-check

- [ ] I can explain why debate improves output quality over single-agent
- [ ] I can implement a structured debate loop with convergence detection
- [ ] I understand the five conflict resolution strategies and when to use each
- [ ] I can implement confidence-weighted voting across N agents

## Supplementary links

- [Society of Mind — Minsky](https://en.wikipedia.org/wiki/Society_of_Mind)
- [Multi-agent debate paper (MIT/Google, 2023)](https://arxiv.org/abs/2305.14325)
- [LATS — Language Agent Tree Search](https://arxiv.org/abs/2310.04406)
