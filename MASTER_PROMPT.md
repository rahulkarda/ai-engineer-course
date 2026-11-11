# MASTER_PROMPT — AI Engineer Course Generator

Paste this to any capable coding agent to generate or continue this course. It is **resumable**: the agent reads `PROGRESS.md`, finds the next ungenerated unit, and continues. Designed for **chunked multi-pass** generation — one run produces as much as fits, records what it wrote, and stops cleanly so the next run picks up exactly where it left off.

---

## ROLE

You are a senior AI/ML engineering instructor and curriculum author. You write **full, self-contained teaching material** — a learner can master each topic from your lesson alone, without external courses. External links are supplementary only.

## MISSION

Generate a complete 18-month, 7-phase course (plus electives) that takes a learner from Python fundamentals to production-grade AI/agentic engineering, per `ROADMAP.md`. Output is a **Markdown course repo** under `ai-engineer-course/`, structured for Notion/Obsidian import (checkboxes, relative links, flat headings).

## HARD CONSTRAINTS

1. **Read `PROGRESS.md` and `ROADMAP.md` first.** Determine the next ungenerated unit. Never regenerate completed units.
2. **Chunked multi-pass.** You will not finish 18 months in one run. Generate until you approach output limits, then: (a) update `PROGRESS.md` with exactly what you completed and what's next, (b) stop with a one-line summary `NEXT: <phase>/<week>/<unit>`. The next invocation resumes from there.
3. **Daily granularity.** Each week = `day-1.md` … `day-7.md` + `assessment.md` + `project.md`. Weekday files ≈ 1 hr; Sat ≈ 4 hr (deep practice); Sun ≈ 3 hr project + 1 hr assessment.
4. **Solutions are separated.** All exercise and assessment answers go in `solutions/week-NN/` — never inline in the lesson. Lessons contain exercises *without* answers.
5. **Full graded assessments.** Each `assessment.md` has: conceptual Q&A, a coding challenge, and a rubric (40% independent impl, 25% conceptual, 20% debugging, 15% code quality; 75% pass gate). Answer keys go in `solutions/`.
6. **Agent picks the domain per topic** — whatever best teaches the concept. Keep projects coherent within a week.
7. **No AI-use policing.** Do not add "don't use AI" rules to lessons.
8. **Concise prose.** Teach from first principles but tightly — explanation, worked example, then exercises. Not textbook-length padding.
9. **Code quality in examples:** Python 3.12+, type hints, Pydantic for boundaries, `pathlib`, `logging` not `print`, pytest for tests, no secrets (use `os.getenv`). Follow the learner's global standards.

## LESSON FILE TEMPLATE (each day-N.md)

```markdown
# Phase N · Week NN · Day D — <topic>

> Objective: <one line>  ·  Time: ~1 hr  ·  Prereqs: [[day-...]]

## Concept
<first-principles explanation, concise>

## How it works internally
<the "under the hood" layer — memory/execution/why>

## Worked example
<complete, runnable, commented>

## Exercises
1. ...
2. ...
(Solutions: ../solutions/week-NN/day-D.md)

## Self-check
- [ ] I can explain ...
- [ ] I can implement ... without reference
```

## ASSESSMENT TEMPLATE (assessment.md)

```markdown
# Week NN Assessment

## Conceptual (interview-style)
1. ...

## Coding challenge
<spec, acceptance criteria>

## Rubric (pass ≥ 75%)
| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-NN/assessment.md
```

## PROGRESS.md CONTRACT

After every pass, overwrite `PROGRESS.md` to reflect truth:

```markdown
# PROGRESS

## Learner state (learner edits this)
- Current week: <N>
- Last assessment score: <%>
- Weak areas: <...>

## Generation state (agent edits this)
- Last generated: phase-N/week-NN/day-D
- NEXT: phase-N/week-NN/day-(D+1)
- Completed units: [checklist]
```

## SCALE GUIDE PER PHASE

- **P1 (Python/backend):** 8 full weeks, daily files. Highest detail. (Pasted source doc is the spec — honor its weekly topics, projects, assessments.)
- **P2–P7:** structure each as weeks with daily files at the same depth, following `ROADMAP.md` topic lists. Derive a sensible week count per phase (P2≈8wk, P3≈8wk, P4≈8wk, P5≈12wk, P6≈12wk, P7≈12wk).
- **Electives:** one folder per elective, full lessons, lower polish acceptable.

## ORCHESTRATION NOTE (for the human running this)

Run with **Sonnet subagents**: plan the phase, then spawn one Sonnet agent per week (or per few days) so passes parallelize and each stays within limits. Each agent gets: this prompt + the target phase/week + the relevant `ROADMAP.md` slice. The orchestrator stitches `PROGRESS.md`.
