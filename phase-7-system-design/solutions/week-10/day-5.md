# Solutions — Week 10 · Day 5: Feedback, Mentoring, Tech-Lead Signals & FDE Behavioral Prep

## Exercise 1 — SBI feedback messages

### Scenario A: Team member merges PRs without waiting for review

> **Situation:** In the last two sprints, three of your PRs were merged — including the payments module refactor on Tuesday — before any reviewer had looked at them.
>
> **Behavior:** When PRs are merged before review, the changes go to production without a second pair of eyes.
>
> **Impact:** Two of those three PRs introduced bugs that appeared in the next day's bug queue. More broadly, it signals to the team that our review process is optional, which makes it harder to hold the standard across the board.
>
> *What I'd like to discuss: Is there a reason review is getting skipped — time pressure, unclear ownership? I want to help fix the root cause.*

---

### Scenario B: Colleague interrupts others in sprint planning

> **Situation:** In today's sprint planning and in last week's session, you spoke over Amara three times while she was mid-sentence.
>
> **Behavior:** Each time, the conversation shifted to your point, and Amara's input was not picked up again.
>
> **Impact:** The team loses Amara's perspective — and from Amara's body language I could tell it was frustrating. Over time, people who get consistently interrupted stop contributing in group settings, which makes sprint planning less useful for everyone.

---

### Scenario C: Direct report consistently under-estimates

> **Situation:** Over the last four sprints, your estimates have averaged 2.1x the actual time taken — tasks scoped at 3 points typically finish in 6–7.
>
> **Behavior:** When I look at your estimation approach, it seems like you're sizing for the happy path without accounting for integration time, review cycles, or edge case handling.
>
> **Impact:** This creates planning problems for the team — when your items slip out of a sprint, it triggers scope renegotiation and affects the team's delivery commitments. I'd like to work on this with you directly.

---

## Exercise 2 — Mentoring questions

### (a) They haven't considered failure modes

> "What happens to a user's request if this service goes down during their checkout?"

(Do not say "you haven't thought about failure modes." Ask a specific question that requires them to think about a specific failure scenario. They will discover the gap themselves.)

### (b) Optimizing for cleverness over readability

> "If the engineer who's on call at 3 a.m. and has never seen this code needs to understand this function in 60 seconds, what would they find confusing?"

(This reframes cleverness as a runtime maintenance cost, not a style preference — and it's concrete enough that they can actually answer it.)

### (c) Not considering dependency unavailability

> "What does the user experience look like if the recommendation service you're calling returns a 503?"

(Specific service, specific error code, specific user impact. This is more productive than "what if it fails?" because it forces them to trace a concrete failure path.)

---

## Exercise 3 — Tech-lead signal story (worked example)

**Title:** Eliminated the team's async communication bottleneck by moving architectural decisions to design docs  
**Competency:** Leadership / multiplier (P)

**S:** On a seven-person team, all significant architectural decisions were made in meetings I ran. Engineers blocked on a decision would wait until I had availability — sometimes 2–3 days — or make the decision themselves and undo it later. The team was shipping 40% of planned story points each sprint.

**T:** I needed to change how decisions were made without imposing process that would slow things further.

**A:**
- Wrote a two-page "lightweight design doc template" and used it for the next three decisions myself, sharing the draft in Slack before any meeting.
- Made it explicit that comments on a doc by EOD would constitute approval unless raised otherwise — shifting from synchronous to async by default.
- Ran two pairing sessions with junior engineers to help them write their first design docs independently.

**R:** Within 6 weeks, 80% of team design decisions were documented and resolved without a meeting. Sprint velocity increased from 40% to 68% of planned points over the same period. One junior engineer who had been waiting for my input on every decision shipped an independent feature end-to-end.

---

## Exercise 4 — FDE scenario response (model answer)

> "I understand the goal is 50ms P99 inference on your current hardware. I need to be direct: at your request volume, that target isn't achievable on CPU-only servers with this type of model — the compute bottleneck is real and not configurable away.
>
> Here are two paths that do work:
>
> 1. **GPU inference (cloud):** We can serve the model on GPU instances at approximately $X/month at your volume. This meets the 50ms target. We can run a 2-week pilot to validate cost before you commit.
>
> 2. **Model distillation on CPU:** We can reduce the model size by roughly 4x through distillation, which would bring CPU latency to approximately 90–120ms P99. This doesn't meet your 50ms target but may meet the user-facing need depending on your use case.
>
> Which of these fits your constraint better, or should we dig into the actual user experience requirement to see if there's a third path?"

**What this answer does:** Names the constraint clearly without hedging. Offers two concrete paths with cost/tradeoff information. Ends with a question that opens the door to further discovery — the customer's stated constraint (50ms, CPU) may not be their actual constraint (user experience, cost ceiling).
