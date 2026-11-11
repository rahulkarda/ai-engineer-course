# Phase 7 · Week 10 · Day 5 — Feedback, Mentoring, Tech-Lead Signals & FDE Behavioral Prep

> Objective: Demonstrate the behavioral signals of a tech lead and forward-deployed engineer — giving/receiving feedback, mentoring, and customer-facing communication.  ·  Time: ~1 hr  ·  Prereqs: [[week-10/day-4]]

## Concept

### Giving and receiving feedback

Feedback is the primary mechanism by which teams improve. Two distinct modes:

**Giving feedback:**

The SBI model (Situation → Behavior → Impact) is a clean, defensible structure:

- **Situation:** "In yesterday's incident review..."
- **Behavior:** "...you attributed the root cause to the on-call engineer's mistake..."
- **Impact:** "...which shut down the conversation about the systemic gap in our runbook."

SBI keeps feedback anchored to observable events rather than personality traits. "You were dismissive" is a trait claim. "In the incident review you interrupted twice and attributed the problem to individual error before the group had finished analyzing" is observable behavior.

**Receiving feedback:**

Senior engineers are expected to model how to receive feedback well:

1. **Listen completely** before responding. Resist the urge to explain or defend.
2. **Acknowledge** what you heard: "What I'm hearing is that my PR comments came across as blocking when I intended them as suggestions."
3. **Ask one clarifying question** if needed: "Can you give me a specific example?"
4. **Close with intent:** "I'll label my PR comments with severity going forward."

Defensive receiving makes feedback expensive to give — it will stop coming.

### Mentoring

Mentoring is not teaching. Teaching transfers information. Mentoring develops judgment.

The key distinction: a mentor's job is to ask questions that help the mentee develop their own reasoning, not to hand them answers.

**Mentoring anti-patterns:**
- Solving the problem for them.
- Giving feedback only when things go wrong.
- Setting stretch goals without scaffolding.

**Mentoring patterns that work:**
- Ask "What have you tried?" before offering anything.
- After they describe a decision, ask "What would you do differently?" before offering your view.
- Point at a gap explicitly: "You mentioned the latency issue but didn't bring up data consistency — what's your thinking there?"

### Tech-lead behavioral signals

A tech lead is not the "best coder on the team." The behavioral signals interviewers look for:

| Signal | Evidence |
|--------|---------|
| **Multiplier, not bottleneck** | Reviews code quickly, unblocks others, makes architectural decisions without hoarding them |
| **Proactive risk surfacing** | Flags risks to stakeholders before they become incidents |
| **Writes for async** | Decisions documented, not just communicated verbally; new members can onboard without scheduling time with you |
| **Calibrates confidence** | Says "I don't know, but here's how I'd find out" rather than guessing |
| **Owns team output** | Takes responsibility for team quality, not just personal contributions |

In behavioral interviews, tech-lead signals appear in stories where you: changed a process, made a call that others disagreed with, unblocked a team, or grew a junior engineer.

### FDE (Forward-Deployed Engineer) behavioral signals

A forward-deployed engineer operates at the intersection of engineering and customer engagement. FDE-specific behavioral competencies:

| Competency | What it looks like |
|-----------|-------------------|
| **Technical credibility with customers** | Can explain tradeoffs clearly to a technical buyer without overselling |
| **Scope management** | Distinguishes "what the customer asked for" from "what solves their problem" |
| **Handling objections** | Addresses concerns with data, not just reassurance |
| **Escalation judgment** | Knows when a customer request is a product gap vs. a configuration issue |
| **Closing the loop** | Follows through on commitments made in customer conversations |

## How it works internally

### The feedback conversation arc

A structured feedback conversation (for giving critical feedback):

1. **Open with context** — "I want to share something I've noticed and get your reaction."
2. **Deliver SBI** — Situation, Behavior, Impact (2–3 sentences).
3. **Pause** — Let it land. Do not fill the silence.
4. **Ask for their perspective** — "What's your read on this?"
5. **Discuss** — Listen before responding to their take.
6. **Agree on next step** — Specific, observable, time-bound.

### The mentoring conversation arc

1. **Ask about their goal first** — "What are you trying to figure out?"
2. **Listen for their current model** — Where are the gaps in their reasoning?
3. **Ask a gap-exposing question** — One question that surfaces the missing consideration.
4. **Offer one concrete resource or framework** — Not more than one; too much is overwhelming.
5. **Set a follow-up** — "Come back and tell me how the approach worked."

## Worked example

### Behavioral question: "Tell me about a time you mentored someone."

**Strong STAR answer:**

**S:** A junior ML engineer on my team was stuck on a feature importance analysis task. After two days, she said the model "wasn't working" — but the metric she was optimizing was model accuracy, which had plateaued. She was about to abandon the approach.

**T:** As tech lead, I wanted to help her unblock, but more importantly I wanted her to develop the diagnostic skill herself, not just get the answer from me.

**A:**
- Asked her to walk me through her analysis out loud — I said nothing, just listened. Within five minutes, I could see she was optimizing the wrong metric (accuracy on a class-imbalanced dataset).
- Instead of pointing this out, I asked: "If the positive class is only 2% of your data, what does 98% accuracy tell you?" She paused, then said "nothing useful."
- Pointed her to the scikit-learn documentation on `classification_report` rather than explaining it myself. She came back the next day having reworked the analysis entirely.
- One month later, she caught the same imbalance issue in a colleague's PR and wrote the comment herself.

**R:** She shipped the feature with the corrected evaluation. The experience compelled me to add "evaluation metric selection" to our team's ML code review checklist, which has since caught two similar issues. I learned that the most durable mentoring moment is when the mentee explains a principle to someone else — that's when you know the transfer actually happened.

---

### FDE behavioral question: "Tell me about a time a customer asked for something you knew was the wrong solution."

**Strong STAR answer:**

**S:** A large enterprise customer requested a custom on-premise deployment of our feature store, citing data residency concerns. The request would have taken 8 weeks of engineering time and created a long-term support burden.

**T:** I was the technical account lead. I needed to address their real concern without committing to a custom deployment that would have diverged from our product roadmap.

**A:**
- Asked them to describe their data residency requirement in detail. The actual concern was that EU customer PII could not leave their country's data center — not that they required on-premise in general.
- Showed them our existing EU region deployment with a data residency configuration flag we had added 6 months earlier. It addressed their compliance requirement exactly.
- Wrote a one-page technical summary of the EU region configuration for their security team to review, which they approved in 3 days.

**R:** Closed the deal with no custom engineering. The configuration flag was something they hadn't known existed — a discovery failure, not a product gap. I instituted a "requirements first, solution second" interview template for all new enterprise evaluations that has since reduced mismatched scope proposals by ~60%.

## Exercises

1. **SBI practice.** Write an SBI feedback message for each scenario:
   - A team member keeps merging PRs without waiting for review.
   - A colleague interrupts others repeatedly in sprint planning.
   - A direct report consistently under-estimates tasks in planning.

2. **Mentoring question design.** For each gap below, write one question (not a statement) that would surface the gap for a mentee without giving them the answer:
   - They haven't considered failure modes in their proposed design.
   - They are optimizing for code cleverness at the expense of readability.
   - They haven't thought about what happens when the service they depend on is unavailable.

3. **Tech-lead signal story.** Write a STAR story (200 words) that demonstrates the "multiplier, not bottleneck" tech-lead signal. The story should involve you changing a process or decision that unblocked others.

4. **FDE scenario.** A customer says: "We need real-time inference at 50ms P99, but we want to run it on our existing CPU-only servers." You know this is technically impossible at their request volume on that hardware. Write a 150-word response that: acknowledges their constraint, explains the gap clearly without jargon, and offers two paths forward.

(Solutions: ../solutions/week-10/day-5.md)

## Self-check

- [ ] I can deliver SBI feedback on a specific behavior without making trait claims
- [ ] I can describe the mentoring conversation arc and the anti-patterns to avoid
- [ ] I can name five behavioral signals of an effective tech lead
- [ ] I can identify FDE-specific behavioral competencies
- [ ] I have a STAR story ready for "tell me about a time you mentored someone"

## Supplementary links

- [Kim Scott — Radical Candor framework](https://www.radicalcandor.com/our-approach/)
- [Will Larson — Staff Engineer (multiplier patterns)](https://staffeng.com/book)
- [Palantir FDE role description (reference for competencies)](https://www.palantir.com/careers/teams/forward-deployed-engineers/)
