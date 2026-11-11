# Phase 7 · Week 12 · Day 5 — Interview-Day Strategy, Offer Evaluation & Negotiation, 30/60/90-Day Plan

> Objective: Master interview-day execution, evaluate and negotiate an offer confidently, and build a 30/60/90-day plan for a new role.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Technical preparation is necessary but not sufficient. Interview-day execution, offer handling, and the first 90 days are career leverage points that most engineers underinvest in.

**Three distinct phases**

```
Phase A — Interview day (execution + mindset)
Phase B — Offer evaluation + negotiation (financial + career calculus)
Phase C — 30/60/90-day plan (ramp + impact + influence)
```

## How it works internally

### Phase A — Interview-day execution

**Before the day**

```
T-7 days:  Final mock loop on weakest area. No new topics.
T-2 days:  Revise your 6 STAR stories. Revise 2 capstone projects.
T-1 day:   Light review, sleep 8 hours. No cramming.
T-0 morning: Read company's recent engineering blog posts (2-3).
```

**During the interview**

| Situation | Response |
|---|---|
| Stuck on coding problem | "Let me think aloud. I'm considering [approach]. The edge case I need to handle is..." |
| Blank on a concept | "I haven't implemented that specific thing, but here's how I'd approach it from first principles..." |
| Clarifying question | Ask exactly one at a time. Confirm your assumption before coding. |
| Running out of time | "I won't finish the optimal solution in time — let me show the brute force and note where I'd optimise." |
| Behavioural memory blank | "Let me take 5 seconds to recall a concrete example." (Silence is OK once, briefly.) |

**After each round (between rounds)**

Write 2 sentences: what you said well, what you'd change. Carry that awareness into the next round. Don't ruminate — execute.

**Signs your loop is going well**

- Interviewer asks follow-up questions (engagement signal).
- "That's a good point" or "yes, exactly" reactions.
- They extend your time or skip to harder variants.

---

### Phase B — Offer evaluation

**Total Compensation (TC) components**

```
TC = Base salary
   + Annual bonus (% of base, varies by company/level)
   + Equity (RSUs/options × vesting schedule × estimated share price)
   + Benefits (health, 401k match, PTO value, remote allowance)
   + Sign-on (one-time; factor into year-1 only)
```

**Equity vesting math (4-year cliff+monthly)**

```
RSU grant: $400K over 4 years
Year 1: 25% cliff = $100K
Years 2-4: monthly vest = $300K / 36 = $8,333/month

If company grows 20%/yr, Year-4 equity value ≈ $400K × 1.20^3 ≈ $691K
(This is projection, not guarantee — liquidity matters)
```

**Evaluation framework (score 1–5 on each)**

| Dimension | Questions to ask |
|---|---|
| Compensation | Is base + equity competitive for level/market? |
| Growth | Will I be the dumbest person in the room? |
| Mission | Do I care about the product in 5 years? |
| Team | Who is my manager? What's the eng culture? |
| Risk | Company stage: pre-series A vs. public — liquidity risk? |
| Location | Remote-first vs. hybrid — real or lip service? |

**Negotiation protocol**

```
1. Never accept verbally on the call — "I'm very excited; I need until [date]."
2. Get competing offers; mention them neutrally: "I have another offer at $X."
3. Ask for what you want in one ask: "Based on my research and competing offers,
   I was hoping for $X base and $Y in RSUs."
4. Focus leverage: equity (most flexible), sign-on (easy one-time), title (level matters for next job).
5. Negotiating tone: collaborative, not adversarial. "Is there flexibility on..." not "I demand..."
6. Written confirmation of all verbal promises before accepting.
```

---

### Phase C — 30/60/90-day plan

**Day 1–30: Listen and learn**

```
Goals:
- Meet every stakeholder (eng, PM, design, data)
- Read all architecture docs, RFCs, post-mortems
- Ship one small bug fix or doc improvement (proves you can execute)
- Identify the top 3 pain points the team has

Output: 1-page written summary shared with manager at Day 30
```

**Day 31–60: Contribute meaningfully**

```
Goals:
- Own one feature or system improvement end-to-end
- Write at least 2 design docs for review
- Close at least 3 pull requests with tests
- Attend oncall rotation (shadow first week)

Output: Working feature in production; clear understanding of quarterly roadmap
```

**Day 61–90: Drive impact**

```
Goals:
- Identify and pitch one initiative you can lead
- Become the go-to person for one subsystem
- Propose process improvement (eval, observability, testing) with data
- Get mid-probation feedback from manager; adjust

Output: Published initiative proposal; measurable metric improvement
```

## Exercises

1. **Offer math**: You receive two offers:
   - Offer A: $180K base, $200K RSU (4-yr vest, 1-yr cliff), 15% bonus target, fully remote.
   - Offer B: $165K base, $400K RSU (4-yr vest, 1-yr cliff), 20% bonus target, 2 days/week in office.
   
   Compute Year-1 TC for both. Which has higher 4-year TC assuming 0% equity growth? Which at 30% annual growth? What non-financial factors might flip your choice?

2. **Negotiation script**: Write a 3-sentence email asking to improve Offer A's base to $195K and sign-on to $20K. Keep tone collaborative.

3. **30-day plan**: For an AI Engineer role at an enterprise software company, write your personal 30-day plan. Include 5 specific deliverables with success criteria.

4. **Interview-day drill**: You are 20 minutes into a coding round and completely stuck. Write the exact words you would say to your interviewer, then describe what you would do next.

5. **Competing offer leverage**: Draft the exact sentence you would say on a phone call to use a competing offer during salary negotiation. Keep it factual and non-combative.

(Solutions: ../solutions/week-12/day-5.md)

## Self-check
- [ ] I can compute TC across offers including equity projections
- [ ] I have a prepared negotiation script and have practiced it aloud
- [ ] I know exactly what I will do in the first 30 days of a new role
- [ ] I have practiced 3 "stuck" recovery responses for coding rounds

## Supplementary
- [levels.fyi — Market compensation data](https://www.levels.fyi)
- [Negotiating Your Salary — Josh Doody](https://fearlesssalarynegotiation.com/)
- [The First 90 Days — Michael Watkins](https://hbr.org/books/watkins)
