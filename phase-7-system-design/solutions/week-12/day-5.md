# Solutions — Week 12 · Day 5: Interview-Day Strategy, Offer Evaluation & Negotiation

## Exercise 1 — Offer math

### Year-1 TC calculation (0% equity growth)

**Offer A**
- Base: $180,000
- Bonus (15% target): $27,000
- Equity year 1 (25% cliff): $200,000 × 0.25 = $50,000
- Sign-on: $0 (not stated)
- **Year-1 TC = $257,000**

**Offer B**
- Base: $165,000
- Bonus (20% target): $33,000
- Equity year 1 (25% cliff): $400,000 × 0.25 = $100,000
- **Year-1 TC = $298,000**

### 4-year TC (0% equity growth)

**Offer A**
- Base + bonus over 4 years: ($180K + $27K) × 4 = $828,000
- Total equity: $200,000
- **4-year TC = $1,028,000**

**Offer B**
- Base + bonus over 4 years: ($165K + $33K) × 4 = $792,000
- Total equity: $400,000
- **4-year TC = $1,192,000**

Offer B wins at 0% growth by $164K over 4 years.

### 4-year TC (30% annual equity growth)

**Offer A**
- Year-1 vest ($50K) holds for 3 more years: $50K × 1.30³ = $87.6K
- Year-2 vest ($50K) × 1.30² = $84.5K (simplified average)
- Approximate total equity value ≈ $200K × 1.30^1.5 avg = ~$280K
- **4-year TC ≈ $1,108,000**

**Offer B**
- Same logic: $400K × 1.30^1.5 avg ≈ $560K
- **4-year TC ≈ $1,352,000**

Offer B still wins at 30% growth. The $200K extra equity grant dominates.

**Non-financial factors that might flip the choice**:

- Offer A is fully remote; Offer B requires 2 days/week in office. If commute costs $500/month + 3 hr/day of time value → this is meaningful.
- If Offer B is at a pre-IPO startup with illiquid equity, the $400K RSU is a projection, not cash — Offer A's lower but certain equity may be preferred.
- Growth trajectory: if Offer A is at a company that promotes faster or has a better technical team, the career compounding may outweigh the TC gap.

---

## Exercise 2 — Negotiation email

**Model email**

> Subject: Re: Offer — [Your Name]
>
> Hi [Recruiter Name],
>
> Thank you again for the offer — I'm genuinely excited about the role and the team. Based on my research into market rates for this level and a competing offer I'm evaluating, I was hoping we could explore a base of $195K and a $20K sign-on to help close the gap. I'm committed to joining if we can get there — would that flexibility be possible?

**Why this works**:
- First sentence: enthusiasm (signals you're likely to accept).
- Second sentence: uses market data + competing offer as neutral justification (not a demand).
- Third sentence: signals commitment conditional on the ask — reduces perceived risk for the recruiter.
- No multi-paragraph justification — ask is clear, concise, non-confrontational.

---

## Exercise 3 — 30-day plan model answer

**Role**: AI Engineer, Enterprise Software Company

| Day | Deliverable | Success criteria |
|---|---|---|
| 1–5 | Meet all 12 team members (eng + PM + design) | 1:1 notes from every meeting |
| 1–10 | Read all architecture docs and last 6 months of post-mortems | Can diagram the system from memory |
| 7–15 | Shadow 1 oncall rotation | Attended ≥ 1 alert triage; understand alert runbooks |
| 10–20 | Close one small bug fix or test gap PR | PR merged with reviewer approval |
| 25–30 | Write 1-page "what I learned" doc shared with manager | Received manager feedback |

---

## Exercise 4 — "Stuck" recovery script

**Exact words to say**

> "I want to be transparent — I'm stuck on the optimal approach. Let me think aloud about what I do know: the naive solution would be [O(n²) approach]. I know that's too slow for the given constraint. I'm thinking there might be a way to use [hash map / two-pointer / sliding window] to get this to O(n). Let me try building that direction."

**What to do next**:
1. Start implementing the brute-force solution while narrating — proves you understand the problem.
2. Once brute force is written, explicitly say: "This is O(n²) — let me optimise. The insight I need is..." and reason aloud from the brute force.
3. If still stuck after 5 more min: "I'm going to submit this O(n²) solution and note that I'd optimise by [specific idea] given more time."

Interviewers care much more about how you handle being stuck than whether you find the optimal solution.

---

## Exercise 5 — Competing offer leverage (model sentence)

**On the phone**:

> "I want to be straightforward — I have another offer at $195K base that I need to respond to by Friday. I'm more excited about this role, and I'd love to find a way to make the numbers work. Is there any flexibility on base or sign-on?"

**Key elements**:
- States the competing number (not vague "I have another offer") — gives the recruiter something concrete to bring to compensation committee.
- States preference clearly — not playing games.
- Specific ask with time constraint — creates gentle urgency without ultimatum.
- Open question — invites collaboration.
