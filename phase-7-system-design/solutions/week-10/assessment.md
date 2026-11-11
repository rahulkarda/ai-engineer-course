# Solutions — Week 10 Assessment Answer Key

## Part 1 — Conceptual questions

### 1. Four STAR elements and failure modes

| Element | Common failure mode |
|---------|-------------------|
| **Situation** | Too long — spends 60%+ of the answer on context, leaving no time for actions and results |
| **Task** | Uses "we" instead of "I" — obscures individual contribution |
| **Action** | Too high-level ("I worked on the solution") — gives no insight into thinking or specific steps |
| **Result** | No numbers — "things improved" is unverifiable; "P95 latency dropped from 4.2s to 0.3s" is memorable |

Secondary failure mode: no learning sentence in the Result — omitting reflection signals lack of growth mindset at senior levels.

---

### 2. What changes between audiences

- **Engineering team → PM:** Remove mechanism names; replace with user-visible outcome and a business metric. The "why it matters" sentence changes from "this eliminates the N+1 query problem" to "this cuts page load time from 4 seconds to 0.3 seconds."
- **PM → CTO:** Remove all feature-level detail; replace with portfolio-level framing. Include cost, timeline, risk mitigation, and a clear ask. End with a decision, not a recommendation.

The one thing that stays constant: the conclusion leads (BLUF). The change is in the supporting evidence — technical for engineers, business outcomes for PMs, strategic alignment for executives.

---

### 3. BLUF principle

BLUF (Bottom Line Up Front) means stating your conclusion in the first sentence, then providing supporting reasoning. It improves communication because: (1) readers can decide whether to read further immediately, (2) time-constrained reviewers get the signal even if they stop after one sentence, and (3) it forces the writer to be clear on their conclusion before elaborating — which often reveals that the conclusion is not fully formed.

Most technical writers build up context and reveal the conclusion last. This pattern is appropriate for persuasive narratives (suspense, courtroom arguments) but not for asynchronous technical communication where the reader is resource-constrained.

---

### 4. Code review labels

| Label | Signal to author |
|-------|----------------|
| **blocking:** | Must be resolved before merge — correctness or security issue |
| **suggestion:** | Take-it-or-leave-it improvement; author makes the call |
| **nit:** | Minor style point; reviewer flags it but doesn't expect it to block |
| **question:** | Genuinely unclear — reviewer is asking, not criticizing |
| **praise:** | Explicitly positive — names what the reviewer learned or appreciated |

(Other valid labels: `important:` — strong recommendation; discuss if you disagree.)

---

### 5. Design doc vs. implementation ticket

A **design doc** argues *why* a particular approach is correct and records the tradeoffs evaluated. It is written before coding begins, enables async feedback, and creates a durable decision record. Write one when the change is cross-team, hard to reverse, or when reasonable engineers would disagree.

An **implementation ticket** specifies *what* to build: acceptance criteria, scope, definition of done. It assumes the *why* has been decided. Write one for work that is well-understood and contained within your team.

---

### 6. SBI feedback model

SBI = Situation (specific event), Behavior (observable action), Impact (effect on team/product/person). It prevents trait claims: saying "you were reckless" is unprovable and triggers defensiveness. "In Tuesday's deploy you merged without review, and the resulting bug took two hours to diagnose" is observable, specific, and separates the behavior from a judgment about the person's character. Trait feedback is hard to act on; SBI feedback names a specific behavior that can be changed.

---

### 7. Tech lead behavioral signals vs. strong IC

| Signal | Meaning |
|--------|---------|
| **Multiplier, not bottleneck** | Enables others' work rather than hoarding technical decisions |
| **Proactive risk surfacing** | Flags risk to stakeholders before incidents, not during |
| **Writes for async** | Decisions are documented so the team doesn't depend on verbal communication |
| **Calibrates confidence** | Says "I don't know, here's how I'd find out" — doesn't bluff |
| **Owns team output** | Takes responsibility for quality across the team, not just personal contributions |

A strong IC optimizes personal output. A tech lead optimizes team output — which sometimes means doing less coding and more unblocking, documentation, and stakeholder management.

---

### 8. FDE-specific behavioral competency

The most distinctive FDE competency is **distinguishing what a customer asks for from what actually solves their problem** (scope management / requirements discovery). In standard software roles, requirements come filtered through product managers. In FDE roles, customers describe their pain directly, often in the form of a proposed solution ("we need on-premise deployment") rather than a need ("we need data residency compliance"). The FDE must hear the solution, probe for the underlying need, and then evaluate whether the stated solution is the best path — often proposing a simpler alternative that the customer hadn't considered. This requires technical credibility (so the customer trusts the alternative) and customer empathy (so they don't feel unheard).

---

## Part 3 — Design doc review answer key

**Review comment 1 — blocking:**
> **blocking:** The "Problem" section doesn't quantify the current performance level or the cost of inaction. "The ML inference API is slow" and "users are complaining" are not actionable — what is the current P99 latency? What is the SLA target? How many users are affected? Without this, reviewers cannot evaluate whether the proposed solution is correctly scoped or assess whether the "Goals" section addresses the actual problem.

**Review comment 2 — important:**
> **important:** "Alternatives considered: None considered at this time" is not acceptable in a design doc. Before a caching layer decision, you should evaluate: (1) in-process caching vs. Redis (operational overhead tradeoff), (2) CDN-level caching (appropriate only if responses are user-agnostic), (3) request deduplication. The choice of Redis should be argued, not assumed. This is a blocking concern if the doc goes to the RFC stage.

**Review comment 3 — question:**
> **question:** The proposed solution says "cache responses for 1 hour" — what is the basis for the 1-hour TTL? If the ML model is retrained daily, a 1-hour TTL means some users will get stale predictions for up to 1 hour after a model update. Is that acceptable? Has this been discussed with the product team?

**Single most important structural gap:** The Goals section — "Make the API faster" and "Reduce load on the model server" — are activity goals, not outcome goals. They have no measurable targets, so there is no way to declare success or failure. Every goal in a design doc must be testable: "Reduce P99 inference latency from X to Y" and "Reduce model server request rate by Z%" are testable goals.

---

## Part 4 — Communication rewrite model answer

**Original (technical, buried conclusion):**
> "Hey team — so we ran the ablation study on the retrieval component last week. The MRL vs. standard dense embedding comparison showed a 12-point delta in recall@10 with a 40ms P95 regression. Given that the production SLA is 120ms and we're currently at 80ms P95, the 40ms headroom is exactly at the limit. We probably shouldn't ship the MRL approach in this release but maybe revisit after the infrastructure upgrade."

**Rewrite for PM (BLUF, ~100 words):**

> Recommend NOT shipping the improved search in this release. The new search approach finds 12% more relevant results, which is good — but it also makes each search 40ms slower. We're currently 40ms away from our maximum allowed response time. Shipping this would put us exactly at the limit with no safety buffer; any traffic spike could breach our performance commitment. Recommend holding it until the infrastructure upgrade next quarter, which will restore our headroom. The feature is ready; it's a timing decision only. Let me know if you want to discuss.

**What was sacrificed and why:** The technical mechanism (MRL, dense embeddings, recall@10) was removed entirely, because a PM cannot make a go/no-go decision based on those terms. What the PM needs is: the magnitude of the benefit, the nature of the risk, the risk quantification, and a clear recommended action. The sacrifice is appropriate because the PM's decision does not require understanding the mechanism — only the outcome and the risk. The technical detail belongs in the design doc, not the decision communication.
