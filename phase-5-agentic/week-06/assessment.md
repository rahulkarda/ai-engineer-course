# Week 06 Assessment

## Conceptual (interview-style)

1. **Router agents & intent classification** — What distinguishes a router agent from a standard LLM chain? Describe two strategies for classifying incoming user intent and the trade-offs between them (e.g., keyword matching vs. LLM-based classification).

2. **Delegation patterns** — Compare "fan-out" delegation (broadcast to multiple workers) with "sequential" delegation (chain workers). When would you choose each? What are the failure-mode differences?

3. **Orchestrator-worker architecture** — Draw (or describe in text) the message-flow diagram for an orchestrator that routes to three specialist workers. How does the orchestrator know when all workers have finished, and how does it aggregate their results?

4. **Dynamic tool selection** — Explain how an agent can decide at runtime which tool to call rather than having a fixed tool list. What role does the intent-classification step play in constraining the available tool set per request?

5. **Confidence thresholds** — An intent classifier returns a probability distribution over three classes: `billing=0.55`, `technical=0.30`, `general=0.15`. Your threshold is 0.70. What should the system do, and why? What are the downstream consequences of setting the threshold too high versus too low?

6. **Human-in-the-loop (HITL)** — Describe three distinct trigger conditions that should pause an agentic workflow and wait for human input. How would you implement an asynchronous approval gate so the agent can process other tasks while waiting?

7. **Approval gates** — A user requests a $500 refund. The agent has verified the policy allows it but is configured to require human approval above $200. Describe the state transitions the workflow must go through: from initial request to final action. What data should be persisted between the "pending approval" and "approved" states?

8. **Escalation** — What is the difference between an "escalation" and a "fallback"? Describe a scenario where a confidence-based escalation to a human is preferable to falling back to a general-purpose LLM response, and explain the system design that makes escalation reliable (no lost requests, no duplicate processing).

---

## Coding challenge

### Specification

Build a minimal routing + HITL system in Python 3.12+ using Pydantic v2 and typed state. The system must:

1. **IntentClassifier** — A class that accepts a `query: str` and returns an `IntentResult` Pydantic model containing:
   - `intent: Literal["billing", "technical", "general"]`
   - `confidence: float` (0.0–1.0)

2. **Router** — A function `route(result: IntentResult) -> str` that:
   - Returns `"reject"` when `confidence < 0.6`
   - Otherwise returns the string name of the appropriate handler (`"billing_handler"`, `"technical_handler"`, `"general_handler"`)

3. **Handlers** — Three handler functions (`billing_handler`, `technical_handler`, `general_handler`) that each accept a `query: str` and return a `HandlerResponse` Pydantic model:
   - `handler: str`
   - `response: str`
   - `requires_approval: bool`

4. **ApprovalGate** — A class with method `check(response: HandlerResponse) -> ApprovalStatus` returning an `ApprovalStatus` model with:
   - `approved: bool`
   - `reason: str`
   - Gate must hold `"refund"` actions (any query containing the word `"refund"`) as `approved=False, reason="human_review_required"` until explicitly cleared
   - Provide a method `approve(response: HandlerResponse) -> ApprovalStatus` that returns `approved=True`

5. **Pipeline** — A function `run_pipeline(query: str, classifier: IntentClassifier, gate: ApprovalGate) -> PipelineResult` that wires the above together and returns a `PipelineResult` model with fields: `query`, `intent`, `confidence`, `routed_to`, `handler_response`, `approval_status`.

### Acceptance criteria

All five tests below must pass with a mocked LLM (no real API calls):

| # | Test | Expected outcome |
|---|------|-----------------|
| 1 | Classifier returns `confidence=0.4` for any query | Router returns `"reject"`; pipeline `routed_to == "reject"` |
| 2 | Classifier returns `billing` intent, `confidence=0.85` | Router returns `"billing_handler"`; `HandlerResponse.handler == "billing_handler"` |
| 3 | Query contains `"refund"`, billing intent, `confidence=0.9` | `ApprovalStatus.approved == False`, `reason == "human_review_required"` |
| 4 | Same refund query after `gate.approve()` called | `ApprovalStatus.approved == True` |
| 5 | Classifier returns `technical` intent, `confidence=0.75` | `routed_to == "technical_handler"`, `requires_approval == False` |

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-06/assessment.md
