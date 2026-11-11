# Week 11 Day 6 Solutions — Build Session Reference Outputs

These are model outputs for the 4-hour sprint. Compare your build output against these examples — the goal is not identical text but equivalent depth and specificity.

---

## Sample Decision Log: Multi-Agent Operations Platform (5 entries)

---

**Decision 1: Orchestration framework selection**

**Date:** Early in project (pre-alpha)

**Context:** Needed a framework to coordinate multiple LLM agents executing in sequence and parallel. Options were custom orchestration, LangChain agent executor, AutoGen, CrewAI, and LangGraph.

**Options:**
- Custom loop: full control, no dependencies, ~3 weeks to build equivalent functionality
- LangChain agent executor: familiar, but single-agent model, poor support for complex state
- AutoGen: strong multi-agent support, but Microsoft-opinionated, limited production tooling
- CrewAI: clean API, but black-box internals, hard to debug state transitions
- LangGraph: explicit graph structure, typed state, checkpointing, good observability surface

**Decision:** LangGraph

**Rationale:** Typed state and explicit graph structure were non-negotiable for debugging multi-agent workflows. CrewAI abstracted away too much. AutoGen's communication model didn't map cleanly to our tool-gateway architecture. LangGraph's checkpointing support also meant we got retry and resume for free.

**Cost:** Steeper learning curve than CrewAI for simple workflows. First week slower.

**Outcome:** Graph structure made debugging state issues tractable. Would choose again.

---

**Decision 2: Tool execution model**

**Date:** During alpha

**Context:** Agents needed access to tools (web search, code execution, file I/O). Two models: each agent has its own tool registry, or all agents route through a shared gateway.

**Options:**
- Distributed (per-agent tool registry): simpler, lower latency, no new service
- Centralized MCP gateway: all tool calls route through one server, adds latency, adds complexity

**Decision:** Centralized MCP gateway

**Rationale:** Per-agent tool registries would scatter credential management across N agents. Audit logging would require instrumenting every agent. Rate limiting would be inconsistent. The security surface with distributed tools was unacceptable for production. Centralized gateway adds ~20ms per tool call — acceptable for workflows where tool calls take 200ms–30s anyway.

**Cost:** Added a new service (MCP server), a new failure mode (gateway unavailable = all tools unavailable), and ~20ms latency overhead.

**Mitigation:** Redis-backed task queue for gateway; circuit breaker pattern for tool call failures.

**Outcome:** Security audit time reduced from estimated 2 days to 2 hours. Single credential rotation path. Would choose again.

---

**Decision 3: Concurrent state management**

**Date:** During beta, after observing silent corruptions

**Context:** Two agents modifying shared workflow state concurrently (one updating a task list, another updating the research context) produced state corruption. LangGraph's built-in checkpointer serializes writes but doesn't prevent read-modify-write races at the application layer.

**Options:**
- Serialize all agent execution (no parallelism): safe, simple, defeats the point of multi-agent
- Distributed lock (Redis SETNX): prevents concurrent writes, adds ~5ms, potential deadlock risk
- Optimistic locking (version tokens): reads include version token, writes validate before committing, stale writes fail fast

**Decision:** Optimistic locking

**Rationale:** Serialization was unacceptable for performance. Distributed locks introduced deadlock risk we didn't want to manage. Optimistic locking aligned with the actual access pattern: writes are infrequent, reads are frequent, conflicts are rare. Conflict rate in testing was < 2% — stale write triggers agent re-planning, total cost is one retry per conflicted operation.

**Cost:** Added version token logic to state reads and writes. Added re-planning path for agents that hit a stale write.

**Outcome:** Zero silent state corruptions post-implementation. Conflict rate 1.8% in load testing.

---

**Decision 4: Agent communication model**

**Date:** During alpha

**Context:** Agents needed to pass information between each other (researcher passes findings to planner; planner passes task breakdown to coder). Options: shared state, message passing, direct function calls.

**Options:**
- Direct function calls: tight coupling, hard to add/remove agents
- Message queue (Celery/Redis): decoupled, asynchronous, but complex for sequential workflows
- Shared typed state (Pydantic models in LangGraph state): single source of truth, readable by all agents, explicit structure

**Decision:** Shared typed state

**Rationale:** For workflows with sequential dependencies (A must complete before B starts), message queues add complexity without benefit. Shared state fits the directed graph execution model. Pydantic enforcement means type mismatches fail at validation, not silently at runtime.

**Cost:** All agents must agree on the state schema. Adding a new field requires updating the schema definition. Less flexible than freeform dicts.

**Outcome:** Caught 4 type mismatches during development that would have been silent runtime bugs with untyped dicts. Would choose again.

---

**Decision 5: Observability strategy**

**Date:** Post-beta, during production hardening

**Context:** Debugging agent workflow failures required understanding: which agent ran, in what order, what tool calls were made, what state was at each step. Standard logging was insufficient — logs from concurrent agents were interleaved and hard to correlate.

**Options:**
- Structured JSON logging with correlation IDs: low overhead, moderate debuggability
- OpenTelemetry distributed traces: high debuggability, adds dependency and instrumentation overhead
- LangSmith: purpose-built for LangChain/LangGraph, strong UX, external dependency, cost

**Decision:** OpenTelemetry

**Rationale:** LangSmith would have been faster to set up, but it's an external dependency with per-trace costs that don't scale well to high-volume production. OpenTelemetry is portable (integrates with any observability backend), free at instrumentation level, and traces to Tempo + Grafana gave us a UI we already operated. The cost was 2 weeks of instrumentation work we hadn't planned.

**Lesson learned:** Should have been decided in Week 1 of the project. Retrofitting tracing required touching every agent and every tool gateway call.

**Outcome:** Full distributed traces for all agent workflows. Can debug a 10-step workflow failure in minutes instead of hours.

---

## Sample Scaling Story: Multi-Agent Platform

**Prototype (Week 1):**
- 2 agents (researcher + planner), hardcoded tools, no real state management
- Single Python script, LangChain, OpenAI API, no persistent storage
- Handled 1 request at a time, no concurrency
- Tool calls: direct API calls, credentials hardcoded in script (placeholder values)
- Latency: ~15s per workflow (dominated by LLM calls)
- Cost: ~$0.25/workflow (unoptimized prompts)
- No tests, no monitoring

**Alpha (end of Phase 1, demo-ready):**
- 3 agents (researcher, planner, coder) with LangGraph orchestration
- Typed state (Pydantic), basic error handling
- MCP server (prototype): centralized tool routing, no auth yet
- Concurrency: 5 simultaneous workflows (manual testing)
- Latency: ~12s per workflow (prompt optimization)
- No production deployment — ran locally

**Beta (Phase 2, integration-tested):**
- 5 agent types, dynamic agent selection based on task type
- MCP gateway: per-agent credential scoping, audit logging, rate limiting
- Optimistic locking for concurrent state management
- Async execution: 20 concurrent workflows tested, no corruptions
- FastAPI wrapper, Dockerized
- Latency: p50 8s, p95 18s (still dominated by LLM calls)
- Cost: $0.15/workflow (caching, better prompts)
- 85% test coverage, CI/CD pipeline

**Production target:**
- 50 concurrent workflows (tested and validated)
- p95 < 800ms for orchestration overhead (excluding LLM latency)
- LLM latency addressed by: streaming responses, parallel tool calls where possible, caching repeated tool results
- Kubernetes deployment: 3 replicas, HPA on CPU + request queue depth
- Full OpenTelemetry tracing, Prometheus metrics, Grafana dashboards
- Cost: $0.09/workflow (batched LLM calls, aggressive caching of research results)
- Target SLA: 99.5% uptime, < 5s workflow latency p50 for typical tasks

**The numbers that matter:**
- Throughput: 1 workflow at a time → 50 concurrent
- Reliability: no error handling → 99.5% SLA
- Cost: $0.25/workflow → $0.09/workflow (64% reduction)
- Security: credentials in script → zero-trust MCP gateway with per-agent scoping

---

## Sample Deep-Dive Q&As: Multi-Agent Platform (5 examples)

---

**Q1: "How do you prevent one agent from accessing tools it shouldn't use?"**

A: "The MCP gateway enforces per-agent tool permissions. When an agent registers, it's assigned a scope — a list of allowed tool namespaces. The gateway validates the scope on every tool call before executing. If the researcher agent tries to call the code execution tool, it gets a 403, not an execution. The scope is defined in the agent configuration, not in the agent's own logic — agents can't grant themselves permissions. We tested this with adversarial prompts asking agents to use out-of-scope tools; the gateway rejected all of them."

---

**Q2: "What happens when an agent fails midway through a 10-step workflow?"**

A: "LangGraph's checkpointer saves state after each node execution. If a node fails, we have the full state up to that point. We implemented three recovery paths: (1) retry the failed node with exponential backoff for transient errors (API timeouts, rate limits); (2) route to an error-handling node that attempts a simplified version of the failed task; (3) surface to a human-review queue if both fail. Most failures (80%+) in testing were transient LLM API errors that resolved on first retry. The checkpoint means we never re-run the first 9 steps of a 10-step workflow — we resume from the failure point."

---

**Q3: "How do you know the agents are producing correct outputs?"**

A: "Three layers: structural validation (Pydantic models enforce output schema — if an agent returns malformed JSON, it fails before being passed downstream), consistency checks (later agents in the workflow validate that their inputs are coherent with prior agents' outputs), and optional human-in-the-loop review nodes we can insert at configurable checkpoints. We don't have a general 'is this correct' evaluator — that's a hard problem. For specific task types (code generation), we run the output, which is an objective quality signal."

---

**Q4: "Why LangGraph over a simpler orchestration approach like a for-loop over agent calls?"**

A: "A for-loop breaks on three things we needed: conditional branching (sometimes the researcher should loop back to gather more data before the planner proceeds — a for-loop can't model that cleanly), fault tolerance (checkpointing and resume require state management the for-loop doesn't provide), and observability (graph edges are explicit and instrumentable; a for-loop's control flow lives in Python logic that's hard to trace). For a 2-agent sequential workflow, a for-loop is fine. For 5 agents with conditional paths, shared state, and production reliability requirements, the graph model justified the complexity."

---

**Q5: "How did you handle the case where agents produce conflicting information?"**

A: "We have a conflict resolution node in the graph that fires when two agents' outputs are inconsistent. The node uses a structured comparison prompt: it presents both outputs to the LLM with explicit instructions to identify the conflict and reason about which is more reliable given the sources. The resolution result is written back to shared state with a confidence score. For high-stakes decisions (above a threshold), we route to a human-review queue instead of auto-resolving. In practice, conflicts were rare (< 5% of workflows in testing) — most came from the researcher and planner having different versions of a rapidly-changing fact. The resolution worked in 87% of auto-resolve cases based on manual audit of 100 examples."

---

## Sample 2-Minute Pitch Script: Multi-Agent Operations Platform

**[30 sec — Problem and context]**
"The project was a production-grade platform for multi-agent LLM workflows. The problem I was solving: existing agent frameworks, including LangChain and CrewAI, don't have production primitives — no credential isolation, no audit logging, no concurrent state safety. They're built for demos, not for systems where a bug means a leaked API key or a corrupted workflow state."

**[45 sec — What I built and the key architectural choice]**
"I built on LangGraph for orchestration — because it gives you explicit graph structure and typed state — and added a Model Context Protocol server as a centralized tool gateway. That gateway is the architectural choice I'm most proud of: every tool call from every agent routes through it. It enforces per-agent permissions, rate limits, and logs every call with agent identity and inputs. The trade-off was ~20ms latency overhead per tool call. Worth it: it reduced the security audit surface from N agents to one server."

**[30 sec — The hard technical problem]**
"Hardest problem: concurrent state modification. Two agents reading and writing the same shared state simultaneously produced silent corruptions that LangGraph's checkpointer didn't catch. Fixed with optimistic locking at the application layer — reads return a version token, writes validate it, stale writes fail fast and trigger agent re-planning. Conflict rate in testing was 1.8%, and every conflict resolved cleanly."

**[15 sec — Results]**
"Platform handles 50 concurrent workflows at p95 under 800ms orchestration overhead. Zero credential leaks in security testing. Audit log made a 2-hour security review take 2 hours instead of 2 days."

*Total: approximately 115 seconds. Adjust pacing for the room.*
