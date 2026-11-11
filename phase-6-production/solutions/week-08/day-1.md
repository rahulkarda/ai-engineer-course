# Week 8, Day 1 — Cloud Deployment Models: Solution Key

## Exercise 1: Classify 5 AI Deployment Scenarios

**Scenario A: Fine-tuned LLM inference with 50 ms p99 latency SLA, dedicated GPU, 24/7 uptime**
- **Classification: IaaS** (e.g., EC2 P4d, GCP A3, Azure NDv4)
- Why: GPU instance types require direct hardware access. PaaS GPU options exist but add abstraction that hurts tail latency. Serverless can't guarantee cold-start-free GPU provisioning. You own the instance → you control driver, CUDA version, and scheduler.

**Scenario B: RAG preprocessing pipeline — nightly ingestion of 10 k documents, runs 2 hours/day**
- **Classification: Serverless or managed batch** (AWS Batch, GCP Cloud Run Jobs, Lambda with Step Functions)
- Why: Intermittent workload with no idle traffic. Paying for compute 24/7 is pure waste. Startup latency is irrelevant (batch job). Serverless or container jobs charge per-CPU-second.

**Scenario C: Multi-tenant AI SaaS API — variable traffic, team wants zero ops overhead**
- **Classification: PaaS / managed container service** (Cloud Run, ECS Fargate, Azure Container Apps)
- Why: PaaS abstracts OS patching, autoscaling, and load balancing. Container image gives reproducibility. Team avoids node management while retaining stateless API control.

**Scenario D: Real-time embedding service called synchronously from a web app, ~50 req/s average**
- **Classification: PaaS (container service)** — Cloud Run, ECS Fargate, or Kubernetes
- Why: Predictable sustained load makes serverless cost-inefficient (cold starts + concurrency charges). Containers scale to 50 req/s easily. PaaS handles autoscaling without provisioning VMs.

**Scenario E: Experimental LLM chatbot prototype — a data scientist's personal demo, used once a week**
- **Classification: Serverless** (Lambda + API Gateway, or Cloud Run min-instances=0)
- Why: Zero traffic most of the time. Cost must be near-zero. Cold start on the first request each week is acceptable. Serverless's scale-to-zero is the right tradeoff.

---

## Exercise 2: IaaS vs PaaS Tradeoffs for a RAG Pipeline

### IaaS (e.g., EC2 + self-managed Kubernetes or Docker)

| Concern | IaaS |
|---------|------|
| **Cost** | Lower per-hour unit cost; pay for reserved/spot capacity. But: idle VMs cost money, ops labor is real cost. |
| **Control** | Full: OS, kernel, GPU drivers, network config, storage layout. |
| **Ops burden** | High: patching, scaling automation, load balancer config, health checks, node replacement. |
| **Latency** | Lower cold start; no container orchestrator overhead for simple cases. |
| **Scaling** | Manual autoscaling group config or custom HPA. Slower to react. |
| **Compliance** | VPC-isolated; can meet strict data-residency requirements. |

### PaaS (e.g., Cloud Run, ECS Fargate, Azure Container Apps)

| Concern | PaaS |
|---------|------|
| **Cost** | Higher per-unit but zero idle cost (scale-to-zero). No ops labor. |
| **Control** | Limited: container entrypoint, env vars, CPU/memory per task. No OS access. |
| **Ops burden** | Low: platform handles patching, health checks, autoscaling, log routing. |
| **Latency** | Cold start penalty when scaling from zero (1–3 s for Python containers). |
| **Scaling** | Automatic, event-driven. Near-instant horizontal scale. |
| **Compliance** | VPC integration available; some PaaS services have data-residency constraints. |

**Recommendation for RAG**: Use PaaS for the API layer (stateless retrieval + generation). Use IaaS/GPU instances only for the embedding model if self-hosted. Managed vector DB (Pinecone, Weaviate Cloud) eliminates storage IaaS entirely.

---

## Exercise 3: Deployment Topology for High-Throughput Inference

**Requirement assumptions**: 1 000 req/s sustained, p50 < 200 ms, p99 < 500 ms, stateless transformer inference.

```
                         ┌─────────────────────────────────┐
Internet ──► Route 53 ──►│  ALB (multi-AZ)                 │
                         │  SSL termination, WAF attached   │
                         └──────────┬──────────────────────┘
                                    │  HTTP/2
                         ┌──────────▼──────────────────────┐
                         │  ECS Fargate Service             │
                         │  (API + vLLM / triton sidecar)  │
                         │  Auto-scaling: target 60% CPU   │
                         │  Min 4 tasks, Max 40 tasks       │
                         └──────────┬──────────────────────┘
              ┌───────────────────┬─┘──────────────────────┐
              │                   │                         │
   ┌──────────▼──────┐  ┌────────▼──────┐     ┌───────────▼──────┐
   │  ElastiCache     │  │  RDS (read    │     │  S3 / model      │
   │  Redis           │  │  replica) for │     │  weights bucket  │
   │  response cache  │  │  chat history │     │  (loaded at init)│
   └─────────────────┘  └───────────────┘     └──────────────────┘
```

**Key decisions**:
- **ALB not NLB**: HTTP/2 multiplexing, sticky sessions optional, WAF integration — all needed for AI APIs.
- **ECS Fargate not Lambda**: 15-minute Lambda timeout is too short for streaming LLM responses. Fargate tasks run indefinitely.
- **GPU tasks on ECS EC2 launch type** (not Fargate) if self-hosting model weights — Fargate does not support GPU.
- **Semantic cache (Redis)**: cache embedding → response mappings. Hit rate 20–40% on typical chat traffic → 20–40% cost reduction.
- **Multi-AZ**: deploy tasks across ≥ 2 AZs. ALB routes to healthy targets automatically.
- **Model weights in S3**: tasks pull weights at container init. EFS alternative for shared NFS but S3 is simpler.

---

## Exercise 4: When Serverless Is NOT Appropriate for AI

Serverless (Lambda, Cloud Functions) is a poor fit when **any** of the following are true:

**1. GPU required**
Lambda and equivalent serverless platforms do not offer GPU instances. Inference on GPT-class models needs VRAM. Serverless cannot provide it.

**2. Long-running inference**
Lambda max timeout is 15 minutes. Streaming LLM responses for complex queries can exceed this. Batch re-ranking or embedding large corpora will timeout. Use ECS Fargate tasks or Step Functions + SQS instead.

**3. Large model artifacts**
Lambda deployment package + layers cap at 250 MB (unzipped). A small embedding model (e.g., `all-MiniLM`) is ~90 MB fine, but any medium-size model (> 500 MB) cannot be bundled. Workaround (EFS, S3 download at cold start) adds latency and complexity that erodes the serverless simplicity argument.

**4. Sustained high-concurrency with tight p99 SLA**
Cold starts on Lambda Python runtimes take 1–3 seconds. Provisioned concurrency mitigates this but charges per-hour — at that point cost approaches ECS Fargate and you lose burst elasticity benefits. If p99 < 200 ms is required, provisioned concurrency is mandatory and expensive.

**5. Stateful streaming responses (WebSocket / SSE)**
Lambda HTTP response must complete within the function timeout. AWS Lambda Response Streaming exists but has limitations. Server-sent events and WebSocket long-connections are better served by a persistent container.

**6. Heavy warm-up cost**
Loading tokenizers, embedding models, or FAISS indexes at cold start adds 5–30 seconds. This is visible to users. Containers with `/tmp` caching between warm invocations partially help, but Lambda's ephemeral nature means cache misses are frequent under burst traffic.

**Rule of thumb**: Use serverless for AI *glue code* (webhook handlers, lightweight API adapters, async triggers). Use containers for *model inference* regardless of model size.
