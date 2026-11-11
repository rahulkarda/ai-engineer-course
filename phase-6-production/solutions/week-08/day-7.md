# Day 7 Solution — Project Review, ADRs & Production Readiness

## Exercise 1: Architecture Decision Record (ADR)

ADRs document decisions so future engineers understand *why*, not just *what*. Each ADR captures the context, options considered, the chosen option, and consequences (positive and negative).

---

### ADR-001: Containerized ECS Fargate over Lambda or EC2

**Status:** Accepted  
**Date:** 2025-01-12  
**Author:** Platform Team  
**Deciders:** Backend Lead, ML Lead, VP Engineering

---

#### Context

We are deploying a FastAPI AI inference API that:
- Calls Claude claude-sonnet-5-5 via the Anthropic API
- Requires consistent low-latency responses (p95 < 2s)
- Handles 5–100 RPS with unpredictable spikes
- Has application startup time of ~15–60 seconds (model warm-up, dependency loading)
- Needs access to PostgreSQL and Redis in the same VPC
- Must support graceful rolling deploys with zero downtime

We evaluated three deployment targets: AWS Lambda, ECS Fargate, and EC2 Auto Scaling Groups.

---

#### Decision Drivers

1. Cold start penalty must not affect user-facing p95 latency
2. Deployment cycle must be under 5 minutes
3. Operational overhead: team has no dedicated DevOps — developers own the stack
4. Cost must scale proportionally with traffic; no idle-capacity charges at zero RPS
5. Must integrate cleanly with ALB for HTTP/2 and WebSocket support
6. Container image size is ~500 MB (heavy ML dependencies)

---

#### Options Considered

**Option A: AWS Lambda (container image)**
- Pro: true pay-per-request pricing; zero idle cost
- Pro: automatic scaling from 0 to thousands
- Con: cold start with 500 MB container is 8–30 seconds — unacceptable for p95 target
- Con: max timeout 15 minutes; streaming responses require Lambda Function URLs or API Gateway WebSocket (complex)
- Con: 512 MB–10 GB memory only; no fine-grained CPU control
- Con: stateful connections (PostgreSQL, Redis) reset on every invocation without connection pooling workarounds (RDS Proxy adds $50+/month)
- **Verdict: Eliminated — cold start is a hard blocker**

**Option B: EC2 Auto Scaling Groups (ASG) with ALB**
- Pro: full control over instance type and OS; cheapest per-unit compute at scale
- Pro: persistent connection to RDS/Redis
- Con: ASG scale-out is 3–5 minutes (instance provision + bootstrap) vs 30–60 seconds for Fargate
- Con: requires AMI management, OS patching, SSH key management
- Con: minimum cost even at zero traffic (at least 2 instances in ASG for HA)
- Con: significant operational overhead for a small team
- **Verdict: Rejected — operational burden too high for team size**

**Option C: ECS Fargate**
- Pro: containerized — same artifact in dev, CI, and production
- Pro: Fargate tasks start in 30–60 seconds; pre-warmed min=2 tasks absorb spikes with zero cold start
- Pro: no EC2 management; AWS owns the underlying host patching
- Pro: native ALB integration with target groups; supports HTTP/2 and WebSocket
- Pro: persistent TCP connections to RDS and Redis (connection pooling works normally)
- Pro: rolling deploys with minimum_healthy_percent=100 ensures zero downtime
- Con: more expensive than Lambda at very low RPS (< ~10K requests/day)
- Con: slightly more Terraform to configure than Lambda
- **Verdict: Accepted**

---

#### Decision

Deploy the AI API as an ECS Fargate service with:
- min_capacity = 2 (HA across two AZs, no cold starts)
- max_capacity = 20
- Target tracking at 60% CPU
- ALB with HTTPS termination
- Private subnets; ALB in public subnets

---

#### Consequences

**Positive:**
- Predictable latency — requests always land on a warm container
- Developer-friendly — `docker run` locally, same image in prod
- Rolling deploys are safe and automatic
- Container Insights gives CPU/memory metrics out of the box

**Negative:**
- Minimum cost of ~$60/month even at zero traffic (2 idle tasks)
- Fargate does not support GPU instances — if we need on-prem inference, we need a separate EC2 or SageMaker path
- Cannot scale to zero for truly cost-sensitive dev environments (use a scheduled scale-to-zero for off-hours)

**Revisit triggers:**
- If traffic consistently < 1K requests/day → reconsider Lambda with provisioned concurrency
- If we add local model inference (e.g. quantized LLaMA) → evaluate ECS on GPU EC2 via capacity providers

---

### ADR-002: AWS Secrets Manager over Parameter Store or Vault

**Status:** Accepted  
**Date:** 2025-01-12

---

#### Context

The application requires three secrets at runtime:
1. `ANTHROPIC_API_KEY` — external API credential
2. `DATABASE_URL` — includes password
3. `REDIS_URL` — includes auth token

These must not appear in: source code, Docker images, ECS task definition plaintext, Terraform state, or CloudWatch logs.

---

#### Options Considered

| | SSM Parameter Store (Standard) | SSM Parameter Store (Advanced) | Secrets Manager |
|---|---|---|---|
| Cost | Free | $0.05/param/month | $0.40/secret/month |
| Auto-rotation | No | No | Yes (native for RDS, custom Lambda) |
| ECS native injection | Yes | Yes | Yes |
| Versioning | Yes | Yes | Yes (with staging labels) |
| Cross-account replication | No | No | Yes |
| API call cost | Free | Free | $0.05/10K calls |

---

#### Decision

Use Secrets Manager for all production secrets.

**Rationale:** Auto-rotation support is the deciding factor. When the Anthropic API key rotates or a database password is cycled, Secrets Manager triggers a Lambda rotation function that updates the secret without any code change or deployment. With Parameter Store, rotation requires manual intervention or custom automation. At $0.40/secret/month and 3 secrets, the cost is $1.20/month — negligible.

**Caching:** Application fetches secrets once at startup and caches with a 5-minute TTL to stay within the free tier of API calls (10K/month included).

---

## Exercise 2: Production Readiness Checklist

Use this checklist before declaring a deployed AI API production-ready. Items marked **[BLOCKER]** must pass before going live.

### Security

- [ ] **[BLOCKER]** No secrets in environment variables, source code, Docker image, or Terraform state
- [ ] **[BLOCKER]** All secrets stored in Secrets Manager; injected via ECS `secrets` field
- [ ] **[BLOCKER]** `.env` file is in `.gitignore`; no `.env` committed in git history
- [ ] **[BLOCKER]** ECS tasks run as non-root user (uid != 0)
- [ ] **[BLOCKER]** ECS security group accepts traffic only from ALB security group (not 0.0.0.0/0)
- [ ] **[BLOCKER]** ALB listener uses HTTPS (TLS 1.2+); HTTP redirects to HTTPS
- [ ] **[BLOCKER]** ECR image scanning enabled; no CRITICAL vulnerabilities in latest image
- [ ] IAM roles follow least privilege — no `*` actions or resources in policies
- [ ] VPC endpoints configured for S3 and Secrets Manager
- [ ] CloudTrail enabled in the account
- [ ] CORS configured to specific origins (not `*` in production)
- [ ] API key or JWT auth on all endpoints except `/health`

### Reliability

- [ ] **[BLOCKER]** `min_capacity >= 2` (HA across AZs)
- [ ] **[BLOCKER]** ALB health check passing; `/health` endpoint responds 200
- [ ] **[BLOCKER]** ECS deployment circuit breaker enabled with rollback
- [ ] **[BLOCKER]** `minimum_healthy_percent = 100` in deployment config
- [ ] `maximum_percent = 200` to allow rolling updates
- [ ] Autoscaling policy configured (CPU target tracking at 60%)
- [ ] Task `stopTimeout = 30` and application handles SIGTERM gracefully
- [ ] `deregistration_delay` on target group matches `stopTimeout`
- [ ] RDS has Multi-AZ enabled (if using RDS)
- [ ] Redis has replication enabled (if using ElastiCache)
- [ ] Retry logic on all external API calls (Anthropic, downstream services)
- [ ] Circuit breaker pattern implemented for external dependencies

### Observability

- [ ] **[BLOCKER]** CloudWatch Logs configured; all containers shipping logs
- [ ] **[BLOCKER]** Structured JSON logging (no unstructured `print()` statements)
- [ ] **[BLOCKER]** No secrets or PII in log output
- [ ] Container Insights enabled on ECS cluster
- [ ] CloudWatch alarms on: CPU > 80%, memory > 85%, ALB 5xx error rate > 1%, unhealthy host count > 0
- [ ] p50/p95/p99 latency metrics tracked (ALB Target Response Time + custom metrics)
- [ ] SNS topic connected to alarms; on-call pager or Slack notification configured
- [ ] Distributed tracing enabled (X-Ray or OpenTelemetry)
- [ ] Dashboard with: RPS, error rate, latency percentiles, task count, CPU/memory

### Cost

- [ ] S3 VPC Gateway endpoint created (eliminates NAT charges for S3)
- [ ] Secrets Manager VPC Interface endpoint created (eliminates NAT for secret fetches)
- [ ] ECR lifecycle policy set (keep last 10 images)
- [ ] CloudWatch Logs retention period set (not infinite)
- [ ] AWS Cost Budget alert configured for the account
- [ ] NAT Gateway count = 1 for non-production, 3 for production (cost vs. HA tradeoff documented)

### Deployment

- [ ] **[BLOCKER]** CI/CD pipeline builds image, pushes to ECR, triggers ECS rolling deploy
- [ ] **[BLOCKER]** Docker image uses multi-stage build; final image < 600 MB
- [ ] Image tags are immutable (not `latest`) in production
- [ ] Image tag tied to git commit SHA for full traceability
- [ ] Rollback procedure documented and tested
- [ ] `terraform plan` runs in CI before `apply`; state stored in S3 with locking

### Operational

- [ ] **[BLOCKER]** Runbook written: how to deploy, rollback, scale, rotate secrets
- [ ] ADRs written for major architectural decisions
- [ ] On-call rotation established
- [ ] Data retention policy defined; old S3 objects lifecycle-managed
- [ ] Disaster recovery plan documented (RTO/RPO targets)
- [ ] Load test run against staging at 2× expected peak RPS

### Application

- [ ] **[BLOCKER]** `/health` endpoint returns 200 and checks critical dependencies (DB connection, cache)
- [ ] `/ready` or equivalent readiness probe distinguished from liveness (optional but recommended)
- [ ] Graceful shutdown: in-flight requests drain before process exits
- [ ] Request timeout set and communicated to clients (e.g. 30s max)
- [ ] Rate limiting implemented at ALB or application level
- [ ] Input validation on all request bodies (Pydantic models)
- [ ] Error responses use consistent JSON schema (never expose stack traces)
- [ ] API versioning scheme in place (e.g. `/v1/...`)

---

**Scoring guidance:**  
All `[BLOCKER]` items must be checked before launch. Non-blocker items should be addressed within the first sprint post-launch. Any unchecked item should have a tracked ticket.
