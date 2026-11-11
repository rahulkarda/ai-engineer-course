# Phase 6 · Week 08 · Day 1 — Cloud Deployment Models

> Objective: Understand IaaS, PaaS, serverless, and containers as architectural choices — and pick the right model for AI workloads.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-5]]

---

## Concept

Every cloud deployment sits somewhere on an **abstraction spectrum**. The further right you go, the less infrastructure you manage — but the less control you have.

```
Raw metal → IaaS → CaaS → PaaS → FaaS (serverless)
   (full control)                    (full abstraction)
```

**IaaS (Infrastructure as a Service)**
Rent virtual machines, block storage, and networking primitives. You manage OS, runtime, dependencies, and scaling. Examples: EC2, GCE Compute Engine, Azure VMs.

**PaaS (Platform as a Service)**
Deploy code or artifacts; the platform manages OS and runtime. You own the application logic. Examples: Elastic Beanstalk, Heroku, Azure App Service.

**CaaS (Containers as a Service)**
Run container images; platform manages scheduling, networking, and health. You own the image build. Examples: ECS (Fargate), Cloud Run, Azure Container Apps.

**FaaS / Serverless**
Deploy function handlers; platform manages everything including compute provisioning. You pay per invocation. Examples: AWS Lambda, Cloud Functions, Azure Functions.

---

## How it works internally

Each model abstracts a different layer of the **control plane / data plane** stack:

| Model | You manage | Platform manages |
|---|---|---|
| IaaS | OS, runtime, app, scaling policy | Hypervisor, physical hardware |
| PaaS | App code, env vars, dependencies | OS, runtime, patching, some scaling |
| CaaS | Container image, task definition | Scheduling, networking overlay, health restarts |
| FaaS | Function handler, IAM | Container lifecycle, cold start, concurrency scaling |

**Why this matters for AI:**
- Model inference is CPU/GPU-intensive and often stateful (loaded weights) — FaaS cold starts hurt.
- LLM API calls are I/O-bound and short — FaaS is often fine for that layer.
- Vector DBs and embedding models need persistent storage — serverless is a poor fit unless managed externally.
- GPU workloads require IaaS or specialized CaaS (e.g., ECS with GPU-enabled instance types, Cloud Run with NVIDIA L4).

**Scaling models differ:**
- IaaS: you configure auto-scaling groups; scale-out is slow (VM boot ~60–90 s).
- CaaS: task/pod scale-out is fast (~5–15 s with warm images).
- FaaS: per-request scaling, near-instant, but concurrency limits and cold starts apply.

---

## Worked example

The following script classifies a set of AI deployment scenarios against the four models. No cloud credentials needed — pure logic.

```python
# classify_deployment.py
# Usage: python classify_deployment.py
# Classifies AI deployment scenarios using a rule-based decision framework.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DeploymentModel(str, Enum):
    IAAS = "IaaS"
    PAAS = "PaaS"
    CAAS = "CaaS (managed containers)"
    FAAS = "FaaS (serverless)"


@dataclass
class AIScenario:
    name: str
    needs_gpu: bool = False
    stateful_model_loaded: bool = False  # weights kept in memory
    avg_request_duration_s: float = 1.0
    requests_per_day: int = 1_000
    team_ops_maturity: str = "medium"  # low | medium | high
    latency_sensitive: bool = False
    notes: str = ""


def classify(scenario: AIScenario) -> tuple[DeploymentModel, str]:
    """
    Rule-based decision framework.
    Returns (recommended_model, rationale).
    """
    # GPU workloads need IaaS or specialized CaaS
    if scenario.needs_gpu and scenario.stateful_model_loaded:
        return (
            DeploymentModel.IAAS,
            "GPU + resident model weights → IaaS (EC2 p3/p4, or CaaS with GPU node pool).",
        )

    # Short I/O-bound calls, low volume → FaaS
    if (
        scenario.avg_request_duration_s < 5
        and not scenario.stateful_model_loaded
        and scenario.requests_per_day < 100_000
    ):
        return (
            DeploymentModel.FAAS,
            "Short I/O-bound calls, no resident state, moderate volume → FaaS (Lambda/Cloud Functions).",
        )

    # Containerised app, team knows Docker, no GPU
    if scenario.team_ops_maturity in ("medium", "high") and not scenario.needs_gpu:
        return (
            DeploymentModel.CAAS,
            "Container-native team, no GPU → CaaS (ECS Fargate / Cloud Run).",
        )

    # Low-ops-maturity teams, no special requirements
    if scenario.team_ops_maturity == "low":
        return (
            DeploymentModel.PAAS,
            "Low ops maturity → PaaS (Elastic Beanstalk / App Service) to minimise toil.",
        )

    # Fallback
    return (
        DeploymentModel.CAAS,
        "Default: CaaS provides good balance of control and managed operations.",
    )


SCENARIOS: list[AIScenario] = [
    AIScenario(
        name="Real-time LLM API proxy",
        needs_gpu=False,
        stateful_model_loaded=False,
        avg_request_duration_s=2.0,
        requests_per_day=50_000,
        team_ops_maturity="medium",
        latency_sensitive=True,
        notes="Wraps a managed LLM API (e.g. Claude, Bedrock). No model weights on host.",
    ),
    AIScenario(
        name="Batch embedding pipeline",
        needs_gpu=True,
        stateful_model_loaded=True,
        avg_request_duration_s=120.0,
        requests_per_day=500,
        team_ops_maturity="high",
        notes="Runs sentence-transformers on-prem embeddings nightly.",
    ),
    AIScenario(
        name="RAG web app",
        needs_gpu=False,
        stateful_model_loaded=False,
        avg_request_duration_s=4.0,
        requests_per_day=20_000,
        team_ops_maturity="medium",
        notes="FastAPI + vector DB query + LLM call per request.",
    ),
    AIScenario(
        name="Startup MVP chatbot",
        needs_gpu=False,
        stateful_model_loaded=False,
        avg_request_duration_s=3.0,
        requests_per_day=500,
        team_ops_maturity="low",
        notes="Two-person team, no DevOps expertise.",
    ),
    AIScenario(
        name="Fine-tuned open-source model serving",
        needs_gpu=True,
        stateful_model_loaded=True,
        avg_request_duration_s=1.5,
        requests_per_day=200_000,
        team_ops_maturity="high",
        latency_sensitive=True,
        notes="Llama-variant, resident in GPU memory, high throughput.",
    ),
]


def main() -> None:
    logger.info("=== AI Deployment Model Classifier ===\n")
    for scenario in SCENARIOS:
        model, rationale = classify(scenario)
        logger.info("Scenario : %s", scenario.name)
        logger.info("Notes    : %s", scenario.notes)
        logger.info("Decision : %s", model.value)
        logger.info("Rationale: %s\n", rationale)


if __name__ == "__main__":
    main()
```

**Expected output (abbreviated):**
```
INFO Scenario : Real-time LLM API proxy
INFO Decision : FaaS (serverless)

INFO Scenario : Batch embedding pipeline
INFO Decision : IaaS

INFO Scenario : RAG web app
INFO Decision : CaaS (managed containers)
```

---

## Decision Framework

```
START
  |
  v
Needs GPU + resident model weights?
  YES → IaaS (GPU instance) or CaaS with GPU node pool
  NO  →
        |
        v
      Request < 5 s AND stateless AND < 100k req/day?
        YES → FaaS (Lambda / Cloud Functions)
        NO  →
              |
              v
            Team knows containers?
              YES → CaaS (ECS Fargate / Cloud Run / Container Apps)
              NO  →
                    |
                    v
                  Low ops maturity?
                    YES → PaaS (Elastic Beanstalk / App Service)
                    NO  → CaaS (default safe choice)
```

---

## Exercises

1. **Topology design**: You are building a production RAG pipeline with three components: (a) an embedding service that runs a local model, (b) a FastAPI query endpoint that calls a managed LLM API, (c) a nightly batch job that re-indexes a document store. For each component, choose a deployment model and justify the choice. Draw a simple network diagram showing how they connect.

2. **Trade-off analysis**: Your team is debating between CaaS and FaaS for a real-time AI inference endpoint. List three concrete scenarios where CaaS wins and three where FaaS wins. What is the single most important variable that drives the decision?

3. **Migration planning**: A team runs a Python LLM app on a single EC2 instance (IaaS). Traffic has grown 10x. Describe the migration path to CaaS (ECS Fargate). What must change in the app, the build process, and the deployment pipeline?

(Solutions: ../solutions/week-08/day-1.md)

---

## Self-check

- [ ] I can explain what each of the four cloud deployment models abstracts away, without notes.
- [ ] I can state two reasons why GPU/stateful AI workloads are a poor fit for FaaS.
- [ ] I can apply the decision framework to a novel AI scenario and justify my choice.
- [ ] I can implement the classifier script from memory, including proper logging and type hints.

---

## Supplementary links

- [AWS — Choosing the right compute service](https://docs.aws.amazon.com/decision-guides/latest/compute-service-overview/compute-service-overview.html)
- [GCP — Choosing a deployment option](https://cloud.google.com/blog/topics/developers-practitioners/where-should-i-run-my-stuff-choosing-google-cloud-compute-option)
- [Azure — Compute decision tree](https://learn.microsoft.com/en-us/azure/architecture/guide/technology-choices/compute-decision-tree)
- [AWS Well-Architected — Serverless lens](https://docs.aws.amazon.com/wellarchitected/latest/serverless-applications-lens/welcome.html)
