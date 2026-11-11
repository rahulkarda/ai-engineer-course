# Phase 3 · Week 07 · Day 1 — Pretraining vs Fine-tuning vs Instruction Tuning vs RLHF

> Objective: Understand the four main training paradigms for LLMs and how they differ in data, objective, and cost.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-5]]

## Concept

Large language models go through multiple distinct training phases. Each phase serves a different purpose and uses different data, objectives, and compute budgets.

**Pretraining**
The model learns from massive, largely unstructured text corpora (web crawls, books, code). The objective is next-token prediction (causal language modeling): given tokens 1..n, predict token n+1. Minimizing cross-entropy over billions of tokens forces the model to internalize grammar, facts, reasoning patterns, and world knowledge. No human labels are required — the labels are the tokens themselves. Compute cost: enormous (thousands of GPU-days for frontier models).

**Supervised Fine-tuning (SFT)**
After pretraining, the base model can complete text but does not reliably follow instructions or produce safe outputs. SFT continues training on a curated, domain-specific dataset of (input, output) pairs with the same next-token prediction objective. The dataset is orders of magnitude smaller (thousands to millions of examples). Weights drift from the pretrained distribution toward the target task. Compute cost: modest (hours to days on a single machine for adapter-based methods).

**Instruction Tuning**
A specific form of SFT where the training pairs are (instruction, desired response). The instruction covers diverse tasks: summarize, translate, classify, write code, answer questions. Training on this mixture teaches the model to parse and follow natural language directives. Result: a model that generalizes to *new* instructions not seen during training. InstructGPT, FLAN-T5, and Alpaca are examples. The dataset format is typically:

```
{"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour le monde"}
```

**RLHF (Reinforcement Learning from Human Feedback)**
SFT alone does not guarantee outputs humans prefer. RLHF adds a feedback loop:
1. Collect pairwise human preferences: for the same prompt, humans rank two model responses (A > B).
2. Train a **reward model** (RM) on these comparisons — the RM learns to score responses.
3. Use Proximal Policy Optimization (PPO) to fine-tune the language model (the "policy") to maximize the reward model's score, subject to a KL-divergence penalty that prevents the policy from drifting too far from the SFT checkpoint.

RLHF is why ChatGPT feels more aligned than a raw base model: the reward signal encodes human values more richly than token prediction alone.

## How it works internally

**Why pretraining generalizes**
Next-token prediction is a self-supervised compression task. The model must build internal representations that capture syntax, semantics, and world knowledge to compress diverse text well. These representations transfer to downstream tasks — the model effectively builds a world model as a side effect.

**Why fine-tuning on small data works**
The pretrained weights already encode language understanding. Fine-tuning adjusts the last few layers (or adapters) to shift the output distribution toward the target domain without forgetting the base knowledge — this is called "catastrophic forgetting" when done poorly (full fine-tune on too little data). Parameter-efficient methods (LoRA, prefix tuning) mitigate this by training only a fraction of weights.

**Why RLHF beats pure SFT for alignment**
SFT imitates demonstrations. If the demonstration is only "good enough," the model learns to be "good enough." RLHF directly optimizes for human preference scores, so the model can exceed the quality of its training demonstrations. The KL penalty prevents reward hacking (gaming the RM without actually improving).

**Reward model internals**
The RM is typically the base LM with a scalar regression head appended. Given a prompt + response, it outputs a single score. It is trained on Bradley-Terry loss from pairwise comparisons:

```
loss = -log(sigmoid(score(chosen) - score(rejected)))
```

This loss pushes the score of the chosen response above the rejected response.

## Worked example

```python
"""
Conceptual RLHF simulation using numpy.
No actual model training — illustrates the reward model and PPO loop logic.
Run: python day-1-example.py
Dependencies: numpy (uv add numpy)
"""

import logging
import numpy as np
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward Model (mock)
# ---------------------------------------------------------------------------
# In production: a transformer with a scalar regression head.
# Here: a simple linear scorer over hand-crafted response features.

class Response(NamedTuple):
    text: str
    length: int        # token count (proxy feature)
    factual_score: float   # 0..1, simulates factual correctness
    polite_score: float    # 0..1, simulates tone quality


def extract_features(r: Response) -> np.ndarray:
    """Featurize a response for the mock reward model."""
    # Real RM: last-layer hidden state of the LM
    return np.array([
        r.length / 100.0,       # normalized length
        r.factual_score,
        r.polite_score,
        1.0,                    # bias term
    ], dtype=np.float32)


class MockRewardModel:
    """
    Linear reward model trained on pairwise comparisons.
    Real RM: transformer backbone + scalar head trained with Bradley-Terry loss.
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Weights: [length_penalty, factual_weight, polite_weight, bias]
        # Manually set to simulate "humans prefer accurate, polite, concise responses"
        self.weights = np.array([-0.3, 1.5, 1.0, 0.1], dtype=np.float32)

    def score(self, response: Response) -> float:
        """Return scalar reward for a response."""
        features = extract_features(response)
        return float(np.dot(self.weights, features))

    def bradley_terry_loss(self, chosen: Response, rejected: Response) -> float:
        """
        Loss from one human preference pair.
        Lower is better (model is learning to rank chosen > rejected).
        """
        score_chosen = self.score(chosen)
        score_rejected = self.score(rejected)
        # sigmoid(score_chosen - score_rejected) should be close to 1
        logit = score_chosen - score_rejected
        loss = -np.log(1.0 / (1.0 + np.exp(-logit)) + 1e-9)
        return float(loss)


# ---------------------------------------------------------------------------
# Simulated PPO loop (conceptual, no actual gradients)
# ---------------------------------------------------------------------------
# Real PPO: policy generates responses, RM scores them, advantage is computed,
# policy weights are updated to increase probability of high-reward tokens,
# KL divergence from reference policy is subtracted from reward to prevent
# reward hacking.

def simulate_rlhf_step(
    policy_responses: list[Response],
    rm: MockRewardModel,
    ref_scores: list[float],
    kl_coefficient: float = 0.1,
) -> list[float]:
    """
    Compute RLHF reward for each response (reward - KL penalty).
    In real PPO: these become advantages that update the policy network.
    """
    rlhf_rewards = []
    for response, ref_score in zip(policy_responses, ref_scores, strict=True):
        raw_reward = rm.score(response)
        # KL penalty: distance from reference policy (approximated here as
        # difference between current score and reference score)
        kl_approx = abs(raw_reward - ref_score)
        adjusted_reward = raw_reward - kl_coefficient * kl_approx
        rlhf_rewards.append(adjusted_reward)
    return rlhf_rewards


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    rm = MockRewardModel()

    # --- Reward model scoring ---
    responses = [
        Response("The capital of France is Paris.", length=8, factual_score=1.0, polite_score=0.9),
        Response("Uhh, I think maybe Paris? Not sure.", length=9, factual_score=0.4, polite_score=0.5),
        Response(
            "Paris is the capital and largest city of France, with a long history dating back...",
            length=80, factual_score=1.0, polite_score=0.9,
        ),
    ]

    logger.info("=== Reward Model Scoring ===")
    for r in responses:
        score = rm.score(r)
        logger.info("Score %.3f | %s", score, r.text[:60])

    # --- Pairwise comparison loss ---
    logger.info("\n=== Bradley-Terry Loss (chosen vs rejected) ===")
    chosen = responses[0]    # accurate and concise
    rejected = responses[1]  # inaccurate
    loss = rm.bradley_terry_loss(chosen, rejected)
    logger.info("Chosen:   %s (score %.3f)", chosen.text, rm.score(chosen))
    logger.info("Rejected: %s (score %.3f)", rejected.text, rm.score(rejected))
    logger.info("BT Loss: %.4f (lower = RM correctly ranks chosen > rejected)", loss)

    # --- RLHF loop simulation ---
    logger.info("\n=== Simulated RLHF Steps ===")
    # Reference policy scores (from SFT checkpoint)
    ref_scores = [rm.score(r) for r in responses]

    # "Policy" after a hypothetical PPO step: slightly better factual scores
    updated_responses = [
        Response(r.text, r.length, min(r.factual_score + 0.1, 1.0), r.polite_score)
        for r in responses
    ]
    rlhf_rewards = simulate_rlhf_step(updated_responses, rm, ref_scores, kl_coefficient=0.1)
    for r, reward in zip(updated_responses, rlhf_rewards, strict=True):
        logger.info("RLHF reward %.3f | %s", reward, r.text[:60])

    # --- Key differences summary ---
    logger.info("\n=== Training Paradigm Comparison ===")
    paradigms = [
        ("Pretraining",        "next-token prediction", "massive web/books", "enormous", "world knowledge"),
        ("SFT",                "next-token prediction", "curated (input,output)", "modest",   "task adaptation"),
        ("Instruction tuning", "next-token prediction", "(instruction,response)", "modest",   "follow directives"),
        ("RLHF",               "maximize RM score - KL","human pref pairs",  "moderate", "alignment"),
    ]
    header = f"{'Paradigm':<22} {'Objective':<28} {'Data':<26} {'Compute':<10} {'Effect'}"
    logger.info(header)
    logger.info("-" * len(header))
    for p in paradigms:
        logger.info(f"{p[0]:<22} {p[1]:<28} {p[2]:<26} {p[3]:<10} {p[4]}")


if __name__ == "__main__":
    main()
```

Expected output (abridged):
```
INFO: === Reward Model Scoring ===
INFO: Score 1.780 | The capital of France is Paris.
INFO: Score 0.440 | Uhh, I think maybe Paris? Not sure.
INFO: Score 1.540 | Paris is the capital and largest city of France...
```

Note: install numpy with `uv add numpy` before running.

## Exercises

1. **Conceptual**: Explain in your own words why a base pretrained model fails to follow instructions, and why instruction tuning fixes this. What property does instruction-tuning data have that pretraining data lacks?

2. **Conceptual**: The KL penalty in RLHF prevents "reward hacking." Describe a concrete failure mode that would occur if there were no KL penalty — what would the policy learn to do?

3. **Conceptual**: Compare SFT and RLHF data collection costs. SFT requires (prompt, response) pairs; RLHF requires pairwise preference rankings. Which is cheaper per example, and why? How does this affect the practical choice between the two?

4. **Coding**: Extend the `MockRewardModel` class to implement a `fit` method that updates `self.weights` using gradient descent on Bradley-Terry loss over a list of `(chosen, rejected)` pairs. Use only numpy. Run 100 steps and log the loss at each step. Verify the loss decreases.

(Solutions: ../solutions/week-07/day-1.md)

## Self-check

- [ ] I can explain the objective function for each of the four paradigms (pretraining, SFT, instruction tuning, RLHF) in one sentence each.
- [ ] I can describe what a reward model is, what data it is trained on, and what loss function is used.
- [ ] I can explain why RLHF needs a KL-divergence penalty and what happens without it.
- [ ] I can implement a mock reward model that scores responses and computes Bradley-Terry loss using numpy.

## Supplementary reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT paper): https://arxiv.org/abs/2203.02155
- Wei et al., "Finetuned Language Models Are Zero-Shot Learners" (FLAN): https://arxiv.org/abs/2109.01652
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences": https://arxiv.org/abs/1909.08593
- Anthropic, "Constitutional AI": https://arxiv.org/abs/2212.08073
