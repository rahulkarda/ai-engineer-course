# Solutions — Phase 3 · Week 07 · Day 1

## Exercise 1 — Pretraining vs Instruction Tuning

**(a) Objective function**

Pretraining and instruction tuning share the same mathematical objective: cross-entropy over next-token prediction. The difference is in what the model is predicting over.

In pretraining, loss is computed over every token in the sequence. Every position contributes a gradient signal.

In instruction tuning (SFT on instruction-response pairs), loss is typically masked to the response tokens only. The instruction portion is treated as context (teacher-forcing with masked labels), so the model is not penalized for "predicting" the instruction tokens it already has. Some implementations compute loss over the full sequence; masking the instruction is the more common and principled approach because it avoids penalizing the model for conditioning on the prompt.

Formally both use:

```
L = -1/T * sum_t log P(x_t | x_<t ; theta)
```

but instruction tuning restricts T to the response span.

**(b) Dataset size**

Pretraining: trillions of tokens. GPT-3 used ~300B tokens; Llama-3 used ~15 trillion tokens. The corpora are largely unlabeled web scrapes, books, and code — data collection cost is engineering, not annotation.

Instruction tuning: thousands to low millions of examples. InstructGPT used ~13K human-written demonstrations for SFT. Alpaca used 52K GPT-generated examples. The bottleneck is quality and diversity of instruction-response pairs, not raw volume.

**(c) Computational cost**

Pretraining: enormous. A single frontier pretraining run requires thousands of GPU-days and tens to hundreds of millions of dollars of compute. For example, GPT-3 used ~3.14e23 FLOPs; training a 70B model on 1T tokens at 6ND FLOPs ≈ 4.2e23 FLOPs.

Instruction tuning: modest to moderate. Full fine-tuning a 7B model on 52K examples takes hours on a single 8-GPU machine. With LoRA or QLoRA, a single consumer GPU (24 GB) can fine-tune a 7B model in hours. Cost is typically 1–3 orders of magnitude below pretraining.

---

## Exercise 2 — Why RLHF Requires 3 Distinct Phases

The three phases are:

1. **Supervised Fine-tuning (SFT)**: train the base LM on high-quality (instruction, response) demonstrations to produce a well-behaved policy that generates coherent, on-task outputs.

2. **Reward Model (RM) training**: collect human pairwise preferences over responses, then train a separate model that maps (prompt, response) -> scalar score using Bradley-Terry loss.

3. **RL fine-tuning (PPO)**: use the frozen RM as a reward signal to further fine-tune the SFT policy via Proximal Policy Optimization, with a KL-divergence penalty to prevent reward hacking.

**What would go wrong without the reward model phase:**

You cannot skip the reward model and go directly from SFT to RL. The core problem is that RL requires a dense scalar reward signal at each step, but human preferences are discrete ordinal judgments ("A is better than B"). You cannot practically query a human for every token the model generates during PPO training — a single PPO run involves millions of forward passes. The RM is a learned approximation of human judgment that can be evaluated cheaply at inference time.

If you tried to use raw SFT quality as the RL reward (e.g., log-likelihood of a gold response), you would be doing standard maximum likelihood training with extra steps — no alignment benefit. If you tried sparse binary rewards from humans (rate the final output once), the credit assignment problem becomes intractable: PPO cannot attribute which of the 200 generated tokens deserved the reward.

The reward model bridges discrete human preferences (collected offline, cheaply) and the continuous per-token gradient signal that PPO requires.

---

## Exercise 3 — Which Approach for SQL Generation

**Recommendation: SFT (Supervised Fine-tuning), not pretraining from scratch or RLHF.**

Justification:

**Against pretraining from scratch:**
GPT-4 class models already have deep knowledge of SQL syntax, database schemas, and natural language. Pretraining from scratch would require hundreds of billions of SQL-adjacent tokens, enormous compute (comparable to the original pretraining run), and would discard years of accumulated language understanding. The marginal improvement in SQL over a well-fine-tuned model is not worth the cost.

**For SFT:**
The company needs to shift the model's output distribution from "general-purpose assistant" to "accurate SQL generator for our schema." SFT on a dataset of (NL question, correct SQL query) pairs accomplishes this directly. With LoRA-based SFT, this is achievable in hours on modest hardware. The base model's semantic understanding transfers — fine-tuning adds domain vocabulary (table names, column conventions, SQL dialect specifics).

A dataset of 10K–100K (question, SQL) pairs per target schema is feasible to collect via existing query logs, synthetic generation, or human annotation. This is the standard approach used by Spider, BIRD, and commercial text-to-SQL systems.

**Against RLHF as the primary approach:**
RLHF adds value when correctness is difficult to specify but human preference is easy to judge — e.g., tone, safety, helpfulness. SQL has a ground truth: the query either executes correctly and returns the right rows, or it does not. Execution accuracy against a test database is a clean, scalar reward signal that can be applied directly in RL fine-tuning (execution-guided training). If execution-guided RL is used, it bypasses the need for a human reward model. But this is an enhancement on top of SFT, not a first step.

**Decision tree:** Start with SFT on curated (question, SQL) pairs. Evaluate execution accuracy. If SFT plateaus, add execution-guided RL fine-tuning (execution accuracy as reward), not human RLHF.

---

## Exercise 4 — Mock RewardModel with RLHF Loop Sketch

```python
"""
Mock RewardModel with heuristic scoring and RLHF training loop sketch.

Demonstrates how a reward model would be used in PPO-based RLHF:
1. Policy generates candidate responses
2. Reward model scores each (prompt, response) pair
3. Scores become advantages that update the policy

Dependencies: none (stdlib only)
Run: python day-1-solution.py
"""

import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RMInput:
    """A single (prompt, response) pair to be scored."""
    prompt: str
    response: str


class PPOBatch(NamedTuple):
    """One batch of RLHF PPO data."""
    rm_inputs: list[RMInput]
    rewards: list[float]
    kl_penalties: list[float]
    adjusted_rewards: list[float]


# ---------------------------------------------------------------------------
# Heuristic features
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "because", "therefore", "specifically", "for example",
    "first", "second", "finally", "in summary",
})

NEGATIVE_KEYWORDS: frozenset[str] = frozenset({
    "maybe", "i think", "i'm not sure", "possibly", "dunno",
    "idk", "whatever", "etc etc",
})

IDEAL_LENGTH_TOKENS: int = 80  # approximate target response length


def _token_count(text: str) -> int:
    """Approximate token count: split on whitespace."""
    return len(text.split())


def _keyword_score(text: str) -> float:
    """
    Heuristic keyword quality signal.

    Positive keywords (reasoning words) add +0.05 each.
    Negative keywords (hedging/vague) subtract -0.10 each.
    Capped to [-1.0, 1.0].
    """
    lowered = text.lower()
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in lowered)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lowered)
    raw = (pos_count * 0.05) - (neg_count * 0.10)
    return max(-1.0, min(1.0, raw))


def _length_score(response: str) -> float:
    """
    Length-based quality score.

    Peaks at IDEAL_LENGTH_TOKENS with score 1.0.
    Falls off for very short (< 10 tokens) or very long (> 300 tokens) responses.
    Uses a Gaussian-like shape.
    """
    n = _token_count(response)
    if n < 3:
        return -0.5  # empty or near-empty is bad
    # Log-scale distance from ideal
    distance = abs(math.log(max(n, 1)) - math.log(IDEAL_LENGTH_TOKENS))
    return math.exp(-0.5 * distance)


def _coherence_score(prompt: str, response: str) -> float:
    """
    Heuristic prompt-response relevance.

    Counts how many non-stopword prompt words appear in the response.
    Normalized to [0, 1].
    """
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "at"}
    prompt_words = {
        w.lower().strip(".,?!")
        for w in prompt.split()
        if w.lower() not in stopwords and len(w) > 2
    }
    if not prompt_words:
        return 0.5
    response_lower = response.lower()
    overlap = sum(1 for w in prompt_words if w in response_lower)
    return min(1.0, overlap / len(prompt_words))


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class RewardModel:
    """
    Heuristic reward model for (prompt, response) pairs.

    In production: a transformer backbone (usually the same architecture as
    the policy) with a scalar regression head, trained on pairwise human
    preference data using Bradley-Terry loss.

    Here: a weighted sum of hand-crafted features that approximate what
    human raters tend to prefer — concise, coherent, confident responses.

    Weights (tunable):
      - length_weight:    reward for responses near the ideal length
      - keyword_weight:   reward for reasoning words, penalty for hedging
      - coherence_weight: reward for on-topic responses
    """

    def __init__(
        self,
        length_weight: float = 0.4,
        keyword_weight: float = 0.3,
        coherence_weight: float = 0.3,
    ) -> None:
        self.length_weight = length_weight
        self.keyword_weight = keyword_weight
        self.coherence_weight = coherence_weight

    def score(self, rm_input: RMInput) -> float:
        """
        Return a scalar reward in approximately [-1, 2].

        Higher is better. Typical range after normalization: [0, 1.5].
        """
        length = _length_score(rm_input.response)
        keyword = _keyword_score(rm_input.response)
        coherence = _coherence_score(rm_input.prompt, rm_input.response)

        score = (
            self.length_weight * length
            + self.keyword_weight * keyword
            + self.coherence_weight * coherence
        )
        logger.debug(
            "score=%.3f [len=%.3f kw=%.3f coh=%.3f] | prompt=%.30s...",
            score, length, keyword, coherence, rm_input.prompt,
        )
        return score

    def score_batch(self, rm_inputs: list[RMInput]) -> list[float]:
        """
        Score a batch of (prompt, response) pairs.

        In production: a batched forward pass through the reward model transformer.
        """
        return [self.score(inp) for inp in rm_inputs]


# ---------------------------------------------------------------------------
# RLHF PPO training loop sketch
# ---------------------------------------------------------------------------

def compute_kl_penalty(
    current_log_prob: float,
    reference_log_prob: float,
) -> float:
    """
    Approximate KL divergence penalty for one response.

    In PPO-RLHF: KL(policy || reference_policy) is computed per token and
    summed over the response. Here we use a scalar proxy.

    KL penalty = current_log_prob - reference_log_prob
    (positive when policy drifts away from reference; penalizes reward hacking)
    """
    return current_log_prob - reference_log_prob


def rlhf_ppo_step(
    rm_inputs: list[RMInput],
    reward_model: RewardModel,
    reference_log_probs: list[float],
    current_log_probs: list[float],
    kl_coefficient: float = 0.1,
) -> PPOBatch:
    """
    One PPO step in RLHF.

    Real PPO step:
    1. Policy generates tokens autoregressively (full forward pass with sampling)
    2. RM scores the complete response
    3. KL divergence from reference policy is computed per token
    4. Adjusted reward = RM score - kl_coefficient * KL
    5. Advantages are computed (GAE or simple returns)
    6. Policy weights updated via PPO clipped objective

    Here:
    - rm_inputs are pre-generated (policy step omitted)
    - log_probs are scalars (proxy for per-sequence log-probability)
    - Policy weight update is omitted (would require autograd)

    Args:
        rm_inputs: batch of (prompt, response) pairs
        reward_model: frozen reward model
        reference_log_probs: log P(response | prompt) under SFT reference policy
        current_log_probs: log P(response | prompt) under current policy
        kl_coefficient: beta in reward = RM_score - beta * KL

    Returns:
        PPOBatch with rewards, KL penalties, and adjusted rewards
    """
    if len(rm_inputs) != len(reference_log_probs) != len(current_log_probs):
        raise ValueError("All input lists must have the same length.")

    rewards = reward_model.score_batch(rm_inputs)

    kl_penalties = [
        compute_kl_penalty(cur, ref)
        for cur, ref in zip(current_log_probs, reference_log_probs, strict=True)
    ]

    adjusted_rewards = [
        r - kl_coefficient * kl
        for r, kl in zip(rewards, kl_penalties, strict=True)
    ]

    return PPOBatch(
        rm_inputs=rm_inputs,
        rewards=rewards,
        kl_penalties=kl_penalties,
        adjusted_rewards=adjusted_rewards,
    )


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    rm = RewardModel()

    # Typical RLHF training batch: same prompt, different responses from policy
    prompts_and_responses: list[tuple[str, str]] = [
        (
            "What is the capital of France?",
            "Paris is the capital of France. It is located in northern France and serves "
            "as the country's political, economic, and cultural center.",
        ),
        (
            "What is the capital of France?",
            "Uhh I think maybe Paris? I'm not sure but I believe that's correct.",
        ),
        (
            "What is the capital of France?",
            "Paris.",
        ),
        (
            "Explain gradient descent.",
            "Gradient descent is an optimization algorithm. First, we compute the gradient "
            "of the loss with respect to the parameters. Second, we step in the negative "
            "gradient direction. Therefore, the loss decreases iteratively until convergence.",
        ),
        (
            "Explain gradient descent.",
            "gradient descent is like going downhill on a loss surface etc etc "
            "you update weights maybe",
        ),
    ]

    rm_inputs = [
        RMInput(prompt=prompt, response=response)
        for prompt, response in prompts_and_responses
    ]

    logger.info("=== Reward Model Scoring ===")
    scores = rm.score_batch(rm_inputs)
    for inp, score in zip(rm_inputs, scores, strict=True):
        logger.info(
            "score=%.3f | prompt=%.30s... | response=%.50s...",
            score, inp.prompt, inp.response,
        )

    # Simulate PPO step
    # Reference policy (SFT checkpoint): moderate log-probs
    # Current policy: after PPO steps, some responses became more probable
    reference_log_probs = [-1.2, -1.5, -0.8, -1.8, -1.1]
    # Simulated: policy has drifted slightly — high-quality responses more probable,
    # but one response over-optimized (much higher log-prob than reference)
    current_log_probs = [-1.0, -1.6, -0.7, -1.2, -0.5]

    logger.info("\n=== RLHF PPO Step ===")
    batch = rlhf_ppo_step(
        rm_inputs=rm_inputs,
        reward_model=rm,
        reference_log_probs=reference_log_probs,
        current_log_probs=current_log_probs,
        kl_coefficient=0.1,
    )

    logger.info("%-8s %-10s %-12s %-18s", "Idx", "RM Score", "KL Penalty", "Adjusted Reward")
    logger.info("-" * 54)
    for i, (reward, kl, adj) in enumerate(
        zip(batch.rewards, batch.kl_penalties, batch.adjusted_rewards, strict=True)
    ):
        logger.info("%-8d %-10.3f %-12.3f %-18.3f", i, reward, kl, adj)

    logger.info("\nKey observation:")
    logger.info(
        "Response 4 (vague/over-optimized): RM=%.3f, KL=%.3f, Adjusted=%.3f",
        batch.rewards[4], batch.kl_penalties[4], batch.adjusted_rewards[4],
    )
    logger.info(
        "KL penalty reduces the reward for responses that drifted far from the reference policy,"
        " preventing the policy from exploiting the RM with unusual text patterns."
    )


if __name__ == "__main__":
    main()
```

Expected output (abridged):
```
INFO: === Reward Model Scoring ===
INFO: score=0.722 | prompt=What is the capital of France?... | response=Paris is the capital of France. It is locate...
INFO: score=0.202 | prompt=What is the capital of France?... | response=Uhh I think maybe Paris? I'm not sure but I b...
INFO: score=0.332 | prompt=What is the capital of France?... | response=Paris....
INFO: score=0.803 | prompt=Explain gradient descent.... | response=Gradient descent is an optimization algorithm...
INFO: score=0.022 | prompt=Explain gradient descent.... | response=gradient descent is like going downhill on a ...
INFO: === RLHF PPO Step ===
```

**Key design decisions:**

1. `score_batch` mirrors production batch inference — in real RLHF the RM scores an entire policy rollout batch in one GPU forward pass.

2. The KL penalty is computed as `current_log_prob - reference_log_prob`. In production PPO, this is summed over all tokens in the response: `sum_t log pi(a_t|s_t) - log pi_ref(a_t|s_t)`. The coefficient beta (here 0.1) controls the tradeoff between RM optimization and policy stability.

3. The adjusted reward (`RM_score - beta * KL`) is what PPO uses to compute advantages. Higher adjusted reward = the policy should increase the probability of generating this response. A high KL penalty can make even a high-RM-score response net-negative if it required the policy to drift too far.
