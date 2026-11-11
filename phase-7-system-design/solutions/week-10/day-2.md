# Solutions — Week 10 · Day 2: Building Your Story Bank (Competency Coverage)

## Exercise 2 — Failure story (worked example)

**Title:** Deployed a model to production without testing distribution shift  
**Competencies:** Failure/learning (P), Impact (S)

**Situation:**  
I was the ML engineer responsible for deploying a new churn prediction model to our customer success platform. The model had strong offline metrics (AUC 0.87 on the held-out test set) and had passed all pre-launch checks. Three weeks after launch, a customer success manager noticed that the churn alerts for enterprise customers had essentially stopped.

**Task:**  
I was responsible for the model's production performance. I needed to diagnose the regression and fix it, and I needed to do this without minimizing what had happened.

**Action:**
- Pulled production prediction distributions and immediately saw the problem: the model was outputting scores clustered near 0.08 for all enterprise customers — far below the alert threshold of 0.5. On the test set, enterprise customers had been a tiny fraction; the model had learned to suppress high scores for that segment.
- Escalated to the team lead before I had a full diagnosis — did not try to fix it quietly first. Wrote a one-paragraph internal incident note within 2 hours.
- Identified that our test set was sampled uniformly across all customer tiers, which underrepresented the enterprise segment by a factor of 12x relative to its revenue weight.
- Rebuilt the training pipeline with stratified sampling, retrained, and validated on a segment-stratified holdout before re-deploying.

**Result:**  
The fix was deployed 4 days after the regression was discovered. Zero enterprise customers were missed during the window — the customer success team had manually covered that period. I added a mandatory "prediction distribution by segment" check to the model evaluation dashboard and a segment-stratified holdout as a required step in our ML training template. The template has been used by 6 subsequent models. I learned that a model that "works" on a uniform test set can fail silently on a specific segment; the testing protocol, not the model, was the root-cause failure.

---

## Exercise 4 — 30-second version (worked example)

**Full story:** Deployed a model without testing distribution shift. Three weeks later, enterprise customer alerts stopped. Root cause: 12x underrepresentation of enterprise in training data. Fixed with stratified sampling, 4 days to resolution, added segment distribution checks that have caught issues in 6 subsequent models.

**30-second version (~60 words):**  
"I shipped a churn model that silently stopped alerting on enterprise customers because they were underrepresented in my training data by a factor of 12. I caught it after 3 weeks, fixed it in 4 days, and added a segment distribution check to our training template. That check has caught two similar issues since."

**When to use the 30-second version:** When an interviewer says "tell me briefly about a failure" — they're testing whether you can own it cleanly without over-explaining. Use it to open, then let them probe for detail if they want it. Use the full version only when the interviewer explicitly asks for more detail or when the opening question is specifically "walk me through what happened."
