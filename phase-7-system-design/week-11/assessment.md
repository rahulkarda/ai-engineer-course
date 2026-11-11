# Week 11 Assessment — Project Deep-Dives, Portfolio, and FDE Interview Readiness

> Pass gate: 75%  ·  Time allowed: 120 minutes  ·  Note: non-coding week — assessment adapted (see rubric note)

---

## Instructions

This assessment covers the full Week 11 material: project deep-dive structure, resume engineering, FDE interview preparation, GitHub portfolio polish, and career narrative construction.

All answers are written responses. No code required. Where a question asks for a number or a specific format, use it — vague answers score partial credit at best.

Total: 100 points. Pass: 75 points.

---

## Part 1: Conceptual (25 points)

Answer 7 of the 8 questions. If you answer all 8, your best 7 scores count. Each question is worth approximately 3-4 points.

**Q1.** What distinguishes a Forward-Deployed Engineer role from a Solutions Engineer or a standard SWE? Name two specific interview signals an FDE panel looks for that a standard SWE panel does not prioritize.

**Q2.** Describe the "5-bullet rule" for project talking points. What does each bullet cover? Why 5 and not 3 or 10?

**Q3.** An interviewer asks: "Walk me through your RAG system." A candidate responds by listing all the technologies used. What is wrong with this response, and what should they say instead?

**Q4.** What is the Clarify → Constrain → Prototype → Communicate framework? Give a concrete example of what "Constrain" looks like in practice — write the actual sentence a candidate would say.

**Q5.** What are the two jobs a resume must do for an AI/ML role? How do these goals conflict, and what is the standard resolution?

**Q6.** Name 4 items from the 10-point GitHub audit checklist that are most likely to create a negative signal with a technical interviewer. Explain why each matters.

**Q7.** How does the Projects section of an AI/ML resume differ from the Experience section in terms of content, format, and purpose?

**Q8.** An FDE interviewer asks: "You have 3 days on-site at a customer. They want RAG over their documents, a Slack notification system, and an API integration with their CRM. What do you do?" Describe how you would approach this answer — what you would say first, and why.

---

## Part 2: Deep-Dive Mock (40 points)

Choose one of the 4 portfolio projects:
- Python backend platform
- Advanced RAG system
- Multi-agent operations platform
- Production AI platform

Write a complete deep-dive narrative for the project you chose. Structure your response as follows:

**2a. Architecture summary (6 pts)**
Describe the major components of the system and how data flows through them. You may use ASCII diagram notation or a written component list with flow descriptions. Must include: data sources, processing components, external services, storage, and at least one identified system boundary.

**2b. Three key decisions with tradeoffs (12 pts — 4 pts each)**
For each decision:
- State the decision clearly (one sentence)
- Give the context that made this decision necessary
- Name the 2-3 alternatives you considered
- Explain why you chose what you chose
- Describe the outcome — did the decision hold? What did you give up?

**2c. Scaling story (8 pts)**
Describe the system at prototype stage and at production stage (or projected production if not yet deployed). Include:
- What the prototype handled (users, requests, data volume)
- What broke first, and how
- What architectural changes addressed that failure
- What the system handles now (or would handle at production scale)
- At least 2 concrete numbers (latency, throughput, cost per request, dataset size, etc.)

**2d. "What I would do differently" (6 pts)**
Name one thing you would change if starting this project over. Be specific — name the exact component, decision, or process. Explain what evidence or experience led you to this conclusion. Explain what the outcome would have been with the change.

This section is worth 6 points because honest retrospection is a strong signal. Vague regrets ("I'd document more") score 2/6. Specific, technically grounded retrospectives with a clear counterfactual score 5-6/6.

**2e. Five anticipated Q&As (8 pts — 1.6 pts each)**
Write 5 questions a strong interviewer would ask about this specific project, plus your model answer for each. Each answer: 3-6 sentences. At least one Q&A must address a failure or a mistake. At least one must address cost or resource consumption.

---

## Part 3: Resume and Narrative Review (20 points)

**3a. Rewrite a weak bullet (8 pts)**

The following bullet appears on a candidate's resume under a project called "Customer Churn Prediction Model":

> "Helped develop a machine learning model to predict customer churn using Python and worked with the data team to get results."

Rewrite this bullet 3 times, each iteration adding more signal. Label each iteration. Your final version must:
- Start with a strong past-tense action verb
- Name the specific technique or technology
- Include a result or scale metric
- Be one line (two lines maximum)

Explain in one sentence what each iteration adds that the previous version lacked.

**3b. Critique a README section (6 pts)**

The following is the opening of a GitHub README for a multi-agent orchestration project:

---
*Project Name: agent-thing*

*This is my project for the AI engineering course. It uses LangGraph and some tools. To run it, you need Python. Install requirements.txt and then run main.py.*
---

List 5 specific problems with this README section. For each problem, write the one-line fix.

**3c. FDE vs. MLOps portfolio tailoring (6 pts)**

You are applying to two roles simultaneously:
- FDE at an enterprise AI vendor (customer-facing, solutions-building)
- MLOps Engineer at a fintech startup (internal platform, reliability-focused)

For each role, identify:
- Which 2 of your 4 capstone projects to lead with and why
- One framing change to the Projects section that better targets that role
- One change to the resume summary line

You may write this as two short paragraphs (one per role) or as a comparison table.

---

## Part 4: Reflection (15 points)

Answer all 3 questions. There are no wrong answers — scoring is based on specificity and honesty, not on the content of your self-assessment.

**4a. Readiness assessment (5 pts)**
Rate your interview readiness on a scale of 1-10 across three dimensions: technical depth (can I answer hard questions about what I built), narrative quality (can I tell the story well), and portfolio completeness (does my GitHub/resume show what I know). For each dimension, explain the specific gap between where you are now and a 10. What would you need to do to close each gap?

**4b. Biggest learning from this week (5 pts)**
What is the single most useful thing you learned or clarified this week — not in general, but specifically something that changed how you are thinking about the job search or your own projects? Be specific: name the concept, the moment it clicked, and one concrete action you will take because of it.

**4c. The hardest question you will face (5 pts)**
Write the one interview question about your portfolio or background that you are least prepared to answer well right now. Explain why it is hard for you specifically. Then write the best answer you can give today — even if it is incomplete. What would you need to learn or build to give a strong answer 4 weeks from now?

---

## Rubric (pass >= 75%)

| Dimension | Weight | Points |
|---|---|---|
| Deep-dive narrative (replaces independent implementation) | 40% | 40 |
| Conceptual understanding | 25% | 25 |
| Resume/portfolio craft (replaces debugging) | 20% | 20 |
| Reflection and self-awareness (replaces code quality) | 15% | 15 |
| **Total** | **100%** | **100** |

**Scoring notes per part:**

Part 1 (Conceptual, 25 pts): Full credit requires specific, accurate answers. Generic answers that could apply to any engineering role score 50%. Answers that show you understand the distinction between AI/ML career tracks score 80-100%.

Part 2 (Deep-Dive, 40 pts): Specificity is the primary signal. Answers with project-specific numbers, named components, and explicit tradeoffs score in the upper range. Answers that could describe any RAG system or any backend without naming your specific decisions score in the lower range.

Part 3 (Resume, 20 pts): The bullet rewrite is scored on the quality of the final version and on whether each iteration genuinely improves on the previous. The README critique is scored on specificity — "this is vague" scores lower than "the setup instructions omit the required environment variables."

Part 4 (Reflection, 15 pts): Scored on specificity and honesty. Self-assessments that are entirely positive score lower than ones that identify a real gap and a plausible path to closing it.

**Note: This is a non-coding week.** Standard rubric dimensions have been adapted — "independent implementation" replaced with deep-dive narrative quality; "debugging" replaced with resume/portfolio craft; "code quality" replaced with reflection and self-awareness.

---

Answer key: ../solutions/week-11/assessment.md
