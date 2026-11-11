# Phase 7 · Week 11 · Day 7 — Resume Draft + GitHub Polish (3-hour session)

> Objective: Produce a polished resume and audit GitHub profile/repos for job-application readiness  ·  Time: ~3 hr  ·  Prereqs: [[day-1]] through [[day-6]]

---

## Concept

A resume for an AI/ML role has two jobs: pass automated screening (ATS, recruiter keyword scan) and give a technical interviewer enough signal to ask the right questions. These two goals partially conflict. This session handles both.

GitHub is the second resume. For engineering roles, hiring managers and technical interviewers will look at your GitHub before the first interview call. An unpolished profile creates doubt that a polished resume cannot repair.

---

## How It Works Internally

### Resume Construction: Section by Section

**Header**
- Name (large), location (city + remote OK if applicable), email, LinkedIn URL, GitHub URL, portfolio URL if it exists
- No photo, no objective statement, no references line
- LinkedIn and GitHub URLs must be clean vanity URLs (linkedin.com/in/yourname, github.com/yourname)

**Professional Summary (3-4 lines)**
Purpose: give a recruiter the 30-second version of who you are. Not a list of buzzwords. One to two sentences on what you build, one sentence on your technical depth, one sentence on your domain focus.

Weak: "Experienced software engineer with expertise in AI, machine learning, and Python development."
Strong: "AI/ML engineer building production LLM systems and multi-agent platforms. Background in distributed systems and MLOps; focused on moving models from prototype to reliable, observable production deployments."

**Experience (reverse chronological)**
Each role: Company, Title, Dates, Location (or Remote). Then 3-5 bullets.

Bullet formula: [Action verb] + [what you did] + [how/with what] + [result or scale]

Weak: "Worked on backend services and helped with ML pipeline improvements."
Strong: "Reduced model inference latency 40% by profiling and batching requests to the embedding service, cutting per-query cost from $0.0008 to $0.0003 at 1M daily requests."

Rules for bullets:
- Start every bullet with a past-tense action verb (Built, Reduced, Designed, Migrated, Automated, Deployed)
- Include one number in at least 2 of every 3 bullets
- No "responsible for" or "helped with" — own the work
- Keep to one line; two lines maximum

**Projects Section**
For each portfolio project: Project name (linked to GitHub), one-line description, technologies used (4-6 items), and 2-3 impact bullets.

Format:
```
Python Backend Platform | github.com/yourname/backend-platform
Production REST API service; FastAPI, PostgreSQL, pytest, Docker
- Handles 200 concurrent users at sub-200ms p95 latency
- Async job processing system with idempotent retry logic, 1,000 jobs/hr throughput
- 94% test coverage across unit, integration, and load tests
```

List all 4 capstone projects. This is the most important section of the resume for AI/ML roles.

**Skills**
Group by category. Do not list everything you have ever touched. List what you can be interviewed on tomorrow.

Categories that work for AI/ML roles:
- Languages: Python (primary), SQL, Bash
- Frameworks/Libraries: FastAPI, LangGraph, LangChain, Pydantic, pytest
- AI/ML: RAG systems, multi-agent architectures, LLM APIs, embeddings, vector search, LLM evaluation
- Infrastructure: Docker, Kubernetes, GitHub Actions, OpenTelemetry
- Data: PostgreSQL, Redis, Qdrant / Chroma / Weaviate (whichever you used)

---

### How to Tailor the Same Resume for Different Roles

Keep one "master resume" with all bullets. Create role-targeted variants by reordering and selecting:

**AI/ML Engineer variant**
- Lead with RAG system and multi-agent platform projects
- Emphasize: LLM integration, evaluation, embeddings, retrieval accuracy metrics
- Skills section: put AI/ML frameworks first
- Summary: mention model quality, evaluation, production reliability

**AI Platform Engineer / MLOps variant**
- Lead with production platform and backend projects
- Emphasize: Docker, k8s, CI/CD, observability, cost management, distributed execution
- Skills section: put Infrastructure first
- Summary: mention reliability, scalability, operational cost

**FDE / Solutions Engineer variant**
- Lead with the project most relevant to the customer's industry (usually RAG or multi-agent)
- Emphasize: speed of delivery, customer-facing demos, cross-functional communication
- Add a "What I build for customers" framing to the summary
- Skills section: put breadth first — show you can work across the stack

The changes are 20% of the resume. Never rewrite from scratch per application.

---

### GitHub Profile Audit: 10-Point Checklist

Work through this checklist for your GitHub profile:

**1. Profile README**
- [ ] github.com/yourname has a profile README (create a repo named exactly your username)
- [ ] README includes: 1-line bio, what you currently build, 2-3 highlighted technologies, links to portfolio or personal site

**2. Pinned repositories**
- [ ] Exactly 6 repos are pinned (GitHub allows 6)
- [ ] All 4 capstone projects are pinned
- [ ] No pinned repos with zero stars and a generic name (test-project, untitled, homework)

**3. Repository READMEs**
- [ ] Every pinned repo has a README with: what the project does (1 paragraph), architecture overview or diagram, setup instructions, and at least one screenshot or diagram
- [ ] README is written for someone who has never seen the code — not for yourself

**4. Contribution graph**
- [ ] Recent contributions visible (not all green, but not empty for months)
- [ ] If the graph is sparse: note that private work contributions can be made visible in GitHub settings

**5. Secrets check**
- [ ] Run `git log --all --full-history -- "*.env"` on each capstone repo — no .env files in history
- [ ] Run `grep -r "api_key\|API_KEY\|Bearer\|sk-\|token" --include="*.py" --include="*.json" --include="*.yaml"` on each capstone repo root — no hardcoded credentials

**6. .gitignore quality**
- [ ] Every Python repo has a .gitignore that excludes: __pycache__, .env, *.pyc, .pytest_cache, .venv, dist/, *.egg-info
- [ ] .env.example exists in every repo that uses environment variables

**7. License**
- [ ] Every public repo has a LICENSE file (MIT is fine for portfolio projects unless otherwise required)

**8. Commit message quality**
- [ ] Last 10 commits on each capstone repo follow conventional commit style or at minimum have descriptive messages — not "fix", "update", "stuff"

**9. Issue and PR hygiene**
- [ ] No open issues that are just TODO comments from 6 months ago
- [ ] If PRs were used during development, they are closed/merged — no dangling draft PRs from incomplete work

**10. Topics and tags**
- [ ] Each repo has 4-6 GitHub topics (e.g., fastapi, python, rag, langchain, llm, production-ai)
- [ ] Topics match what recruiters and engineers search for — not vanity tags

---

### Final Polish Tasks

**LinkedIn headline alignment**
LinkedIn headline should match your resume summary in substance, even if not word-for-word.
Current pattern to avoid: "Software Engineer at [Company]"
Target pattern: "AI/ML Engineer — LLM Systems, Multi-Agent Platforms, Production ML"

**Project URLs in resume**
Every project in the resume Projects section should have a live GitHub link. Test every link before submitting any application. A 404 on a portfolio link is an immediate negative signal.

**Portfolio page (if applicable)**
If you have a personal site or a GitHub Pages site, it should:
- Load in under 3 seconds
- Show the 4 capstone projects with screenshots or demos
- Link to both the GitHub repo and a live demo if one exists
- Have no broken links

---

## Worked Example

### Resume Bullet Transformation

Original (from a real AI engineer's early resume):
> "Was part of the team that built the RAG system for document search."

Problems: passive, no ownership, no number, no technology, no result.

Rewrite progression:
1. Add ownership: "Built the document chunking and embedding pipeline for a RAG system."
2. Add technology: "Built the document chunking and embedding pipeline for a RAG system using LangChain and Qdrant."
3. Add result: "Built the document chunking and embedding pipeline for a RAG system using LangChain and Qdrant; reduced mean retrieval latency from 1.8s to 340ms."
4. Add scale: "Built the chunking and embedding pipeline for a 50,000-document RAG system (LangChain, Qdrant); reduced mean retrieval latency from 1.8s to 340ms, improving analyst query completion rate by 35%."

Each iteration adds signal without adding length beyond what is justified.

---

## Exercises

1. Write the resume Projects section for all 4 capstone projects using the format specified above. Each entry: project name (linked), one-line description, technologies list, 2-3 impact bullets. Do not write "I am proud of" or "this was challenging" — only deliverables and numbers.

2. Run the 10-point GitHub audit checklist on your actual GitHub profile. Record: which items pass, which fail, and what the specific fix is for each failure. Complete the fixes during this session.

3. Rewrite your current LinkedIn headline and professional summary using the templates from this lesson. If you do not have a LinkedIn profile, draft both as if you were creating one today.

4. Choose one of the three role variants (AI/ML Engineer, AI Platform Engineer, FDE). Identify which 3 resume bullets from your experience section are most relevant for that variant and which 3 are least relevant. Explain your ranking — one sentence per bullet.

(Solutions: ../solutions/week-11/day-7.md)

---

## Self-check

- [ ] Resume has a Projects section with all 4 capstone projects, using the correct format
- [ ] Every bullet in the Projects section starts with an action verb and includes at least one number
- [ ] GitHub profile passes all 10 checklist items (or failures are documented with a fix plan)
- [ ] LinkedIn headline describes what you build, not just your current title
- [ ] I can explain in one sentence how I would adjust my resume for each of the 3 role variants

---

## Supplementary

- [Resume Worded — AI resume analyzer](https://resumeworded.com/) — free tool for ATS scoring and bullet-level feedback; useful for a quick sanity check on keyword coverage
- [GitHub Profile README generator (arturssmirnovs)](https://github.com/arturssmirnovs/github-profile-readme-generator) — generates a structured profile README template you can customize; faster than starting from scratch
- [The Pragmatic Engineer: Résumé tips for senior engineers](https://blog.pragmaticengineer.com/the-pragmatic-engineers-resume-template/) — field-tested advice from Gergely Orosz, focused on engineers targeting senior and staff roles at top-tier companies
