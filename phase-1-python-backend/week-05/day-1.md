# Phase 1 · Week 05 · Day 1 — Git: Repos, Commits, Branches & Pull Requests

> Objective: Use Git fluently for solo and collaborative work — init, stage, commit, branch, merge, rebase, and open a PR.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-7]] (any editor/terminal familiarity)

## Concept

### Why version control?

Without VCS you either overwrite history or pile up `file_v2_final_FINAL.py` copies. Git gives you:
- A complete timeline of every change, with who made it and why.
- Safe parallel development via branches — experiment without touching `main`.
- Collaboration primitives: pull requests let teammates review before code merges.

### The three-zone mental model

```
Working tree  →  Staging area (index)  →  Local repo  →  Remote repo
  (edit)           (git add)              (git commit)    (git push)
```

Changes live in the **working tree** until you `add` them to the **index** (staging). `commit` bakes a snapshot into the **local repo**. `push` uploads that snapshot to a remote (GitHub, GitLab, etc.).

### Commits are snapshots, not diffs

A Git commit stores:
- A tree (snapshot of all tracked files).
- Parent commit SHA(s).
- Author, timestamp, message.

Git computes diffs on demand from two snapshots. This is why `git log -p` is always available.

### Branches

A branch is just a lightweight pointer (a file with a 40-char SHA inside `.git/refs/heads/`). Creating a branch costs nothing — no files are copied. `HEAD` is a pointer to the current branch tip.

```
main:   A─B─C
              ↑ HEAD (if on main)

feature:  A─B─C─D─E
                    ↑ HEAD (if on feature)
```

### Merge vs Rebase

**Merge** creates a merge commit that joins two branch tips:
```
main:    A─B─C───M
                ╱
feature:      D─E
```
History is preserved exactly as it happened. Merge commits are explicit.

**Rebase** replays `feature` commits on top of `main`, rewriting their parent pointers:
```
Before:  main: A─B─C   feature: A─B─D─E
After:   main: A─B─C─D'─E'  (D,E replayed with new SHAs)
```
Result: a clean linear history. Trade-off: rewrites SHAs, so **never rebase commits that others have pulled**.

Rule of thumb: rebase local feature branches before pushing; merge when integrating to `main`.

### Pull Requests

A PR (GitHub term; GitLab calls it MR) is not a Git concept — it's a platform feature. It wraps a branch comparison and provides a review surface: comments, approvals, CI checks, then a merge button. The workflow:

1. Create feature branch locally.
2. Push branch to remote.
3. Open PR: base `main` ← compare `feature`.
4. Teammates review, CI runs.
5. Merge (or rebase-merge, or squash-merge).

## How it works internally

### SHA-1 content addressing

Every Git object (blob, tree, commit, tag) is identified by SHA-1 of its content. This means two identical files get one stored blob. It also means you cannot silently alter history — the SHA chain is a Merkle tree; changing any commit changes all descendant SHAs.

### The index (staging area)

`.git/index` is a binary file mapping paths to blob SHAs. `git add` updates this file; `git commit` turns it into a tree object. This is why you can stage part of a file (`git add -p`) — you're editing the index, not the working tree.

### Fast-forward merge

When `main` is a direct ancestor of `feature`, Git can just advance the `main` pointer to `feature` tip — no merge commit needed. This is called a **fast-forward** merge. Use `--no-ff` to force a merge commit (keeps explicit branch topology in history).

### Conflict resolution

A conflict occurs when two branches modify the same lines. Git writes conflict markers into the file:
```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> feature
```
You edit the file to the desired state, `git add` it, then `git commit` to finish the merge.

## Worked example

```bash
# ── 1. New repo ───────────────────────────────────────────────────────────────
mkdir ledger && cd ledger
git init
echo "# Ledger" > README.md
git add README.md
git commit -m "chore: initial commit"

# ── 2. Feature branch ─────────────────────────────────────────────────────────
git checkout -b feature/add-ledger-module
# (create src/ledger.py with some content)
git add src/ledger.py
git commit -m "feat(ledger): add basic ledger class"

# ── 3. Another commit on the branch ───────────────────────────────────────────
# (edit src/ledger.py)
git add -p src/ledger.py     # stage interactively — review each hunk
git commit -m "feat(ledger): add debit/credit methods"

# ── 4. Rebase onto latest main before PR ──────────────────────────────────────
git checkout main
echo "# updated" >> README.md
git commit -am "docs: update readme"

git checkout feature/add-ledger-module
git rebase main              # replay feature commits on top of updated main

# ── 5. Push and open PR ───────────────────────────────────────────────────────
git push -u origin feature/add-ledger-module
# Then open PR on GitHub: base main ← compare feature/add-ledger-module

# ── 6. Simulate a conflict ────────────────────────────────────────────────────
git checkout main
echo "line from main" >> shared.txt && git add shared.txt && git commit -m "fix: main change"

git checkout feature/add-ledger-module
echo "line from feature" >> shared.txt && git add shared.txt && git commit -m "fix: feature change"

git rebase main
# CONFLICT in shared.txt — edit to resolve, then:
git add shared.txt
git rebase --continue
```

### .gitignore essentials

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
dist/
*.egg-info/

# Environment / secrets
.env
*.env
.env.*
!.env.example

# Editor
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

Key rule: `.env` must always be ignored. `.env.example` (with placeholder variable names, no values) must be committed.

## Exercises

1. Create a new Git repo `bank-app`. Make three commits on `main`: add `README.md`, add `src/account.py` (any content), add `tests/` directory with a placeholder file. Inspect the full log with `git log --oneline --graph`.

2. Create a branch `feature/transfer` from `main`. Add two commits. Then, while on `feature/transfer`, run `git log --oneline --graph --all` and describe the branch topology you see.

3. From a clean state (both branches have separate changes to the same line), trigger a merge conflict. Resolve it manually, finish the merge, and verify with `git log`.

4. Write a `.gitignore` for a Python project that has a `.env` secrets file, a `venv/` directory, and `__pycache__` folders. Verify with `git status` that those paths appear as "Ignored" after you create them.

5. Explain in one paragraph: why does rebasing rewrite commit SHAs? Why is it unsafe to rebase commits that others have already pulled?

(Solutions: ../solutions/week-05/day-1.md)

## Self-check
- [ ] I can init a repo, stage specific files, and write a conventional commit message.
- [ ] I can create and switch branches, and explain what HEAD points to.
- [ ] I understand the merge vs rebase trade-off and when to use each.
- [ ] I can resolve a merge conflict by editing conflict markers and completing the merge.
- [ ] I have a working `.gitignore` that excludes `.env` and Python build artifacts.
- [ ] I can push a branch and describe the steps to open a PR.

## Supplementary links
- [Pro Git Book (free)](https://git-scm.com/book/en/v2)
- [Conventional Commits spec](https://www.conventionalcommits.org/)
- [GitHub Docs — About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
- [Atlassian — Merging vs Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
