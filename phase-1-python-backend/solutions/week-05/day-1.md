# Solutions — Week 05 · Day 1 (Git)

## Exercise 1 — Three commits on main

```bash
mkdir bank-app && cd bank-app
git init
echo "# Bank App" > README.md
git add README.md
git commit -m "chore: initial commit"

mkdir -p src tests
echo "# account module" > src/account.py
git add src/account.py
git commit -m "feat(account): add account module placeholder"

touch tests/.gitkeep
git add tests/.gitkeep
git commit -m "chore(tests): add tests directory"

git log --oneline --graph
# * abc1234 chore(tests): add tests directory
# * def5678 feat(account): add account module placeholder
# * ghi9012 chore: initial commit
```

## Exercise 2 — Branch topology

```bash
git checkout -b feature/transfer
echo "def transfer(): pass" >> src/account.py
git add src/account.py
git commit -m "feat(transfer): stub transfer function"

echo "# transfer notes" >> README.md
git add README.md
git commit -m "docs: add transfer notes"

git log --oneline --graph --all
# * abc1234 (HEAD -> feature/transfer) docs: add transfer notes
# * def5678 feat(transfer): stub transfer function
# * 111aaaa (main) chore(tests): add tests directory
# * 222bbbb feat(account): add account module placeholder
# * 333cccc chore: initial commit
```

Explanation: `feature/transfer` is two commits ahead of `main`. `HEAD` points to `feature/transfer`. The graph shows both pointers on the same commit history up to the branch point.

## Exercise 3 — Resolve a conflict

```bash
# Setup: both branches modify the same line in README.md
git checkout main
echo "main contact: alice@example.com" >> README.md
git commit -am "docs: add main contact"

git checkout feature/transfer
echo "main contact: bob@example.com" >> README.md
git commit -am "docs: add feature contact"

git merge main
# CONFLICT in README.md
# Edit README.md — remove conflict markers, keep desired content:
# main contact: alice@example.com
git add README.md
git commit -m "merge: resolve contact conflict, keep main"

git log --oneline --graph
# *   abc (HEAD -> feature/transfer) merge: resolve contact conflict
# |\
# | * def (main) docs: add main contact
# * | ghi docs: add feature contact
# |/
# * ... earlier commits
```

## Exercise 4 — .gitignore

```gitignore
# .gitignore
__pycache__/
*.pyc
*.pyo
.venv/
venv/
dist/
*.egg-info/
.env
*.env
.env.*
!.env.example
.vscode/
.idea/
*.swp
.DS_Store
```

Verify:
```bash
mkdir -p __pycache__ .venv
touch .env __pycache__/test.pyc
git status  # these should show as "Ignored" with git status --ignored
```

## Exercise 5 — Rebase rewrites SHAs (written answer)

Git commit SHAs are computed as SHA-1 of the commit's content: tree hash, parent hash(es), author, timestamp, and message. When you rebase, each rebased commit gets a new parent (the tip of the target branch). Since the parent hash is part of the input to SHA-1, the output SHA changes — even if the diff is identical.

If a teammate has already pulled the original commits, their local history includes the old SHAs. After a rebase, those SHAs no longer exist on the remote. The teammate's `git pull` or `git merge` will see two diverged histories and produce a confusing merge that re-introduces the original commits. The golden rule: **never rebase commits that others have based work on**.
