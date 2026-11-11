# Solutions — Phase 1 · Week 01 · Day 1 — Python Setup, Interpreter, venv, Packages

---

## Exercise 1 — Verify your Python installation

**Command:**
```bash
python --version
# or on some systems:
python3 --version
```

**Expected output:**
```
Python 3.12.x
```

Why it works: The `--version` flag asks the Python interpreter to print its version string and exit. Use `python3` on macOS/Linux where `python` may still point to Python 2.

---

## Exercise 2 — Create and activate a virtual environment

```bash
# Create the venv in a hidden folder called .venv
python -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Confirm the active interpreter is inside the venv
which python          # macOS/Linux → /path/to/project/.venv/bin/python
Get-Command python    # Windows
```

Why it works: `python -m venv .venv` copies (or symlinks) a fresh Python binary into `.venv/bin/` and creates an isolated `site-packages/` directory. Activating the venv prepends `.venv/bin` to your `PATH`, so every subsequent `python` and `pip` call uses that isolated copy instead of the system one.

---

## Exercise 3 — Install packages and freeze requirements

```bash
# Install two packages into the active venv
pip install requests httpx

# Capture exact versions to a requirements file
pip freeze > requirements.txt

# Inspect what was recorded
cat requirements.txt
# certifi==...
# httpx==...
# requests==...
# ... (transitive dependencies also listed)

# Deactivate the venv
deactivate

# Confirm we're back on the system Python
which python   # /usr/bin/python3 or similar — NOT inside .venv
```

Why it works: `pip freeze` dumps every installed package with its pinned version in `name==version` format. Pinning versions makes builds reproducible — anyone who installs from this file gets the exact same dependency tree.

---

## Exercise 4 — Recreate the environment from requirements.txt

```bash
# Create a second, empty venv
python -m venv .venv2

# Activate it
source .venv2/bin/activate

# Install everything recorded in the freeze
pip install -r requirements.txt

# Verify packages are present
pip list
# Output should include requests and httpx at the same versions

deactivate
```

Why it works: `-r requirements.txt` tells pip to read each `name==version` line and install them in one pass. This is how CI systems and teammates reproduce your exact environment from scratch.

---

## Exercise 5 — Written answer: why avoid global installs?

**Question:** Why is `pip install <package>` without a venv considered bad practice?

**Answer:**

Global installs write packages into the system Python's single `site-packages/` directory. Two problems follow immediately:

1. **Version conflicts.** `projectA` needs `requests==2.28.2` for an old API; `projectB` needs `requests==2.31.0` for a new feature. Only one can be installed globally — one project breaks.

2. **Pollution and reproducibility.** The system Python accumulates unrelated packages over time. `pip freeze` on a global install lists dozens of packages that have nothing to do with your project, making `requirements.txt` noisy and brittle.

A venv isolates each project's dependencies so versions never collide and `pip freeze` is clean.

---

## Self-check answers

- **Can you find the Python binary path?** `which python` (macOS/Linux) or `Get-Command python` (Windows).
- **What does `pip freeze` do?** Prints all installed packages with pinned versions in `name==version` format.
- **What does `deactivate` do?** Removes `.venv/bin` from `PATH`, restoring the original shell environment.
