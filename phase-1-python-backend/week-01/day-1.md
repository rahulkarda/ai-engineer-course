# Phase 1 · Week 01 · Day 1 — Python Setup, Interpreter, Virtual Environments, Packages

> Objective: Install Python 3.12, understand how the interpreter works, and isolate project dependencies with virtual environments.  ·  Time: ~1 hr  ·  Prereqs: none

## Concept

### Installing Python 3.12

**macOS** — Use [Homebrew](https://brew.sh):
```bash
brew install python@3.12
python3.12 --version
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

**Windows** — Download the installer from [python.org](https://www.python.org/downloads/). Check "Add Python to PATH" during install. Verify:
```powershell
python --version
```

Use `python3` (or `python3.12`) on macOS/Linux to avoid accidentally invoking an OS-bundled Python 2.

---

### The Python Interpreter

Python has two execution modes:

**REPL (Read-Eval-Print Loop)** — interactive mode. Launch with `python3`. Each expression is evaluated immediately and its result printed. Good for exploration.
```
$ python3
>>> 2 + 2
4
>>> "hello".upper()
'HELLO'
>>> quit()
```

**Script mode** — run a `.py` file end-to-end:
```bash
python3 script.py
```

---

### Virtual Environments

A **virtual environment** is an isolated Python installation scoped to one project. Without it, every project shares one global `site-packages` directory — installing `requests==2.28` for project A breaks project B that needs `requests==2.31`.

A venv solves this by creating a self-contained directory with:
- its own Python binary (symlink to the system interpreter)
- its own `site-packages` directory
- its own `pip`

---

### pip and requirements.txt

`pip` is Python's package installer. Key commands:

```bash
pip install rich            # install latest
pip install rich==13.7.0    # pin exact version
pip uninstall rich
pip list                    # show installed packages
pip freeze                  # output pinned list (use for requirements.txt)
pip install -r requirements.txt  # install from a file
```

`requirements.txt` is a plain text file listing packages and versions. Commit it to version control so teammates reproduce your exact environment.

---

### Project Structure

A minimal Python project:
```
my-project/
├── .venv/               # virtual environment (never commit this)
├── src/
│   └── my_project/
│       └── main.py
├── requirements.txt
└── .gitignore           # must include .venv/
```

## How it works internally

When you run `python3 script.py`, the interpreter does this in sequence:

1. **Tokenize** — split source text into tokens (keywords, identifiers, literals).
2. **Parse** — build an Abstract Syntax Tree (AST) from the tokens.
3. **Compile** — convert the AST to CPython **bytecode** (a stack-based instruction set). The result is cached in `__pycache__/script.cpython-312.pyc` to skip re-compilation on subsequent runs if the source is unchanged.
4. **Execute** — the CPython virtual machine interprets the bytecode, instruction by instruction.

You can inspect bytecode with the `dis` module:
```python
import dis

def add(a: int, b: int) -> int:
    return a + b

dis.dis(add)
```

A **virtual environment** works by manipulating `sys.path` — the list of directories Python searches when you `import` a module. Activating a venv prepends its `site-packages` to `sys.path`, so `import requests` finds the venv's copy, not the global one.

## Worked example

Create a project from scratch, install a dependency, write a script, and freeze requirements.

```bash
# 1. Create project directory
mkdir expense-tracker
cd expense-tracker

# 2. Create virtual environment
python3.12 -m venv .venv

# 3. Activate (macOS/Linux)
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# 4. Confirm you're using the venv's Python
which python   # should show .venv/bin/python

# 5. Install rich (a terminal formatting library)
pip install rich

# 6. Write a script
```

Create `src/hello.py`:
```python
# src/hello.py
from rich.console import Console

console = Console()

def greet(name: str) -> None:
    console.print(f"[bold green]Hello, {name}![/bold green]")

if __name__ == "__main__":
    greet("AI Engineer")
```

```bash
# 7. Run it
python src/hello.py

# 8. Freeze dependencies
pip freeze > requirements.txt
cat requirements.txt
# rich==13.x.x
# (plus rich's transitive deps)

# 9. Deactivate when done
deactivate
which python   # back to system Python
```

To recreate the environment on another machine:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exercises

1. Install Python 3.12 on your machine. Verify with `python3.12 --version` and paste the output.

2. Create a virtual environment named `.venv` inside a new project directory. Activate it and verify the active Python path with `which python` (macOS/Linux) or `where python` (Windows).

3. With the venv active, install `requests` and `httpx`. Freeze them to `requirements.txt`. Deactivate the venv and run `which python` again — show that the path changed back to the system Python.

4. Create a second empty venv named `.venv2`. Activate it and run `pip install -r requirements.txt`. Verify both packages installed with `pip list`.

5. In two sentences, explain what happens to a package you install globally (outside a venv) when you activate a venv, and why that isolation matters on a team with multiple projects.

(Solutions: ../solutions/week-01/day-1.md)

## Self-check

- [ ] I can explain what the CPython bytecode compilation step does and where `.pyc` files live
- [ ] I can create, activate, and deactivate a virtual environment from memory
- [ ] I can install packages, freeze requirements, and recreate an environment on a clean machine
- [ ] I understand why `sys.path` is the mechanism that makes venv isolation work

---

## Supplementary

- [Python 3.12 venv docs](https://docs.python.org/3.12/tutorial/venv.html)
- [CS50P — Introduction to Programming with Python](https://cs50.harvard.edu/python/)
- [pip user guide](https://pip.pypa.io/en/stable/user_guide/)
