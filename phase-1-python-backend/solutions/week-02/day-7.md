# Solutions — Week 02 · Day 7

Day 7 is a build and reflection day — no new exercise solutions. See `solutions/week-02/project.md` for the complete reference implementation of the Financial Transaction Processor.

## Mental model check — model answers

**Question 1**: When would you choose a `set` over a `list` for storing transaction IDs?

Use a `set` when you need fast membership testing (`id in seen`) or uniqueness enforcement. `x in set` is O(1) vs O(n) for a list. You lose: order (sets are unordered), duplicate tracking (sets deduplicate automatically), and index access. If you need both membership speed and insertion order, use a `dict` keyed by ID.

**Question 2**: Explain `*args` and `**kwargs` without the words "positional" or "keyword".

`*args` collects any extra unnamed values passed to a function into a tuple inside the function. `**kwargs` collects any extra named values into a dict. They let you write functions that accept a flexible number of inputs without listing each one in the signature. You can forward them unchanged to another function with `fn(*args, **kwargs)`.

**Question 3**: What is wrong with catching bare `Exception`?

`except Exception: pass` silently discards every error — bugs, typos, network timeouts, corrupt data, programming errors — making the program appear to work when it is failing. The correct approach is: catch the specific exception types you know how to handle, log them with full context, and let unexpected exceptions propagate so you see them immediately.
