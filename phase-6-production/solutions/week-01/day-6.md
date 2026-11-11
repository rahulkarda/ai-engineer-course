# Solutions — Day 6: Build Session (Pointer + Self-check)

Day 6 is a guided implementation day. The full reference implementation is provided inline in the lesson (`day-6.md`) — every file (`models.py`, `network_sim.py`, `node.py`, `cluster.py`) and every test is written out in full.

There is no separate solution to reveal; the lesson IS the walkthrough.

## Self-check for Day 6

If your tests are all green, you have successfully implemented the reference design. Use these questions to verify your understanding before the assessment:

1. **`NetworkSim.should_deliver`** — what two conditions cause a replication message to be dropped? How does `stats` track them?

2. **`KVNode.write` idempotency** — what happens when the same `idempotency_key` is submitted twice? Where is the key stored and how is it checked?

3. **`KVNode.apply_replication` LWW** — what is the condition for discarding an incoming replication message? Why does this prevent an old write from overwriting a newer one?

4. **`Cluster.quorum_read`** — in the worst case, how many nodes return `None`? Does `quorum_read` still work? Why?

5. **`Cluster.sync_replicas`** — when does this differ from normal replication? What does it represent in a real system (hint: partition heal)?

## Common bugs to check

- Forgetting to update `is_stale` on `ReadResult` — non-primary reads should have `is_stale=True`
- Using `time.time()` instead of `time.monotonic()` for versions — `time.time()` can go backward
- Mutating the `Entry` after creation — use `model_copy` if you need a modified copy
- Not initialising `__init__.py` files — `src/` and `tests/` both need them
- Using `print()` anywhere — replace with `logging`
