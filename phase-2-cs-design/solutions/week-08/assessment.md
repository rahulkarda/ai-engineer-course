# Solutions — Week 08 · Assessment

## Question 1 — REST Maturity Levels 2 vs 3

**Level 2** uses HTTP verbs and status codes correctly: `GET /orders/42`, `POST /orders`,
`DELETE /orders/42`, returns `201 Created` vs `404 Not Found`. Resources are addressable
and verbs carry semantic meaning.

**Level 3** adds **HATEOAS** (Hypermedia as the Engine of Application State): responses
embed links that tell the client what actions are available next and where to find them.

```json
// Level 2 response — client must know the URL structure out-of-band
{ "id": 42, "status": "pending", "amount": 99.99 }

// Level 3 response — client discovers transitions from the response itself
{
  "id": 42,
  "status": "pending",
  "amount": 99.99,
  "_links": {
    "self":    { "href": "/api/v1/orders/42" },
    "confirm": { "href": "/api/v1/orders/42/confirm", "method": "POST" },
    "cancel":  { "href": "/api/v1/orders/42/cancel",  "method": "DELETE" }
  }
}
```

**Practical difference:** Level 3 lets clients navigate the API without hard-coded URL
patterns. When the server changes a URL, existing clients still work as long as they
follow links. The cost is larger responses and more complex clients. Most production APIs
target Level 2; Level 3 is rare outside hypermedia-native domains (HAL, JSON:API, Siren).

---

## Question 2 — Idempotency in Payment APIs

**Why it is critical:** Networks are unreliable. A payment request may time out after the
server has already debited the account. Without idempotency, a retry debits twice. In
payments, double-charge is a regulatory and trust failure.

**Implementation:**

```python
# src/payment_service/routes/payments.py
import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/payments", tags=["payments"])

# In production: Redis or a DB table with (idempotency_key, response, created_at)
_idempotency_store: dict[str, dict] = {}


class PaymentRequest(BaseModel):
    account_id: str
    amount: float
    currency: str


@router.post("", status_code=201)
async def create_payment(
    body: PaymentRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> dict:
    # 1. Normalise the key to prevent trivial collisions
    key = hashlib.sha256(idempotency_key.encode()).hexdigest()

    # 2. Return the cached response if the key was seen before
    if key in _idempotency_store:
        logger.info("Idempotent replay for key %s", idempotency_key)
        return _idempotency_store[key]

    # 3. Process the payment (DB write, ledger entry, etc.)
    result = {
        "payment_id": f"pay_{key[:12]}",
        "status": "completed",
        "amount": body.amount,
        "currency": body.currency,
    }

    # 4. Persist the response before returning
    _idempotency_store[key] = result
    return result
```

**Rules:**
- Key must be client-generated (UUID v4), not server-generated.
- Store key → full response, not just status. Replay the exact same body.
- Key expiry (e.g., 24 h) prevents unbounded storage growth.
- The store write must be atomic with the payment commit (use a DB transaction, not a
  separate cache write after the fact, or accept that crash-between-commit-and-cache
  requires reconciliation on startup).

---

## Question 3 — Technical Layer vs Business Domain Decomposition

**Technical layer decomposition** splits by horizontal concern:

```
presentation-service  →  handles all HTTP
business-service      →  all domain logic
data-service          →  all DB access
```

Every user story touches all three layers, so a single feature change requires
coordinating deploys across all three services. This is **distributed monolith** — the
deployment boundary does not match the change boundary.

**Business domain decomposition** (Domain-Driven Design bounded contexts) splits by
capability:

```
order-service         →  owns order lifecycle end-to-end (API + logic + DB)
inventory-service     →  owns stock levels end-to-end
customer-service      →  owns customer profiles end-to-end
```

A single feature change is typically contained within one service. Teams own a service
end-to-end. Database schemas are private — no shared tables across service boundaries.

**Rule of thumb:** If two services must be deployed together every time a feature ships,
the decomposition boundary is wrong.

---

## Question 4 — Async Messaging vs Synchronous REST

**Choose synchronous REST when:**
- The caller needs the result to proceed (e.g., payment confirmation before redirecting).
- Latency SLA is tight and the callee is highly available.
- The operation is a query, not a command.

**Choose async messaging when:**
- The caller does not need an immediate result (fire and forget).
- The callee is a bottleneck or has unpredictable latency (queue absorbs spikes).
- The operation must be durable: the message must survive caller or callee restarts.
- Fan-out is needed: one event triggers multiple downstream consumers.
- The caller and callee have mismatched uptime requirements (decoupled availability).

**Example:** When an order is confirmed, the following happen asynchronously:
- Email service sends a confirmation email.
- Analytics service records the conversion event.
- Warehouse service schedules picking.

None of these need to complete before the API returns `201 Created` to the customer.
Coupling them synchronously would increase p99 latency by the sum of all three call
latencies and couple the Order Service's availability to all three.

---

## Question 5 — Write Amplification in LSM Trees

**LSM (Log-Structured Merge) tree** accepts all writes to an in-memory buffer (memtable)
and periodically flushes sorted immutable files (SSTables) to disk. Reads are fast; the
problem arises during **compaction**.

**Write amplification** = bytes written to disk / bytes of logical user data written.

Compaction merges multiple SSTables by re-reading and re-writing all their data to produce
a new, sorted, deduplicated file. A single user write of 1 KB may ultimately cause 10–30
KB of disk writes across compaction cycles at different levels.

**When it is acceptable:**
- Write throughput requirements are high and the workload is write-heavy (time-series,
  event logs, message queues). LSM accepts the amplification trade to linearise writes and
  avoid random-write penalty on spinning disks and even on SSDs (write endurance matters).
- Reads can tolerate Bloom filter overhead and multi-file lookups.

**When it is not acceptable:**
- Read-heavy workloads where B-tree's O(log n) direct lookup outperforms LSM's multi-SST
  merge reads.
- Storage with limited write endurance (embedded flash) where write amplification
  shortens device life unacceptably.

**Mitigation:** Tiered vs levelled compaction strategies trade off write amplification
against space amplification. RocksDB exposes both. Tune based on measured workload.

---

## Question 6 — Sharding by user_id Hash and Cross-Shard Queries

**Sharding by `user_id` hash** collocates all data for a single user on one shard. Queries
scoped to one user (e.g., `SELECT * FROM orders WHERE user_id = ?`) touch exactly one
shard — O(1) shard lookup.

**What breaks:** any query that is not scoped to a single `user_id`:
- `SELECT COUNT(*) FROM orders WHERE status = 'pending'`
- `SELECT * FROM orders ORDER BY created_at DESC LIMIT 10` (global leaderboard)
- `SELECT user_id, SUM(amount) FROM orders GROUP BY user_id ORDER BY SUM(amount) DESC`

These queries require a **scatter-gather**: send the query to every shard, receive N
partial result sets, merge and re-sort in the application layer. Cost grows linearly with
shard count. Correctness of `ORDER BY` + `LIMIT` requires fetching `limit * shard_count`
rows from each shard and re-sorting globally.

**Mitigations:**
- Maintain a separate analytics replica (denormalised, single node or columnar store like
  Redshift/BigQuery) fed by CDC from all shards.
- For specific global aggregates (total count, global leaderboard) maintain a separate
  materialised summary table that is updated asynchronously.
- Accept the scatter-gather cost for infrequent admin queries and document the trade-off.

---

## Question 7 — MVCC and Read-Committed Isolation

**MVCC (Multi-Version Concurrency Control)** stores multiple versions of each row, tagged
with the transaction ID that created or deleted them.

Under **read-committed** isolation, a reader sees the latest committed version of each row
as of the time that individual row is read. Crucially, the reader does not need to acquire
a shared lock on the row: it reads the most recent version whose creating transaction has
already committed.

```
Timeline:
T1 (writer): BEGIN, UPDATE orders SET status='confirmed' WHERE id=42  -- not yet committed
T2 (reader): SELECT * FROM orders WHERE id=42
  -- sees the OLD version (status='pending') because T1 is not yet committed
T1: COMMIT
T2: SELECT * FROM orders WHERE id=42  (a second read within the same transaction)
  -- now sees status='confirmed'  (read-committed allows this "non-repeatable read")
```

**Why no read locks:**
- Writers write a new version; they do not overwrite or lock the old version.
- Readers access the old version until the writer commits.
- Writers block other writers on the same row (write-write conflict) but never block
  readers, and readers never block writers.

**Cost:** row storage overhead from keeping multiple versions. Postgres uses a VACUUM
process to reclaim dead row versions (tuples). Long-running read transactions can
block VACUUM and cause table bloat.

---

## Question 8 — CAP Theorem: Partition Tolerance in Practice

**Partition tolerance** means the system continues to operate when **network partitions**
occur — i.e., when the network between nodes drops or delays messages such that some nodes
cannot communicate with others.

In practice, this is not optional for a distributed system deployed across multiple
machines or data centres: network partitions happen (hardware failure, misconfiguration,
packet loss, datacenter split-brain). You cannot simply choose "no partition tolerance"
because that would mean the system halts entirely whenever any two nodes cannot talk.

**What the trade-off actually means:**
When a partition occurs, you must choose for the isolated nodes:
- **CP (Consistency over Availability):** Refuse requests on the minority partition side
  rather than serve potentially stale data. The system returns errors or timeouts. Example:
  ZooKeeper, etcd — they block writes to the minority partition.
- **AP (Availability over Consistency):** Serve requests on both sides of the partition,
  accepting that they may diverge. Reconcile (merge) after the partition heals. Example:
  Cassandra, DynamoDB in eventual-consistency mode.

**CA is not a real choice for distributed systems:** A single-node database is CA, but
once you have multiple nodes, you have partition risk. "CA" means "works fine until a
partition, then breaks completely."

---

## Question 9 — Optimistic vs Pessimistic Locking

**Pessimistic locking:** Acquire a lock before reading the data, hold it until the
transaction completes. Other writers are blocked for the duration.

```sql
-- PostgreSQL: lock the row before reading
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Row is now locked; other transactions block on UPDATE or SELECT ... FOR UPDATE
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
```

- Good when: conflicts are frequent (high contention), holding incorrect state is
  expensive, and the critical section is short.
- Bad when: lock hold time is long (locks under user-think time), or the system is
  read-heavy (unnecessary contention).

**Optimistic locking:** No lock on read. Include a version number (or ETag) in the read.
On write, check that the version has not changed. If it has, abort and retry.

```python
# Application-level optimistic lock using a version column
row = db.query("SELECT id, balance, version FROM accounts WHERE id = ?", account_id)
new_balance = row.balance - amount
updated = db.execute(
    "UPDATE accounts SET balance = ?, version = version + 1 "
    "WHERE id = ? AND version = ?",
    new_balance, account_id, row.version
)
if updated.rowcount == 0:
    raise ConcurrentModificationError("Account was modified concurrently; retry.")
```

- Good when: conflicts are rare (low contention), read-heavy, or transactions span
  external systems where holding a DB lock is not possible.
- Bad when: high contention causes many retries, turning into a spin-wait.

**Rule:** Pessimistic for high contention or expensive retries. Optimistic for low
contention, long transactions, or distributed scenarios where DB locks are unavailable.

---

## Question 10 — 200 OK with Error in Body

```json
HTTP/1.1 200 OK
Content-Type: application/json

{ "success": false, "error": "Insufficient funds", "code": "INSUFFICIENT_FUNDS" }
```

**What is wrong:**
1. **HTTP status code is a lie.** Middleware, proxies, circuit breakers, and load
   balancers all use HTTP status codes to determine success or failure. They see `200 OK`
   and classify the response as successful. The error is invisible to infrastructure.
2. **Monitoring breaks.** Alerting on 4xx/5xx error rates will miss these failures
   entirely. Error budgets will appear healthier than they are.
3. **Client code is harder to write.** Clients must parse the body to detect failure,
   cannot rely on status code branching, and must handle both this envelope format and
   standard HTTP errors from gateways.
4. **Retries behave incorrectly.** Retry-on-failure logic keyed on 5xx will not retry
   these failures even when it should.
5. **RFC 7807 / RFC 9457 exists for this purpose.** Return the appropriate HTTP status
   code (`400`, `402`, `409`, etc.) with `Content-Type: application/problem+json`.

**Correct approach:**
```json
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/problem+json

{
  "type": "https://api.example.com/problems/insufficient-funds",
  "title": "Insufficient Funds",
  "status": 422,
  "detail": "Account balance 50.00 is less than requested transfer amount 100.00.",
  "instance": "/transfers/txn-abc123"
}
```

---

## Coding Challenge — POST /api/v1/transfers

Full implementation with idempotency, atomic in-memory debit/credit, RFC 7807 error on
insufficient funds.

```python
# src/transfer_service/main.py
from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from typing import Annotated

from fastapi import FastAPI, Header, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Transfer Service", version="1.0.0")

PROBLEM_JSON = "application/problem+json"

# ---------------------------------------------------------------------------
# In-memory state (replace with DB in production)
# ---------------------------------------------------------------------------

_lock = threading.Lock()

_accounts: dict[str, float] = {
    "acc-alice": 1000.00,
    "acc-bob":     500.00,
    "acc-charlie":  50.00,
}

# idempotency_key_hash -> recorded response dict
_idempotency_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TransferRequest(BaseModel):
    from_account: str = Field(..., min_length=1)
    to_account: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")


class TransferResponse(BaseModel):
    transfer_id: str
    from_account: str
    to_account: str
    amount: float
    currency: str
    status: str


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def problem_response(
    *,
    problem_type: str,
    title: str,
    http_status: int,
    detail: str,
    instance: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={
            "type": problem_type,
            "title": title,
            "status": http_status,
            "detail": detail,
            "instance": instance,
        },
        headers={"Content-Type": PROBLEM_JSON},
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/transfers",
    response_model=TransferResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_transfer(
    body: TransferRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> TransferResponse | JSONResponse:
    key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
    instance = f"/api/v1/transfers/{idempotency_key}"

    with _lock:
        # --- Idempotency replay ---
        if key_hash in _idempotency_store:
            logger.info("Idempotent replay key=%s", idempotency_key)
            return _idempotency_store[key_hash]  # type: ignore[return-value]

        # --- Validate accounts exist ---
        if body.from_account not in _accounts:
            return problem_response(
                problem_type="https://api.example.com/problems/account-not-found",
                title="Account Not Found",
                http_status=404,
                detail=f"Source account '{body.from_account}' does not exist.",
                instance=instance,
            )
        if body.to_account not in _accounts:
            return problem_response(
                problem_type="https://api.example.com/problems/account-not-found",
                title="Account Not Found",
                http_status=404,
                detail=f"Destination account '{body.to_account}' does not exist.",
                instance=instance,
            )
        if body.from_account == body.to_account:
            return problem_response(
                problem_type="https://api.example.com/problems/invalid-transfer",
                title="Invalid Transfer",
                http_status=422,
                detail="Source and destination accounts must be different.",
                instance=instance,
            )

        # --- Check balance ---
        current_balance = _accounts[body.from_account]
        if current_balance < body.amount:
            return problem_response(
                problem_type="https://api.example.com/problems/insufficient-funds",
                title="Insufficient Funds",
                http_status=422,
                detail=(
                    f"Account '{body.from_account}' has balance {current_balance:.2f} "
                    f"{body.currency}, which is less than the requested transfer "
                    f"amount {body.amount:.2f} {body.currency}."
                ),
                instance=instance,
            )

        # --- Atomic debit + credit ---
        _accounts[body.from_account] -= body.amount
        _accounts[body.to_account] += body.amount

        transfer_id = f"txn-{uuid.uuid4().hex[:16]}"
        result = TransferResponse(
            transfer_id=transfer_id,
            from_account=body.from_account,
            to_account=body.to_account,
            amount=body.amount,
            currency=body.currency,
            status="completed",
        )

        # --- Record for idempotency ---
        _idempotency_store[key_hash] = result

        logger.info(
            "Transfer completed id=%s from=%s to=%s amount=%.2f",
            transfer_id, body.from_account, body.to_account, body.amount,
        )
        return result
```

```python
# tests/test_transfers.py
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.transfer_service.main import app, _accounts, _idempotency_store

client = TestClient(app, raise_server_exceptions=False)


def _key() -> str:
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory state before each test."""
    _accounts.clear()
    _accounts.update({"acc-alice": 1000.00, "acc-bob": 500.00, "acc-charlie": 50.00})
    _idempotency_store.clear()
    yield


def test_successful_transfer():
    resp = client.post(
        "/api/v1/transfers",
        json={"from_account": "acc-alice", "to_account": "acc-bob", "amount": 200.00},
        headers={"Idempotency-Key": _key()},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "completed"
    assert _accounts["acc-alice"] == pytest.approx(800.00)
    assert _accounts["acc-bob"] == pytest.approx(700.00)


def test_insufficient_funds_returns_422_problem_json():
    resp = client.post(
        "/api/v1/transfers",
        json={"from_account": "acc-charlie", "to_account": "acc-alice", "amount": 100.00},
        headers={"Idempotency-Key": _key()},
    )
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "https://api.example.com/problems/insufficient-funds"
    assert body["status"] == 422
    # Balances unchanged
    assert _accounts["acc-charlie"] == pytest.approx(50.00)


def test_idempotency_replay_does_not_double_charge():
    key = _key()
    payload = {"from_account": "acc-alice", "to_account": "acc-bob", "amount": 100.00}
    headers = {"Idempotency-Key": key}

    resp1 = client.post("/api/v1/transfers", json=payload, headers=headers)
    resp2 = client.post("/api/v1/transfers", json=payload, headers=headers)

    assert resp1.status_code == 201
    assert resp2.status_code == 201
    assert resp1.json()["transfer_id"] == resp2.json()["transfer_id"]
    # Only one debit
    assert _accounts["acc-alice"] == pytest.approx(900.00)


def test_source_account_not_found_returns_404():
    resp = client.post(
        "/api/v1/transfers",
        json={"from_account": "acc-ghost", "to_account": "acc-alice", "amount": 10.00},
        headers={"Idempotency-Key": _key()},
    )
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"


def test_same_account_transfer_rejected():
    resp = client.post(
        "/api/v1/transfers",
        json={"from_account": "acc-alice", "to_account": "acc-alice", "amount": 10.00},
        headers={"Idempotency-Key": _key()},
    )
    assert resp.status_code == 422
```
