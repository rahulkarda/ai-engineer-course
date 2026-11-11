# Solutions — Week 08 · Day 3

## Exercise 1

**Service decomposition for a monolithic e-commerce app**

Starting domains in the monolith: Users, Products, Orders, Payments, Notifications, Inventory.

**Proposed services and bounded contexts:**

| Service | Bounded Context | Core Responsibility |
|---------|----------------|---------------------|
| **Identity & Auth** | IAM | Registration, login, JWT issuance, roles, password reset |
| **Catalog** | Product Discovery | Product definitions, pricing, search index, category taxonomy |
| **Order Management** | Order Lifecycle | Order creation, status machine (pending → confirmed → shipped → delivered → returned) |
| **Payment** | Financial Transactions | Charge, refund, idempotent payment processing, ledger |
| **Inventory** | Stock Control | Stock levels, reservations, replenishment, warehouse locations |
| **Notifications** | Communication | Email/SMS/push delivery, template rendering, delivery receipts |

**Boundary justification using DDD bounded contexts:**

1. **Identity vs Catalog vs Orders** — Different ubiquitous languages. "User" means an authenticated principal in IAM, but means "buyer with shipping address" in Order Management. Merging them forces a shared `User` model that grows to serve every context, creating tight coupling.

2. **Order Management vs Payment** — Order state and financial state have different failure modes. An order can exist in a "payment pending" state without a completed charge. Keeping them separate allows Payment to be replaced (e.g., swap Stripe for Adyen) without touching order logic. The seam is an event: `PaymentCaptured(order_id, amount)`.

3. **Order Management vs Inventory** — Inventory needs to respond to stock queries from Catalog (browsing) and hold reservations for Orders (checkout). Its change rate and scaling profile differ from Order state transitions. The seam: Order emits `OrderPlaced`, Inventory reacts and confirms or rejects via `StockReserved` / `StockUnavailable`.

4. **Notifications as leaf service** — Notifications has no domain logic of its own; it reacts to events from all other services. It never calls back into Order or Payment. This makes it a safe first extraction target — removing it from the monolith leaves zero inbound dependencies to untangle.

5. **Catalog vs Inventory** — "Product" means different things: Catalog owns the marketing description and pricing; Inventory owns how many units exist in which warehouse. Merging them creates a 1:many mismatch (one product SKU, many warehouse locations).

---

## Exercise 2

**Data ownership and seams for each service**

| Service | Data It Owns | Integration Seam |
|---------|-------------|-----------------|
| **Identity & Auth** | `users` table (credentials, profile, roles) | Issues JWTs; others validate tokens without calling back |
| **Catalog** | `products`, `categories`, `prices` | Publishes `ProductUpdated` event; consumed by search index and Order service for price snapshot |
| **Order Management** | `orders`, `order_line_items`, `order_status_history` | Emits `OrderPlaced`, `OrderCancelled`; listens to `PaymentCaptured`, `StockReserved` |
| **Payment** | `payment_intents`, `charges`, `refunds`, `ledger_entries` | Listens to `OrderPlaced`; emits `PaymentCaptured`, `PaymentFailed`, `RefundIssued` |
| **Inventory** | `stock_levels`, `stock_reservations`, `warehouses` | Listens to `OrderPlaced`; emits `StockReserved`, `StockUnavailable`, `LowStockAlert` |
| **Notifications** | `notification_log`, `delivery_receipts`, `templates` | Listens to events from all services; never owns business data |

**Key seam decisions:**

- **Order stores a price snapshot**, not a foreign key into Catalog. If Catalog changes a price, existing order line items are unaffected. This is the "price at time of purchase" invariant — critical for financial auditing.
- **Identity is not replicated** into Order or Payment. Services validate the JWT locally using a shared public key. Only the `user_id` claim travels in service calls; the full profile is fetched from Identity only when needed (rare).
- **Inventory reservations have a TTL.** An `OrderPlaced` event triggers a stock reservation. If Payment does not confirm within N minutes, Inventory releases the hold automatically. This prevents ghost reservations without requiring a distributed transaction.
- **Notifications owns no business state.** It records that a notification was sent (audit log), but it does not own order status or payment status. If it goes down, orders and payments continue without interruption.

---

## Exercise 3

**Strangler Fig migration — extracting Notifications from the monolith**

The Strangler Fig pattern (Martin Fowler) works by routing traffic incrementally to the new service while the monolith continues running, then removing the monolith code once the new service handles 100% of the load.

**Step-by-step for Notifications:**

**Step 1 — Identify all notification triggers in the monolith.**
Grep for every `send_email()`, `send_sms()`, `push_notify()` call. Map each to its trigger: "order confirmed", "payment failed", "password reset", etc. This is the interface contract you must replicate.

**Step 2 — Build the new Notifications service.**
Implement a standalone service that:
- Consumes events from a message broker (e.g., Kafka or RabbitMQ).
- Renders templates and calls the email/SMS provider APIs.
- Logs delivery receipts to its own database.
Deploy it. Do not yet switch traffic to it.

**Step 3 — Introduce an event bus in the monolith.**
Modify the monolith so that instead of calling `send_email()` directly, it publishes an event: `OrderConfirmed(order_id, user_id, email, items)`. Initially, the monolith both publishes the event AND still calls `send_email()` directly (dual-write). This validates that events are correctly shaped before cutting over.

**Step 4 — Enable the new service to consume events.**
Point the Notifications service at the event bus. Run both paths in parallel (monolith still sends, new service also sends). Compare delivery logs to verify parity.

**Step 5 — Disable direct sends in the monolith.**
Once parity is confirmed, remove the `send_email()` / `send_sms()` calls from the monolith. The monolith now only publishes events. The new service is the sole notification sender.

**Step 6 — Remove dead code from the monolith.**
Delete the notification modules, email client config, and template files from the monolith codebase. Remove the dependency on the email provider SDK from the monolith's dependency list.

**Step 7 — Verify and monitor.**
Run for one release cycle. Monitor delivery rates, bounce rates, and error logs in the new service. Rollback plan: re-enable the dual-write path in the monolith if the new service degrades.

**Why Strangler Fig works:** At no point is there a "big-bang" cutover. The monolith is never fully offline. If the new service misbehaves, the fallback is one flag flip away. Risk is proportional to the migration step, not to the total size of the system.

---

## Exercise 4

**Microservices recommendation for a 3-engineer, 500 DAU startup**

**Recommendation: Do not adopt microservices. Build a well-structured monolith.**

**Reasoning:**

**Operational overhead exceeds team capacity.** Microservices require service discovery, distributed tracing, a message broker, per-service CI/CD pipelines, container orchestration (Kubernetes or equivalent), and on-call runbooks for each service boundary. With 3 engineers, every hour spent on infrastructure is an hour not spent on product. At 500 DAU, a single well-tuned Postgres instance and a monolith on one Render/Railway/Fly.io deployment will handle the load without any of that overhead.

**Distributed systems introduce failure modes you don't need yet.** Network partitions, partial failures, eventual consistency, and idempotency are hard problems. A startup's value is in validating the product, not in solving distributed systems problems before they are necessary.

**Organisational fit matters.** Conway's Law states that system architecture mirrors communication structure. With 3 engineers, you have one team — one bounded context. Service boundaries that map to team boundaries provide value at 10+ teams. With one team, boundaries create friction without the coordination benefit.

**The right time to split** is when:
- A specific module has a fundamentally different scaling requirement (e.g., image processing needs GPU machines; the rest runs on cheap VMs).
- Teams are growing and stepping on each other in the monolith.
- Deploy velocity is blocked by coupling (one change requires coordinating 5 modules).

**What to do instead:** Build a modular monolith. Enforce module boundaries in code (separate packages, no circular imports, clear interfaces between domains). Use the same bounded contexts from Exercise 1 as internal Python packages (`catalog`, `orders`, `payments`, etc.) with explicit API contracts between them. When the time comes to extract a service, the code is already structured for it — the Strangler Fig migration from Exercise 3 becomes straightforward.

**Rule of thumb:** "Microservices are a solution to organisational problems, not a solution to technical problems." — Sam Newman. At 500 DAU and 3 engineers, you have neither problem.
