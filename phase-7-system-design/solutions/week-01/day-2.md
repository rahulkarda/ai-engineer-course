# Solutions — Day 2: Functional vs Non-Functional Requirements

---

## Exercise 1 — File storage service (Dropbox)

**Functional requirements:**
1. Users can upload files up to 5 GB.
2. Users can download files by name or by sharing a link.
3. Files are automatically synced across all devices logged in with the same account.
4. Users can organise files into folders.
5. Deleted files are moved to a trash folder and can be restored within 30 days.

**Non-functional requirements + architectural implications:**

| NFR | Implication |
|---|---|
| Availability: 99.99% | Multi-region deployment; active-active or active-passive with automatic failover |
| Upload bandwidth: p99 < 30 s for 1 GB on 100 Mbps | Chunked multi-part upload; CDN upload endpoints at edge |
| Sync latency: changes visible on other devices within 10 s | WebSocket or server-sent events for change notifications; not polling |
| Storage durability: 11 nines (no data loss) | Replication factor ≥ 3; cross-region backup |
| Scale: 50 M DAU, each with avg 5 GB stored | 250 PB total storage; object store (S3 equivalent), not block storage |

---

## Exercise 2 — Clarifying questions for "Design Twitter"

1. **"Is this read-heavy or write-heavy from a latency perspective — i.e., should posting be instantaneous, or can it tolerate 1–2 s while the feed is fast?"**
   Impact: determines whether fan-out-on-write (materialised feeds) or fan-out-on-read is appropriate. If posting must feel instant and feeds can lag slightly, fan-out-on-write with async workers.

2. **"Are we designing the public timeline (all tweets), the home feed (people you follow), or both?"**
   Impact: the home feed requires a follow graph traversal; the public timeline is a simple reverse-chronological sorted set. Different storage models.

3. **"What's the scale — specifically, how many followers can a celebrity account have?"**
   Impact: an account with 100 M followers changes the fan-out problem completely. Need a special path for "megastar" accounts (pull model for celebrity tweets; push for normal users).

---

## Exercise 3 — Classification

| Statement | Type |
|---|---|
| "Users can follow other users." | **Functional** |
| "Feed must load in under 300 ms for 95% of requests." | **Non-Functional** (latency SLO) |
| "Support 10 k new user registrations per day." | **Non-Functional** (scale/capacity) |
| "Posts can include images up to 20 MB." | Borderline — stated as a capability (F), but also implies a constraint (NF). Accept either; note both aspects. |
| "System must be available in EU and US with data residency." | **Non-Functional** (availability + compliance) |

---

## Exercise 4 — Handling ambiguous requirement in an interview

The exact question to ask:

> "When you say 'notifications should be delivered quickly' — can we define that more precisely? For example, is the target that a user receives a push notification within 5 seconds of the event? Or is near-real-time — within 60 seconds — acceptable? That will determine whether I need WebSockets and a push pipeline, or whether a polling-based approach with a queue is sufficient."

Key elements: (1) acknowledge the ambiguity explicitly, (2) offer two concrete options with different technical implications, (3) let the interviewer choose. Do not decide for them without asking.
