# Architecture

This describes the implementation and its known limits. The initial review was
2026-10-17; access control, PostgreSQL verification and backup handling were revised
on 2026-10-18. Proposed queue/review changes are not existing guarantees.

## Runtime topology

| Component | Current role |
|---|---|
| FastAPI | API, authentication, validation and rate limits |
| PostgreSQL | Complaints, users, draft resolutions, labels, telemetry and audit records |
| Redis | Stream queues and refresh-token revocation |
| Qdrant | 384-dimensional narrative embeddings for similarity search |
| Neo4j with APOC | Company/product/issue relationships and seeded reference material |
| Ollama | Optional local inference when a compatible model is installed |
| Streamlit | Reviewer interface; Python makes API requests on the server |

Development Compose includes these services. Production Compose starts the workers
as services and makes Ollama opt-in. The managed hosting configuration runs the API
and both workers in one container under supervisord, with external datastores.
Both production environment examples set `LLM_SKIP_LOCAL=true`: their default
classification path is cloud inference, not the fine-tuned local model.

## State and queue boundary

PostgreSQL is authoritative business storage. Qdrant and Neo4j are derived indexes;
rebuild scripts exist, but rebuild completeness and duration need verification.
Redis Streams use consumer groups and pending-message reclamation. These mechanisms
allow redelivery; they do not make database updates and queue publication atomic.

```mermaid
sequenceDiagram
    participant U as Client
    participant API as FastAPI
    participant PG as PostgreSQL
    participant R as Redis
    participant W as Classification worker
    participant L as Configured model provider
    U->>API: Submit complaint
    API->>PG: Insert and flush pending complaint
    API->>R: Publish classification message
    API->>PG: Dependency commits transaction
    API-->>U: Complaint ID
    R->>W: Read or reclaim message
    W->>L: Classify outside database transaction
    W->>PG: Persist classification and telemetry
    W->>R: Acknowledge message
```

At the reviewed baseline, publishing precedes the request transaction's commit.
A worker may race the insert, and a database rollback can leave a queued message.
A lost enqueue can leave pending work without a durable job record. Replaying a
message can also overwrite later case state. There is no verified automatic
reconciliation guarantee. Transactional job persistence and stale-result fencing
are required improvements, not properties of the current stream implementation.

Synchronous model calls are offloaded with `asyncio.to_thread`. The resolution
worker records `agent_triggered` in an initial transaction, then persists the
result and logs in another. The database lock does not span model inference.

## Classification and drafting

The classifier uses an installed local model unless local inference is skipped;
otherwise the configured cloud provider is used. On provider failure, the reviewed
classifier returns a deterministic negative/dispute-resolution/urgency-3 result
with `provider="none"` and `succeeded=false`. The worker can store it as a normal
classification. This is not a maximum-severity escalation or reliable manual-review
state. Treat that row as unavailable model output, not a prediction.

The resolution agent uses a bounded flow: gather context, draft, validate,
regenerate with feedback, then return a draft or escalate after the retry limit.
Context gathering tolerates tool errors, so a completed draft can lack a source.
Rejection feedback enters the regeneration prompt. The stored trace describes
tool calls and workflow steps; it is not access to a model's private reasoning.

## Checks and human review

| Check | What it checks | What it does not establish |
|---|---|---|
| Structure | Length and acknowledgment/next-step keywords | A correct or useful response |
| Content patterns | Selected prohibited wording and numeric PII patterns in output | Complete PII removal or protection of cloud-bound inputs |
| Citation matching | Title substrings in a separate citation list against retrieved titles | Correct law, supported prose, jurisdiction or applicability |
| Tone judge | Model scores for empathy, professionalism and actionability | Factual or legal accuracy |

The tone judge runs only after deterministic checks pass. Unreachable judging
produces a violation, but the current UI can show an unrun layer as passed.
The agent constructor has a pass-through validator default; production workers
explicitly inject the real engine. A completed check is a limited signal for
human review, not a compliance certification.

Approval currently chooses a draft by complaint rather than requiring the exact
visible draft ID and version. Editing the displayed text does not save a revision.
The resulting `resolved` status means approval, not delivery or a verified customer
outcome. See [product scope](product_scope.md) for the intended outcome boundary.

## Authentication and API conventions

JWT access tokens travel in the Authorization header. Refresh tokens use an
httpOnly cookie in Streamlit's per-session HTTP client; the browser does not call
the API directly. Database role checks, not hidden controls, must authorize writes.
Public registration is disabled by default outside development/test; when enabled
it creates viewers. The configured demo identity is rejected if it is privileged.
Staff provisioning is an explicit administrative operation; see [access](access.md).
Token rotation and transport findings remain separate from role enforcement.

Wire contracts use Pydantic schemas separately from SQLModel rows. Queue endpoints
filter/order/page in SQL and send narrative previews; detail endpoints load the
full case. Several charts use database aggregates. Other observability endpoints
load unbounded rows into Python, and the analytics page makes serial graph calls;
these need workload measurements and query limits before scale claims.

## Telemetry and verification

Workers persist model/provider/token/latency/cost fields with some business writes.
This is not a complete provider-call ledger: failed attempts, historical labels,
unknown pricing and zero-duration fallback records limit interpretation. Recorded
cost is an estimate for logged calls, not the cost of the corpus or total operation.

The 2026-10-17 baseline verification at commit `0e75774` passed 543 backend tests
with 11 warnings. These use SQLite and injected provider/store/Redis fakes. They do
not prove PostgreSQL migrations, transaction races, deployment readiness or browser
behavior. No Streamlit AppTest suite was present at that baseline. Use the current
test output for later revisions rather than treating this dated count as a gate.

## Recovery and database checks

Complete version 2 exports include all application tables in one PostgreSQL
read-only snapshot. Restore requires a matching migrated schema and empty target,
verifies files and relationships, and commits all rows together. See
[backup recovery](backup_recovery.md). Legacy partial exports remain incomplete.

The current [test foundation](testing.md) adds PostgreSQL migration, constraint,
transaction-visibility, concurrency and default-restore checks. Client contract
tests cover demo access behavior; there is still no browser/AppTest coverage.
