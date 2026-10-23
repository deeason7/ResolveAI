# API reference

Summary of every endpoint. Interactive schemas live at
`http://localhost:8010/api/docs` (OpenAPI) — this page is the map, that page is
the territory. All routes are prefixed `/api/v1`.

**Auth model:** register/login return a short-lived bearer access token and
set a rotating httpOnly refresh cookie. Everything below except
register/login/refresh/health requires `Authorization: Bearer <token>`.
Reads require an account because the corpus contains consumer narratives;
bulk import additionally requires the `admin` role.

## Auth — `/auth`

| Method | Path | Returns | Notes |
|---|---|---|---|
| POST | `/register` | 201 + access token | Sets refresh cookie |
| POST | `/login` | 200 + access token | Sets refresh cookie |
| POST | `/refresh` | 200 + new access token | Rotates the refresh cookie |
| POST | `/logout` | 204 | Revokes the refresh token (Redis blocklist) |
| GET | `/me` | 200 user profile | |

## Complaints — `/complaints`

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/` | Paginated list | Filters: `status`, `product`, `company`, `state`; `limit`/`offset` |
| GET | `/queue` | Priority-sorted triage page | Actionable statuses by default; filters incl. `urgency_min/max` (422 if min>max); 200-char previews |
| GET | `/{id}` | Full complaint | 404 unknown id |
| GET | `/{id}/similar` | Top-K nearest narratives | Vector search; self excluded; optional `product` filter; `limit` 1–10; **503 if the similarity subsystem is down** (page degrades, complaint still loads) |
| POST | `/` | 201 + complaint | Commits complaint + classification job atomically; optional `Idempotency-Key` |
| POST | `/bulk-import` | Import counts | **Admin.** Server-local CSV path, sandboxed to mounted data dirs; idempotent |

## Resolutions — `/resolutions`

| Method | Path | Returns | Notes |
|---|---|---|---|
| POST | `/{complaint_id}/generate` | **202** job identity/status | Optional `Idempotency-Key`; 409 if already queued/running, response approved or no current classification |
| GET | `/{complaint_id}` | Latest draft | Draft text, per-layer guardrail violations, reasoning steps, version |
| GET | `/{complaint_id}/revisions` | All versions | |
| POST | `/{complaint_id}/approve` | 200 outcome | Body: `resolution_id`, `expected_case_version`; saved draft must pass guardrails |
| POST | `/{complaint_id}/reject` | **202** job identity/status | Body: `resolution_id`, `expected_case_version`, `feedback` (10–2000 chars) |

The latest-draft and revision responses include `case_version`, `input_revision`,
`legacy_provenance` and `review_action`. Send the exact displayed draft's `id`
and `case_version` as `resolution_id` and `expected_case_version`. A stale case,
older draft or cross-case ID returns an actionable 409. Repeating an exact
rejection returns its original job; a different decision on reviewed text conflicts. Missing identity returns 422; there is no implicit latest-draft approval.
Reload and read the saved response before submitting another decision.

Approval changes the case to `response_approved`: the exact saved response was
approved, but delivery and customer outcome are unrecorded. Legacy `resolved`
rows retain that status with `legacy_approval_outcome_unknown=true`. They do not
establish delivery or resolution. Every new decision records an immutable review
event with the reviewer, draft ID, versions, decision and text hash.

Generation and rejection commit the case transition, review evidence (for
rejection), and job together. Capacity rejection returns 503 without accepting
changes. Intake and generation accept an `Idempotency-Key` of 1–128 characters,
scoped to the actor and route. Repeating the same content returns its original
case/job; changing content under that key returns 409. Rejection additionally
uses the exact reviewer/draft/case-version identity when no key is supplied.
A replay returns the original job's current status without repeating the review.
See [durable jobs](jobs.md) and [review concurrency](review_concurrency.md).

## Workspace — `/workspace`

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/board` | Case counts and job health | Authenticated; per-kind queued/running/failed counts and oldest pending age |
| POST | `/enqueue/classification` | Enqueued count and kind | **Staff.** `limit` 1–500; skips existing intents |
| POST | `/enqueue/resolution` | Enqueued count and kind | **Staff.** `limit` 1–500; current classified escalations only |

Batches commit all selected transitions and jobs together. Capacity rejection
returns 503 without partial acceptance. A new batch request can select another
batch of eligible cases.

## Jobs — `/jobs`

All job inspection and retry endpoints require an analyst or admin role.
Responses omit payloads, request keys and lease credentials.

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/` | Paginated jobs | `status` defaults to `failed`; optional `kind`; `limit` 1–100 and `offset` |
| GET | `/{job_id}` | Job state and safe failure category | Includes attempts, retry generation and progress timestamps; 404 unknown job |
| POST | `/{job_id}/retry` | Current job | Body: displayed `expected_attempts`, `expected_retry_generation`; stale/ineligible retry returns 409 |

Retry grants the same failed intent another bounded attempt budget. Repeating
that exact retry request returns the accepted generation without multiplying work.

## Analytics — `/analytics`

| Method | Path | Returns |
|---|---|---|
| GET | `/sentiment/trends?days=N` | Daily (day, sentiment, count) buckets; `unclassified` is an explicit bucket |
| GET | `/products/breakdown` | Per-product totals + urgency histogram counts |
| GET | `/companies/risk?limit=N` | Top-N by volume with severity columns (avg urgency, urgent count, extreme count) |

## Knowledge graph — `/graph`

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/company/{name}` | Company profile | Totals, risk score, linked violations, product mix; 404 unknown |
| GET | `/product/{name}/regulations` | Regulations for a product | Optional `issue` narrowing; empty list is a valid 200 |
| GET | `/explore?node_id=&depth=` | Bounded subgraph `{nodes, edges}` | Matches name or id; `depth` capped at 3 (dense graph — fan-out is real) |

## LLMOps — `/llmops`

| Method | Path | Returns |
|---|---|---|
| GET | `/costs?days=N` | Daily calls/tokens/spend per provider + totals |
| GET | `/latency?days=N` | p50/p95/avg/max per operation |
| GET | `/routing?days=N` | Calls per (provider, fallback-flag) — `none` = deterministic fail-closed |
| GET | `/drift?days=N` | Classifier output mix by classify-call date |
| GET | `/guardrails?layer=&limit=` | Flattened violation log, newest first; `total_violations` counts beyond the page |

## Health

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Service health summary (used by the container healthcheck) |

## Conventions worth knowing

- **List endpoints paginate** with `limit`/`offset` and return `total`.
  K-nearest endpoints (`/similar`) return bare `items` — similarity search
  doesn't paginate.
- **409 means "the API is being honest"**: generate/approve conflicts state
  their reason (case changed, response approved, no current classification) rather
  than failing generically.
- **202 means async**: generate/reject queue work for the resolution
  worker — poll the resolution endpoints or watch the dashboard.
- Validation errors are FastAPI-standard 422s with field-level detail.
- **Rate limits**: 200/min globally, tightened to 20/min on `/auth/login` and
  `/auth/register` (each attempt costs a bcrypt verify) and 10/min on the
  workspace enqueue routes. Over the line is a 429.
- **Login is deliberately uninformative**: an unknown address and a wrong
  password return the same 401, and take the same time to do it.
