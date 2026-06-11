# API reference

Summary of every endpoint. Interactive schemas live at
`http://localhost:8010/docs` (OpenAPI) — this page is the map, that page is
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
| POST | `/logout` | 200 | Revokes the refresh token (Redis blocklist) |
| GET | `/me` | 200 user profile | |

## Complaints — `/complaints`

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/` | Paginated list | Filters: `status`, `product`, `company`, `state`; `limit`/`offset` |
| GET | `/queue` | Priority-sorted triage page | Actionable statuses by default; filters incl. `urgency_min/max` (422 if min>max); 200-char previews |
| GET | `/{id}` | Full complaint | 404 unknown id |
| GET | `/{id}/similar` | Top-K nearest narratives | Vector search; self excluded; optional `product` filter; `limit` 1–10; **503 if the similarity subsystem is down** (page degrades, complaint still loads) |
| POST | `/` | 201 + complaint | Queues classification (best-effort enqueue; `pending` status is the durable signal) |
| POST | `/bulk-import` | Import counts | **Admin.** Server-local CSV path, sandboxed to mounted data dirs; idempotent |

## Resolutions — `/resolutions`

| Method | Path | Returns | Notes |
|---|---|---|---|
| POST | `/{complaint_id}/generate` | **202** queued | 409 if already running / already resolved / not classified yet |
| GET | `/{complaint_id}` | Latest draft | Draft text, per-layer guardrail violations, reasoning steps, version |
| GET | `/{complaint_id}/revisions` | All versions | |
| POST | `/{complaint_id}/approve` | 200 outcome | Gated to drafts that passed guardrails; 409 if already resolved |
| POST | `/{complaint_id}/reject` | **202** queued | Body: `feedback` (min 10 chars) — fed into the regeneration prompt |

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
  their reason (already running, already resolved, not classified) rather
  than failing generically.
- **202 means async**: generate/reject queue work for the resolution
  worker — poll the resolution endpoints or watch the dashboard.
- Validation errors are FastAPI-standard 422s with field-level detail.
