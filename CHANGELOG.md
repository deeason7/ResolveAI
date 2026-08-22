# Changelog

All notable changes to ResolveAI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut from `main`; pushing a `v*` tag runs the full lint/test/image
gate and ships the tagged commit to the hosted demo.

## [1.4.1] — 2026-08-21

### Fixed

- **Rate limits and audit-log IPs now identify the caller, not the proxy.**
  uvicorn only honours `X-Forwarded-For` when the immediate peer is trusted,
  and that defaults to `127.0.0.1` alone — so behind the hosted platform's
  ingress every caller collapsed into a single identity. The rate limiter keyed
  on the proxy, which made the 20/min limit on the credential endpoints a
  shared bucket rather than per-client protection, and `audit_logs.ip_hash`
  recorded the ingress node instead of the client. The image now trusts a
  private-range peer, which is the only thing that can open a connection to it.
  Deliberately a CIDR and not `*`: uvicorn walks the forwarded chain in reverse
  to the first untrusted hop, but `*` short-circuits to the end a caller can
  inject. Three tests pin that distinction.

## [1.4.0] — 2026-08-21

Security and operability. No new user-facing surface — this release closes two
real defects found by auditing the running system, and makes the deployment
path automatic.

### Security

- **Closed a user-enumeration oracle on login.** Credential checking
  short-circuited when no account matched the submitted email, so a miss
  returned without ever running bcrypt. The gap was measurable — roughly
  184 ms for a registered address against ~0.1 µs for an unregistered one —
  which is trivially readable over a network and turns the login endpoint into
  a membership test for any email. Both branches now perform the same work by
  verifying against a dummy hash when no user is found.
- **Rate-limited the credential endpoints** to 20/min. Each attempt buys a
  bcrypt verification, so the general default let a small number of clients
  saturate CPU on a modest host.
- **Client IPs in the audit log are now keyed (HMAC-SHA256).** An unkeyed
  digest over a 32-bit address space is one precomputed table away from
  plaintext.
- **Refresh-token revocation keys on a token digest** rather than the raw JWT,
  with a dual read so tokens revoked under the previous scheme stay revoked.
- **Broadened liability detection in the resolution guardrails.** The pattern
  required the noun to sit immediately after the verb, so `admits liability`
  was caught while `admits full legal liability` was not, and
  `accepts responsibility` was never matched at all — the exact phrasings a
  drafting model reaches for. Qualifiers between verb and noun are now allowed
  and the noun set covers responsibility and blame.

### Added

- **Tagged releases deploy automatically.** A `v*` tag runs the same lint,
  test and image-build gate every branch gets, then promotes the tagged commit
  to the hosted Space. Merging to `main` deploys nothing — the tag is the
  commitment.
- **State backup and restore** (`scripts/backup_state.py`). Scope is decided by
  derivability: the complaint corpus is re-ingestable from the CFPB's public
  bulk download and the vector and graph stores are rebuilt from it, so only
  application-generated rows — accounts, resolutions, inference logs, the audit
  trail — are archived. Restore is idempotent and refuses to run when the
  backup's schema revision doesn't match the database.
- **Retention cap on the Redis work streams.** Acknowledging a message clears
  the consumer group's pending list but leaves the entry in the stream, so both
  queues grew for the life of a deployment. Producers now write with an
  approximate `MAXLEN`, tunable via `STREAM_MAXLEN`.
- **Prompt-injection test coverage** asserting that complaint narratives reach
  only the user turn, that the system prompt has no interpolation seam, and
  that the output guardrails reject an unsafe draft even on the assumption the
  model was fully steered.

### Fixed

- A signed token carrying a non-UUID subject returned 500 instead of 401.
- `refresh` and `logout` now use the pooled Redis client instead of opening a
  connection per request.
- Declared the `sqlalchemy[asyncio]` extra. SQLAlchemy gates its greenlet
  dependency behind a platform list that includes `aarch64` but not the
  `arm64` reported by Apple Silicon, so a clean resolve there produced an
  environment that failed at the first async query.
- Moved the CI workflows off the deprecated Node 20 action runtime.
- `scripts/` had no ruff configuration and was never linted in CI, so
  formatting results depended on the invocation directory.

### Changed

- Documentation realigned against the code: the retired Llama 3.3 70B model
  reference, the advertised test count, the documented status code for
  `POST /logout`, and the API version rendered on the public docs page.

## [1.3.0] — 2026-07-08

### Changed

- **Migrated the cloud provider model** from `llama-3.3-70b-versatile` to
  `openai/gpt-oss-120b` ahead of the former's end-of-life, at roughly a
  quarter of the input cost per million tokens.

### Added

- **Proactive tokens-per-minute pacing** for cloud completions. Reactive
  backoff cannot help a burst that trips the wall on its first call, so the
  client now estimates cost, acquires from a token bucket before dispatch, and
  reconciles against real usage afterwards. Disabled by default.
- **RBAC authorization matrix tests** covering every role against every
  protected endpoint.

### Fixed

- Honor `Retry-After` on 429s raised through the structured-output wrapper.
  The wrapper retries internally and re-raises as its own exception type, which
  the client had been treating as a provider outage — sending a recoverable
  rate limit down the deterministic fallback path.

## [1.2.2] — 2026-06-19

### Fixed

- Made the Neo4j database name configurable, as managed Aura instances name the
  default database after the instance ID rather than `neo4j`.

### Added

- GPU classification benchmark and a resolution-agent + guardrail benchmark for
  measuring the fine-tuned model at scale.

## [1.2.1] — 2026-06-17

### Fixed

- Installed CPU-only torch in the Space image, cutting the build from 8.89 GB
  to 2.13 GB and bringing it under the hosted tier's image limit.

## [1.2.0] — 2026-06-17

### Added

- **Managed-tier deployment support** — the system runs end to end on free
  managed services with no card on file: Postgres, Redis, Qdrant, Neo4j and a
  hosted inference provider, with the backend packaged as a single-container
  Space and the dashboard on Streamlit Cloud.
- TLS support for managed Postgres, URL + API-key configuration for hosted
  Qdrant, a configurable stream poll cadence for managed Redis, and a
  `SEED_ROWS` knob to size the corpus to a free-tier database.

## [1.1.0] — 2026-06-17

### Added

- Production deployment kit: hardened compose file, Caddy reverse proxy with
  automatic TLS, environment template and seed script.

## [1.0.0] — 2026-06-16

First complete release. All six build phases delivered.

### Added

- **Triage queue** with priority sorting, and product/company facet filters.
- **Workspace control room** — pipeline board with batch enqueue.
- **Viewer role** enforced read-only across every write route.
- **Stranded-message reclaim**: work left pending by a dead consumer is
  recovered rather than lost.
- `UNIQUE(complaint_id, version)` on resolutions.
- Architecture, fine-tuning, API reference and demo documentation; CI running
  lint, tests and a Docker build check on every branch.

## [0.5.0] — 2026-06-11 — Frontend Dashboard

### Added

- Streamlit dashboard: app shell with auth gate, dashboard home, triage queue,
  complaint detail with the agent review flow, analytics deep-dive, knowledge
  graph explorer, and an LLMOps observatory over the inference logs.
- A guided read-only tour for walk-up visitors, and page-aware engineering
  notes in the sidebar.
- Sentiment, product and company aggregate endpoints; similar-complaint lookup
  backed by vector search.

## [0.4.0] — 2026-06-10 — Agentic Intelligence Layer

### Added

- **Resolution agent** — tools, prompts and the orchestration loop that drafts
  a resolution from retrieved precedent and regulation.
- **Four-layer guardrail engine** validating every draft before it can be
  surfaced.
- Resolution worker draining the escalation stream, resolution lifecycle
  statuses with violation persistence, and the review API.

## [0.3.0] — 2026-06-02 — Knowledge & Retrieval Layer

### Added

- Neo4j graph store, seeded from curated regulations and resolution patterns
  plus relationships mined from Postgres; expanded to 66 regulations.
- Knowledge-graph read API, kept current by the classification worker.

## [0.2.0] — 2026-06-01 — Fine-Tuning Pipeline

### Added

- **QLoRA fine-tuning pipeline** for Qwen2.5-3B: label engineering with a
  teacher model, class-weighted loss for sentiment imbalance, an evaluation
  harness, and GGUF export for local serving.
- Local-first LLM client with cloud fallback, and the complaint classifier.
- Redis-stream classification worker.
- **LLMOps tracking** — token counts, cost and latency recorded per inference.

## [0.1.0] — 2026-05-24 — Foundation & Data Pipeline

### Added

- FastAPI backend, Docker Compose stack, and JWT authentication with a Redis
  revocation list.
- CFPB complaint ingestion — streaming bulk download with reservoir sampling
  into a 200K-row corpus — plus the complaints CRUD API.
- Sentence-transformer embedding service and Qdrant vector store, with a 200K
  embedding backfill.

[1.4.1]: https://github.com/deeason7/ResolveAI/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/deeason7/ResolveAI/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/deeason7/ResolveAI/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/deeason7/ResolveAI/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/deeason7/ResolveAI/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/deeason7/ResolveAI/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/deeason7/ResolveAI/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/deeason7/ResolveAI/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/deeason7/ResolveAI/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/deeason7/ResolveAI/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/deeason7/ResolveAI/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/deeason7/ResolveAI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/deeason7/ResolveAI/releases/tag/v0.1.0
