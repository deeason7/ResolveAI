# Durable processing jobs

Complaint intake, generation, rejection and workspace batches commit their case
changes and job intent in one PostgreSQL transaction. Classification completion
commits labels, telemetry, indexing/drafting jobs and its own completion together.
Workers poll PostgreSQL directly; Redis remains necessary for authentication and
production request limits. A Redis outage does not erase committed processing work.

## Ownership and retries

Each job records its kind, case/input revision, request/review identity, attempt
budget, scheduled retry, heartbeat and lease. A short claim transaction selects an
available job with `FOR UPDATE SKIP LOCKED`, increments its attempts and assigns a
random lease token. The default lease is 120 seconds, renewed every 30 seconds.
Database time decides whether a lease is still live. External work holds no case
lock or database transaction.

Saving a result locks the case before the job and checks the live lease plus case
and input versions. An expired or replaced worker cannot save a draft, telemetry
or completion. A crash after commit needs no separate acknowledgement; a crash
before commit permits expiry/retry. The external provider may still receive the
call more than once. Late/discarded attempts are not yet a complete cost ledger.

Default automatic attempts are bounded at three, including reclaimed attempts.
Retryable failures back off from five seconds up to 300 seconds. Provider outages
remain visible as unavailable classification or failed drafting; they do not
produce fallback labels or successful-looking responses. Exhaustion leaves a
retained `failed` job. Invalid input and unexpected programming failures require
inspection instead of unlimited automatic retries. A saved draft with failed
checks is a completed job awaiting human review, rather than a transport retry.

Staff can inspect `GET /api/v1/jobs/` (failed by default, bounded at 100 per page)
and `GET /api/v1/jobs/{id}`. Responses expose safe error categories and attempt
counts, without request payloads or lease tokens. `POST /api/v1/jobs/{id}/retry`
requires the displayed `expected_attempts` and `expected_retry_generation`.
It reuses the same intent, grants another bounded budget and preserves lifetime
attempts. Replaying that exact retry returns the existing generation. Changed
inputs, newer review or an inconsistent retry identity returns 409. Workspace
shows pending/running/failed counts and the age of the oldest unfinished job.

## Admission and request identity

`JOB_MAX_PENDING` defaults to 10,000 queued/running jobs. PostgreSQL serializes
external admission before taking case locks. Requests exceeding capacity receive
503 before changes are accepted. This is an admission threshold, not a retention
limit: existing jobs are never trimmed, and accepted workflows may add bounded
follow-up jobs beyond it. Batches accept 1–500 cases and skip existing intents.

Intake and generation accept a bounded `Idempotency-Key` header. Keys are scoped
to actor and route; the same content returns its original case/job, including
when capacity is full. Different content with the same key returns 409. Without
an intake key, separate submissions create separate cases. A rejection defaults
to its exact draft/version identity, so repeating the same decision and feedback
returns the original job without reopening an approval or generating another
review event. A new batch request may admit another batch of eligible cases.

## Derived indexes

Classification schedules an independent index job. Qdrant upserts by complaint
ID, and Neo4j uses unique contribution receipts so replaying the same association
does not repeatedly increment counters. Partial store failure retries indexing
without rerunning inference. Job payloads retain extracted entities for replay.

Indexing checks the input revision. If an old external write finishes after a
newer input, it requests a durable rebuild of the current snapshot, including
when the old lease has expired. Same-input writes are equivalent; case review
status is not copied into the vector payload. External stores cannot participate
in the PostgreSQL transaction. A process dying during an external write still
requires reconciliation. Legacy aggregate baselines and taxonomy corrections
need a canonical rebuild; receipts do not repair historical over-counts. Neither
derived store is authoritative for review or case state.

## Running and migration boundary

The application supports only `JOB_BACKEND=postgres`; there is no automatic Redis
fallback. Fresh environments migrate through `f7c5b0e3d816`, then run:

```sh
python -m app.workers.classification_worker
python -m app.workers.resolution_worker
python -m app.workers.index_worker
```

Run each as its own process. Production Compose and supervisord define all three.
The bounded operator CLI uses the same transactional batch function:

```sh
python -m app.services.job_cli classify 50
python -m app.services.job_cli resolve 20
python -m app.services.job_cli monitor
python -m app.services.job_cli wait classify 30
```

It uses the operator's configured database; processing can invoke the configured
providers. `scripts/run_live_batch.sh` delegates to this CLI. These commands are
operational actions, not startup checks or offline tests.

**Existing deployments require a separate reviewed cutover.** The migration adds
an empty jobs table; it neither consumes old streams nor guesses work for legacy
cases. Before enabling the new producers/workers, prepare a verified backup,
pause old producers/consumers, inventory unread/pending stream entries and eligible
cases, preserve source IDs, deduplicate intents, and account for every exclusion.
Approved/reviewable cases must not be reopened to drain old work. Keep the old
stream snapshot read-only until reconciliation and a synthetic flow are verified.
That cutover tooling and live inventory are the next implementation step.

Do not run old Redis workers alongside PostgreSQL workers. Once durable jobs exist,
downgrade refuses to delete them. Use a compatible release or forward fix; never
blindly replay old streams. Complete backups include jobs, payloads, versions and
attempt/lease state. Restore into a matching empty schema with old workers stopped;
expired restored leases can be reclaimed by the matching worker release.
