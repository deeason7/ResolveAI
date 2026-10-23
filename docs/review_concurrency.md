# Saved response review and concurrent processing

A decision identifies one persisted draft and the case version displayed with
it. The API briefly locks the case, checks the submitted identity and versions,
verifies that the draft is the latest eligible revision, then writes the state,
review event and audit entry in one transaction. Competing decisions produce
one success and one 409 instructing the reviewer to reload. Rejection permanently
invalidates that revision for approval, even if generation later fails.

`case_version` advances when successful classification, generation, review or a
saved draft changes the case. Failed attempts preserve the job's expected version
so a bounded retry can recover without inventing new input. `input_revision`
advances when a new classification is accepted; future narrative/manual-label
edits must advance it too. Approval changes state without changing inputs. Draft
version numbers count saved responses, allocated while holding the case lock and
protected by a unique constraint.

Durable jobs carry their own identity, expected case version and input revision.
Workers check eligibility before inference, end the read transaction during
external work, then lock the case and the owned job before persisting output.
An expired lease, obsolete input or completed job cannot reopen approved text or
replace a newer draft. Claims do not advance case versions. Classification-triggered
work remains `escalated` during inference; explicit requests use `agent_triggered`.

A rejection and its replacement job commit together. Repeating the same rejection
identity and feedback returns the original job; it does not repeat the review or
reopen a later approval. Reusing a request key with different content returns 409.
Duplicate external calls can still occur after lease expiry, but their persisted
business effects are fenced. See [durable jobs](jobs.md) for admission, bounded
attempts, staff retries and the migration boundary.

## Reviewer behavior

The UI retains the entire displayed saved draft and its case version across
reruns. An explicit reload replaces that snapshot. Buttons send its identity;
a 409 does not silently fetch a different draft and resubmit approval. The draft
text is read-only and downloadable. Saved human edits are a separate workflow.

New approvals use `response_approved`, meaning response text was approved, with
no delivery or customer outcome asserted. Legacy `resolved` cases remain
unchanged and marked as legacy approval/outcome unknown. Migration assigns a
version baseline of 1 and marks existing drafts `legacy_provenance=true`; it does
not reconstruct historical input provenance or fabricate review events.

## Deployment and recovery

This is a coordinated application/worker/client change. Prepare a complete
backup and verified recovery target. Stop old writers and consumers before
applying migration `e6b4a9d2c705`, then start matching API, workers and frontend.
Do not mix old writers with the new version protocol: they do not advance case
versions. Existing clients must reload; missing review identities fail closed.
No shared database migration or deployment is performed by the local test suite.

Inventory queued and in-flight work before enabling the PostgreSQL producers and
workers. The new binaries do not consume Redis streams. Existing stream messages
and stranded cases need a reviewed reconciliation/cutover; the additive migration
does not invent jobs for them. Do not reset approved or reviewable cases or replay
historical rejection feedback to make a queue drain. See [jobs](jobs.md).

PostgreSQL prevents updates to saved draft rows and updates/deletes to review
events. Reviewed drafts cannot be deleted. Corrections require a new revision.
Once reviews exist, downgrade refuses to delete that history; prefer a compatible
forward fix. Restoring an older backup requires its matching application/schema
in an isolated target before a forward upgrade.

## Verification

`backend/tests/integration/test_review_concurrency.py` uses independent PostgreSQL
sessions, barriers and events for competing decisions, duplicate generation,
concurrent result persistence and slow inference crossing an approval or input
change. It also checks raw SQL immutability and downgrade refusal. The backup
integration tests restore exact review relationships and text hashes. SQLite and
frontend contract tests cover stale/cross-case IDs, missing identities, rejection
invalidation and retained UI snapshots; they do not establish browser behavior.
