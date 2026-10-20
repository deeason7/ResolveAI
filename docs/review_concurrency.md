# Saved response review and concurrent processing

A decision identifies one persisted draft and the case version displayed with
it. The API briefly locks the case, checks the submitted identity and versions,
verifies that the draft is the latest eligible revision, then writes the state,
review event and audit entry in one transaction. Competing decisions produce
one success and one 409 instructing the reviewer to reload. Rejection permanently
invalidates that revision for approval, even if generation later fails.

`case_version` advances on a committed classification outcome, generation request,
review decision or worker completion. Dispatch-failure recovery also advances it
to invalidate messages whose enqueue outcome is uncertain. `input_revision`
advances when a new classification is accepted; future narrative/manual-label
edits must advance it too. An approval changes case state without changing its
inputs. Draft version numbers count saved response revisions; they are allocated
while holding the case lock and protected by a unique constraint.

Both queue message types require `expected_case_version` and `input_revision`.
Workers check eligibility before inference, release the read transaction during
external work, and lock/recheck the case before committing a result. An old
message or a result from an obsolete snapshot cannot reopen approved text or
replace a newer draft. A claim does not advance versions, allowing a failed
attempt to retry the same message. Classification-triggered work remains
`escalated` during inference; explicit generation requests use `agent_triggered`.

Duplicate inference calls can still occur. Only the first eligible completion
can persist; this is not an exactly-once provider-execution guarantee. Stale
attempt cost accounting, durable jobs, leases, bounded retries and automatic
reconciliation remain separate work. A process crash between database commit and
Redis publish can still strand accepted work. The endpoints report known dispatch
failures, invalidate uncertain attempts and leave an explicit retry path; this
does not close the crash window.

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

Inventory queued and in-flight work before cutover. Old unversioned messages are
logged and discarded because they cannot prove their source snapshot. Requeue
eligible pending classification and escalated resolution cases through the
updated workspace endpoints. Inspect old `agent_triggered` cases separately;
there is no automatic reconciliation yet. Do not reset approved or reviewable
cases or replay historical rejection feedback to make a queue drain.

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
