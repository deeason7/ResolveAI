# Classification outcomes and rollout

Classification evidence is separate from complaint lifecycle. A case can retain
its previous lifecycle while a new classification attempt is unavailable.

| Outcome | Meaning |
|---|---|
| pending | New case has no completed classification attempt |
| succeeded | A provider returned a validated classification |
| unavailable | Latest attempt could not obtain a classification |
| manual | Reserved for human-supplied labels; manual editing is not implemented yet |
| legacy_unknown | Existing labels have no established provenance |

New unavailable cases have no invented labels or priority. They enter needs_review
and appear under Needs triage in the queue. Failed reclassification preserves prior
labels, source and classification time, but marks them stale. Generation,
regeneration, batch drafting and resolution-worker inputs reject stale/unavailable
labels. Successful fallback records its actual provider; elapsed classification
time includes failed attempts and pacing. Outages log classify_unavailable with
unknown usage/cost rather than contributing to successful drift samples.

Manual-source and resolved cases are skipped by classification workers. A second
read under a row lock protects a case changed during classification inference.
This does not provide full input-version fencing, durable retries, exactly-once
external calls or protection for an already-running resolution attempt.

The interface makes triage work visible. It does not yet let staff save manual
labels or complete a case without the drafting workflow. Controlled retry belongs
with durable jobs; do not substitute direct database edits as the routine workflow.

## Deployment order

1. Identify the target revision and take a complete backup using compatible code.
   Prove a restore into a separate database before changing shared data.
2. Stop old writers/workers and drain or reconcile in-flight work. Mixed old/new
   classification workers are unsafe: old code can still write plausible fallback
   labels without maintaining the new provenance fields.
3. Apply the forward migrations. a7c3d9e5f102 changes only the users.role insert
   default to viewer. c4a8e2d9f601 adds classification fields; existing rows become
   legacy_unknown, while later inserts default to pending. No existing role,
   label, draft, approval or lifecycle is rewritten.
4. Deploy matching API, frontend and workers. Confirm the queue/detail contracts,
   then run a synthetic provider-outage case and inspect persisted state and logs.
5. Resume traffic only after accounting for the old queue. Full queue cutover and
   reconciliation tooling remain unfinished; this document is not a cutover proof.

Old readers can still select their original columns after these additive changes.
Old writers do not maintain the new evidence contract, so compatibility of columns
does not make a rolling mixed-worker deployment safe.

## Rollback and recovery

Keep the additive columns when rolling back application code for diagnosis. An old
binary cannot safely resume classification because it restores the outage defect.
Stop processing and fix forward, or restore the backed-up revision into a separate
target under an explicit recovery plan. Downgrading c4a8e2d9f601 drops provenance;
downgrading a7c3d9e5f102 restores the analyst insert default. Neither downgrade is a
routine data-preserving rollback.

Version 2 backups cover all new classification fields. Restore a pre-upgrade
archive to its recorded revision first, verify it, then upgrade. A schema mismatch
must fail visibly rather than silently omitting new fields.
