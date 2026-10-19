# Product scope and evidence

Reviewed 2026-10-17. This is a prototype for complaint triage and response review,
not a system that establishes whether a consumer's underlying problem was solved.

## Initial user hypothesis

The proposed first user is a reviewer on a small team handling credit-reporting
disputes. The working hypothesis is that one queue, relevant case context, and a
reviewable draft reduce the effort of preparing a response. No interviews or
observed customer workflow establish that hypothesis yet. Validate it with relevant
reviewers before expanding the product; proxy-user feedback is a separate signal.

The existing software accepts several financial product categories. Narrowing the
initial demonstration to credit reporting is a product choice for evaluation, not
an implemented restriction on intake.

## Data and outcome boundaries

- Demonstrations use public historical CFPB records and clearly synthetic cases.
  Do not enter account numbers, contact details, or private customer documents.
- A public complaint is an allegation. A historical company response category is
  not a verified account of remediation, and a retrieved neighbor is not a legal
  precedent or proof that a proposed response will work.
- Imported, classified, drafted, approved, delivered, and resolved are different
  counts. The current `resolved` status is set by approval; it does not establish
  delivery, consumer satisfaction, or correction of a credit report.
- Drafts require a reviewer to check facts, cited sources, and commitments. Current
  pattern checks and tone scores do not certify legal accuracy or compliance.
- Hosted configuration skips local inference. Cloud calls can contain complaint
  text and retrieved context; output pattern redaction does not anonymize requests.

## One synthetic workflow to evaluate

**Case:** Alex Example disputes a fictional account on a credit report. The
complaint names Example Credit Bureau and contains no real identifying details.

1. A staff user submits the synthetic narrative and records its case ID.
2. A reviewer finds it in the queue and inspects the classification alongside the
   original text. Provider failure must be distinguished from a scored case.
3. The reviewer checks retrieved public cases and reference material, then requests
   a draft that promises investigation rather than a particular result.
4. The reviewer changes unsupported wording and checks exactly what will be stored
   and approved. Saved human edits and exact-version approval are required workflow
   improvements; the current editable text box is not a persisted revision editor.
5. Export or simulated delivery should be recorded separately from approval.
   Closure should require an explicit outcome, including unknown or unresolved.
   These outcome steps are proposed, not current application capabilities.

Measure task completion, time to a reviewed response, correction count, and whether
reviewers can identify stale or unavailable evidence. Record participants, dataset,
revision, failures, and comparison workflow. No time savings or resolution-rate
improvement has been established by this scope document.
