# Demo script

Demonstrate observable behavior with public records and synthetic intake. Explain
what a screen proves and where the implementation still needs work. See
[product scope](product_scope.md) for the initial reviewer hypothesis.

## Pre-flight

Use a disposable local environment for write demonstrations. Verify an explicitly
provisioned staff account and a separate viewer account before presenting. A viewer
must not be able to write through the API, even if it bypasses the UI.

```bash
docker compose up -d
docker compose exec -d api python -m app.workers.classification_worker
docker compose exec -d api python -m app.workers.resolution_worker
```

Check the dashboard at `http://localhost:8511` and API docs at
`http://localhost:8010/api/docs`. Confirm the selected provider: production examples
skip local inference, while local Ollama requires the exported model. A live model
call may use external quota or incur cost; do not demonstrate with personal data.

## Read-only walkthrough

1. **Dashboard:** show the loaded record count and date range. Imported records
   are not evidence of classification, drafted responses, or resolved cases.
2. **Triage:** inspect ordering, filters, unscored cases, and an original narrative.
   Compare the proposed priority with the actual text rather than asserting it is
   correct because a model assigned it.
3. **Complaint detail:** show similar public complaints and their response
   categories. These are context, not verified successful resolutions.
4. **Analytics and graph:** explain that company scores and reference links are
   derived signals; do not present them as legal or causal findings.
5. **Observatory:** inspect logged calls and their provider. Explain incomplete
   failure/cost coverage before interpreting a total.

The public demo's hosting and account state must be verified for the revision
being presented. A source-code change alone does not update a hosted deployment.

## Synthetic staff workflow

Submit this through `POST /api/v1/complaints/` using a staff bearer token:

```json
{
  "narrative": "This is a synthetic demonstration. Alex Example found an unfamiliar account on a credit report and asks Example Credit Bureau to investigate. The example includes no real account number or identifying documents.",
  "product": "Credit reporting, credit repair services, or other personal consumer reports",
  "issue": "Incorrect information on your report",
  "company": "Example Credit Bureau",
  "state": "GA"
}
```

Record the case ID, then:

1. Watch the case enter the queue. Report observed completion and latency; do not
   promise a classification or a fixed response time in advance.
2. Inspect the narrative and classification together. If no provider answered,
   describe the failure instead of presenting the deterministic fallback as a
   prediction. The reviewed baseline uses urgency 3, not maximum severity.
3. Request a draft. Inspect tool results, proposed text and check outcomes. A
   citation-title match does not verify the law or support every sentence.
4. Reject an unsupported promise with concrete feedback and inspect the new text.
   The editable draft box is currently for copy-out: edits are not saved.
5. Before any approval, compare the visible and stored draft. Exact revision
   approval is an outstanding integrity requirement. Do not demonstrate this
   baseline as suitable for real customer decisions.
6. Explain that the application's `resolved` label currently records approval.
   Delivery, remediation and customer satisfaction have not been measured.

## Failure demonstration and cleanup

Use controlled local provider stubs or a deliberately isolated test configuration
for failure scenarios. The current Redis publication and replay paths have known
transaction gaps; describe those rather than promising lossless recovery.

Keep synthetic case IDs with the demo notes. Do not delete rows by a broad company
name selector or approve a draft merely to tidy a demo. Reset only an explicitly
disposable environment after confirming that no real work is stored there.
