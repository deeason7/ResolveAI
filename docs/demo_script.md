# Demo script — the 10-minute walkthrough

A run-of-show for demoing ResolveAI live (interview, screen-share, or
hallway). Two acts: the guided tour sells the product, the live pipeline
sells the engineering.

## Pre-flight (2 minutes, before anyone is watching)

```bash
docker compose up -d          # 7/7 healthy
# start both workers (they're on-demand processes, not always-on services):
docker compose exec -d api python -m app.workers.classification_worker
docker compose exec -d api python -m app.workers.resolution_worker
```

- Check `http://localhost:8511` loads and `http://localhost:8010/docs` answers.
- Have a signed-in browser tab ready (your own account, not the demo user —
  you'll need the write buttons).
- A `GROQ_API_KEY` in `.env` (or the local GGUF loaded in Ollama) so
  classification has a brain.

## Act 1 — the guided tour (3 minutes)

Open the site logged out. **Point at the demo card** — "recruiters don't
register, so the login page leads with a read-only guided tour." Click
**Launch the demo** and let the tour boxes do the talking:

1. **Dashboard** — 200K real CFPB complaints; cards anchor to the data's own
   clock; the donut shows *unclassified* honestly.
2. **Triage Queue** — priority-ordered worklist, all server-side.
3. **Complaint Detail** (open one from triage) — narrative, classification,
   similar complaints with historical outcomes.
4. **Analytics → Graph Explorer → Observatory** — fast flip through; open
   the sidebar's **🛠️ Under the hood** on one page: "every page documents
   its own architecture decisions."

Exit demo, sign in as yourself.

## Act 2 — the live pipeline (5 minutes)

### Submit a complaint (via `/docs`, which shows off the API)

Authorize in Swagger with your bearer token (login response), then
`POST /api/v1/complaints/` with the showpiece:

```json
{
  "narrative": "My car was repossessed last night even though I have bank records proving every payment cleared on time. I called four times this week and was hung up on twice. I work night shifts and now have no way to get to my job. I am desperate and about to lose my income over your bank's accounting error.",
  "product": "Vehicle loan or lease",
  "issue": "Repossession",
  "company": "BigBank Auto Finance",
  "state": "OH"
}
```

Spares, if you want contrast:

```json
{
  "narrative": "There is an account on my credit report I never opened. I filed a dispute thirty days ago and have heard nothing back. I need this investigated and removed because it is dragging my score down while I am trying to qualify for a mortgage.",
  "product": "Credit reporting, credit repair services, or other personal consumer reports",
  "issue": "Incorrect information on your report",
  "company": "Equifax",
  "state": "GA"
}
```

```json
{
  "narrative": "I noticed a small maintenance fee on my statement this month that I do not remember being told about when I opened the account. Could someone explain what it covers and whether there is an account type without it?",
  "product": "Checking or savings account",
  "issue": "Fees or interest",
  "company": "BigBank",
  "state": "CA"
}
```

### Narrate the pipeline (~30s of waiting, fill it with the story)

"The API committed the row and queued a stream message — Postgres is the
truth, Redis is just the signal. The worker picks it up, tries the local
fine-tuned 3B first, falls back to Groq, and if *nothing* answers it
fail-closes to maximum severity — the system never guesses silently."

### Show the results

1. **Triage Queue** — the repossession lands high: extreme negative, urgency
   5, priority near the top. The fee question (if submitted) sits far below.
2. **Open detail** — classification chips; **similar complaints** panel:
   "cosine search over 200K embedded narratives, ~140ms, each neighbor shows
   how it was historically resolved."
3. **Generate resolution** — 202; ~15s later the draft appears (refresh).
   Walk the panel top to bottom: agent chain-of-thought (tools it called,
   what came back), **four guardrail layers** each with a verdict, the
   draft citing regulations the graph actually returned.
4. **Reject with feedback** — type something opinionated: *"Too apologetic,
   and do not promise the repossession will be reversed — commit only to an
   investigation with a named timeline."* The v2 draft incorporates it —
   "human feedback enters the same regeneration path as guardrail
   violations."
5. **Approve v2** — complaint flips to resolved.
6. **LLMOps Observatory** — the calls you just caused are already in the
   spend chart and latency bars: "telemetry is written in the same
   transaction as the work; this page can't lie."

## If asked "what happens when the LLM is down?"

Don't hand-wave — show the receipts: Observatory → routing donut → the red
`none` slice. "Sixteen real fail-closed events from a rate-limit storm
during development: every one flagged maximum severity for human review.
The failure path isn't theoretical; it's logged."

## Reset between demos

Approve or reject any drafts you created, and your submitted complaints are
real rows — either keep them (they make the queue livelier) or delete the
synthetic companies via SQL if you want the corpus pristine.
