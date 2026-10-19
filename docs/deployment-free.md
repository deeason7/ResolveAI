# Managed demonstration deployment

This is the existing alternative to the single-VM runbook
([deployment.md](deployment.md)). It splits the stack across managed services and
uses a smaller corpus. Provider plans, card requirements, storage limits, idle
policies and pricing have not been reverified; no zero-cost or availability
promise is made. Confirm current account limits before deploying.

The hosting configuration uses the `deploy` branch. Local changes do not update
hosted behavior until a reviewed deployment completes. Use only public/synthetic
data while the workflow and recovery limits in [architecture.md](architecture.md)
remain unresolved. Production examples skip local inference and send model
requests to the configured cloud provider.

## What runs where

```
                Streamlit Community Cloud
 internet ───▶  frontend (Streamlit)  ── server-side ──▶  Hugging Face Space
                                                          ├─ api (FastAPI)
                                                          ├─ classification-worker
                                                          └─ resolution-worker
                                                              │  (one container,
                                                              │   supervisord)
                    ┌──────────────┬──────────────┬──────────┴─────┐
                    ▼              ▼              ▼                ▼
                 Neon           Upstash       Qdrant Cloud      Neo4j Aura
                 (Postgres)     (Redis)       (vectors)         (graph)
                                                   LLM ▶ Groq (configured plan)
```

The browser only ever talks to Streamlit; Streamlit calls the Space's API
server-side. That's why `ENVIRONMENT=production` leaving the API's CORS allow-list
empty is correct here too — there's no cross-origin browser call to permit.

## Prerequisites

Accounts and credentials for these services are required. Choose current plans
from the providers' published terms; the table describes component roles only.

| Service | Holds |
|---------|-------|
| [Neon](https://neon.tech) | PostgreSQL business records |
| [Upstash](https://upstash.com) | Redis streams and token revocation |
| [Qdrant Cloud](https://cloud.qdrant.io) | Narrative embeddings |
| [Neo4j Aura](https://neo4j.com/cloud/aura-free/) | Derived graph |
| [Hugging Face](https://huggingface.co) | API and both workers |
| [Streamlit Community Cloud](https://streamlit.io/cloud) | Dashboard |
| [Groq](https://console.groq.com) | Cloud classification and drafting |

---

## 1. Provision the managed backends

Create each service, then collect its connection details into a local
`.env.free` (copy `.env.free.example` — it documents every value):

- **Neon** → copy the connection string. Swap the driver to `postgresql+asyncpg://`
  and **drop** the `?sslmode=require` query (asyncpg ignores it); TLS is applied by
  `DB_REQUIRE_SSL=true` instead.
- **Upstash** → copy the `rediss://` URL as-is. The `rediss` scheme turns on TLS
  automatically. The `.env.free` cadence knobs (`WORKER_BLOCK_MS=30000`,
  `WORKER_RECLAIM_EVERY=4`) reduce idle polling. Measure actual commands and
  compare them with your plan; this is not a quota guarantee.
- **Qdrant Cloud** → copy the cluster URL into `QDRANT_URL` and the key into
  `QDRANT_API_KEY`.
- **Neo4j Aura** → save the generated password at creation (shown once), then copy
  the `neo4j+s://` URI. Aura ships **APOC Core**, which is all the graph code uses
  (`apoc.path.subgraphAll`) — no extra plugins to enable.
- **Groq** → copy the API key into `GROQ_API_KEY`.

---

## 2. Backend → Hugging Face Space

The root `Dockerfile` builds one image that runs the API and both stream workers
together under `supervisord` (this topology uses a single container; each program
restarts independently). To deploy:

1. **Create a Space** with the Docker SDK and hardware appropriate to the workload.
2. **Give it the code.** Add the Space as a git remote and push the `deploy`
   branch to it (the Space builds from its own repo root, which is why the
   `Dockerfile` lives at the repo root):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push --force space deploy:main
   ```
   Only the first push is manual — after that, releases ship themselves
   (see *Releasing* below).
3. **Add the front-matter.** HF needs a YAML header at the top of `README.md` on
   the branch it builds. Prepend this on `deploy`:
   ```yaml
   ---
   title: ResolveAI
   emoji: 🛠️
   colorFrom: indigo
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
   `app_port: 7860` matches the `PORT` the Dockerfile sets and uvicorn binds.
4. **Set the secrets.** In the Space's *Settings → Variables and secrets*, add
   every key from your filled `.env.free` (DATABASE_URL, DB_REQUIRE_SSL, REDIS_URL,
   QDRANT_URL/QDRANT_API_KEY, NEO4J_URI/USER/PASSWORD, GROQ_API_KEY, the JWT
   secrets, ENVIRONMENT=production, and the two WORKER_* knobs).

> Confirm the current hosting plan's idle/pause behavior. Paused workers do not
> process queued complaints; a successful frontend load does not prove worker progress.

---

## 3. Seed the managed backends (once, from your machine)

`seed_all.sh` drives a local Compose stack, so it doesn't fit the all-managed
topology. Instead, run the same five steps locally, pointed at the managed
endpoints. You need the backend dev environment (`cd backend && pip install -e
".[dev]"`).

```bash
# From the repo root: fill in the managed endpoints, then load them into the env.
cp .env.free.example .env.free        # edit with your Neon/Upstash/Qdrant/Aura/Groq values
set -a; . ./.env.free; set +a

cd backend
export PYTHONPATH=.

# 1. schema on Neon (DB_REQUIRE_SSL=true applies TLS)
.venv/bin/alembic upgrade head

# 2. example sample size; verify database/index capacity for your selected plans
.venv/bin/python ../scripts/download_cfpb_data.py \
  --output /tmp/cfpb_30k.csv --limit 30000 --workdir /tmp/cfpb_wd

# 3. ingest into Neon (idempotent: ON CONFLICT DO NOTHING)
.venv/bin/python - <<'PY'
import asyncio
from app.services.data_ingestion import ingest_cfpb_csv
print(asyncio.run(ingest_cfpb_csv("/tmp/cfpb_30k.csv")))
PY

# 4. embed into Qdrant Cloud; measure runtime and completion on your hardware
.venv/bin/python ../scripts/populate_vector_db.py

# 5. seed the Aura graph (top 500 companies)
.venv/bin/python ../graph_seed/seed_graph.py --limit 500
```

Inspect stage output and persisted state before retrying. The download/extraction
and import paths have known interruption and batch-size limits; do not infer a
complete import from the existence of a nonempty output file.

---

## 4. Frontend → Streamlit Community Cloud

1. **New app** from your GitHub repo, branch **`deploy`**, main file
   `frontend/app.py`. `frontend/requirements.txt` and `.streamlit/config.toml`
   are picked up automatically.
2. **Set the secret.** In *Advanced settings → Secrets*, paste (see
   `.streamlit/secrets.toml.example`):
   ```toml
   API_URL = "https://<your-space>.hf.space"
   ```
   Streamlit exposes top-level secrets as environment variables, which is how the
   dashboard's `os.environ.get("API_URL")` reaches the Space.

---

## 5. Verify end to end

Open the Streamlit URL with an explicitly provisioned viewer account for browsing
and a staff account for the synthetic write checks. Registration does not grant
staff access. Then:

- The **Board** / **Triage** pages load complaints → Neon is connected.
- **Workspace → enqueue a small classification batch** → Upstash + the Space's
  classification worker are wired; watch a complaint move to *classified* /
  *escalated*.
- A complaint's **Similar** list returns matches → Qdrant Cloud is populated.
- A **company profile** / graph view renders → Aura + APOC are working.

If a page errors with "API unreachable", the `API_URL` secret is wrong or the
Space is still cold-starting (give it a minute on first hit).

## Releasing

`deploy` is `main` plus the front-matter above — nothing else. Pushing a version
tag hands the rest to CI (`.github/workflows/deploy.yml`):

```bash
git checkout main && git merge --no-ff develop
git tag -a v1.4.0 -m "..." && git push origin main --tags
```

The workflow re-runs lint, tests and Compose image builds, refuses to ship a tag that
isn't on `main`, merges the tagged commit into `deploy`, and force-pushes
`deploy` to the Space's `main` to trigger a rebuild. Merging to `main` on its own
deploys nothing; the tag triggers remote publication. The gate does not currently
build/test the root hosting Dockerfile or verify post-deploy readiness; do not
equate a successful Git push with a healthy deployed workflow.

It needs one secret, `HF_TOKEN` — a Hugging Face token with **write** access to
the Space, under *Settings → Secrets and variables → Actions*. Without it the
job fails at the publication step. Inspect remote branch and hosting state
before retrying a failed release.

A merge conflict fails the run; inspect the actual conflicting paths, which may
include `README.md` where the front-matter lives. Fix it locally, push `deploy`, and
re-run the workflow from the Actions tab.

## Backups

PostgreSQL contains irreplaceable business identities and relationships, including
complaint UUIDs, users, labels, drafts, telemetry and review records. Downloading the
same public corpus again does not recover those UUIDs or reconnect child records.
Vector and graph stores are derived; rebuild them only after restoring the database.

Use a full logical PostgreSQL backup with a compatible `pg_dump` client, secure
credentials, a manifest/checksum and a restore drill into an isolated empty target.
See [the operational backup direction](deployment.md#operations). Use a client
matching the server major version; client setup effort is not a reason to omit
authoritative complaint rows. Never tar a running PostgreSQL data directory.

Legacy `backup_state.py` exports defaulted to omitting complaints. Treat those
archives as partial; re-ingesting the corpus and skipping conflicting IDs is not
a verified recovery procedure. Do not override schema checks or merge into a
nonempty target to make a restore appear successful. Use the verified version 2
[archive procedure](backup_recovery.md) for new backups; this does not repair old archives.

Backup files contain sensitive account and complaint data. Keep them outside Git,
restrict access, and prove that representative identities, relationships and draft
history survive restoration before relying on an archive.

## Capacity and inactivity

Measure storage, Redis command usage and inference tokens against the selected
plans. Polling settings reduce idle traffic but do not eliminate backlog growth or
provider limits. Define an observable failure path for suspended services; a page
load or `GET /health` only proves the process answered, not that datastores and
workers are functioning.

Check each provider's current pause, archival and deletion terms. Keep backups
outside the provider and record recovery steps. Do not rely on scheduled keepalive
requests as a guarantee against deletion or a substitute for a verified restore.

## Revised account and archive procedures

Before enabling a public deployment, follow [account provisioning](access.md)
for the viewer demo and explicitly selected staff users. Existing roles are not
automatically remediated by changing the default registration role.

Use the [version 2 backup/restore procedure](backup_recovery.md) for complete
application archives. It includes complaints by default, requires a matching
empty restore target, and removes the legacy --force/partial-restore behavior.
Hosted access, backup scheduling and production recovery still need verification.
