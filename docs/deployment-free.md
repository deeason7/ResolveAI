# Free-tier deployment ($0, no credit card)

This is the alternative to the single-VM runbook ([deployment.md](deployment.md)).
Instead of renting one box to run the whole Compose stack, it **splits the stack
across free, no-credit-card managed tiers** — so it costs nothing and stays up
for a year or more, with no $200 credit burning down.

Same code as `main`; the difference is config + a smaller corpus, carried on the
**`deploy`** branch (the branch the hosting platforms track). Nothing here is a
fork.

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
                                                   LLM ▶ Groq (free tier)
```

The browser only ever talks to Streamlit; Streamlit calls the Space's API
server-side. That's why `ENVIRONMENT=production` leaving the API's CORS allow-list
empty is correct here too — there's no cross-origin browser call to permit.

## Prerequisites

Six free accounts, **none requiring a card**:

| Service | Free tier | Holds |
|---------|-----------|-------|
| [Neon](https://neon.tech) | ~0.5 GB | Postgres (system of record) |
| [Upstash](https://upstash.com) | 500K commands/mo | Redis streams |
| [Qdrant Cloud](https://cloud.qdrant.io) | 1 GB cluster | embeddings |
| [Neo4j Aura](https://neo4j.com/cloud/aura-free/) | 1 instance | knowledge graph |
| [Hugging Face](https://huggingface.co) | CPU Basic Space | API + both workers |
| [Streamlit Community Cloud](https://streamlit.io/cloud) | 1 public app | dashboard |
| [Groq](https://console.groq.com) | free tier | classification + drafting LLM |

---

## 1. Provision the managed backends

Create each service, then collect its connection details into a local
`.env.free` (copy `.env.free.example` — it documents every value):

- **Neon** → copy the connection string. Swap the driver to `postgresql+asyncpg://`
  and **drop** the `?sslmode=require` query (asyncpg ignores it); TLS is applied by
  `DB_REQUIRE_SSL=true` instead.
- **Upstash** → copy the `rediss://` URL as-is. The `rediss` scheme turns on TLS
  automatically. The `.env.free` cadence knobs (`WORKER_BLOCK_MS=30000`,
  `WORKER_RECLAIM_EVERY=4`) keep the two idle workers under the 500K/mo command
  cap — leave them in.
- **Qdrant Cloud** → copy the cluster URL into `QDRANT_URL` and the key into
  `QDRANT_API_KEY`.
- **Neo4j Aura** → save the generated password at creation (shown once), then copy
  the `neo4j+s://` URI. Aura ships **APOC Core**, which is all the graph code uses
  (`apoc.path.subgraphAll`) — no extra plugins to enable.
- **Groq** → copy the API key into `GROQ_API_KEY`.

---

## 2. Backend → Hugging Face Space

The root `Dockerfile` builds one image that runs the API and both stream workers
together under `supervisord` (a free Space is a single container; each program
restarts independently). To deploy:

1. **Create a Space** → SDK **Docker**, hardware **CPU basic** (free).
2. **Give it the code.** Add the Space as a git remote and push the `deploy`
   branch to it (the Space builds from its own repo root, which is why the
   `Dockerfile` lives at the repo root).
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

> A free Space **pauses after ~48 h idle** and wakes on the next visit. While
> paused it runs nothing — which also means it spends zero Upstash commands.

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

# 2. download a ~30K sample (fits Neon 0.5 GB / Qdrant 1 GB)
.venv/bin/python ../scripts/download_cfpb_data.py \
  --output /tmp/cfpb_30k.csv --limit 30000 --workdir /tmp/cfpb_wd

# 3. ingest into Neon (idempotent: ON CONFLICT DO NOTHING)
.venv/bin/python - <<'PY'
import asyncio
from app.services.data_ingestion import ingest_cfpb_csv
print(asyncio.run(ingest_cfpb_csv("/tmp/cfpb_30k.csv")))
PY

# 4. embed into Qdrant Cloud (~15 min for 30K on a laptop CPU)
.venv/bin/python ../scripts/populate_vector_db.py

# 5. seed the Aura graph (top 500 companies)
.venv/bin/python ../graph_seed/seed_graph.py --limit 500
```

Every step is idempotent, so a run interrupted halfway just resumes on the next
invocation.

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

Open the Streamlit URL and **Try the demo** (or register). Then:

- The **Board** / **Triage** pages load complaints → Neon is connected.
- **Workspace → enqueue a small classification batch** → Upstash + the Space's
  classification worker are wired; watch a complaint move to *classified* /
  *escalated*.
- A complaint's **Similar** list returns matches → Qdrant Cloud is populated.
- A **company profile** / graph view renders → Aura + APOC are working.

If a page errors with "API unreachable", the `API_URL` secret is wrong or the
Space is still cold-starting (give it a minute on first hit).

## Limits to know

- **Neon 0.5 GB** caps the corpus at ~30K rows — that's the `SEED_ROWS` default
  used above.
- **Groq free tier** meters tokens/minute, so a large enqueue paces itself
  through the worker's rate-limit backoff rather than failing.
- **Upstash 500K/mo** is the reason for the widened worker poll cadence; don't
  drop `WORKER_BLOCK_MS` back to the dev default on this deployment.
- **HF idle pause** means the first visit after a quiet spell is slow while the
  Space wakes.
