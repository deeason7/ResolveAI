# Deployment

This is the runbook for putting ResolveAI on a public URL. It uses a separate
production Compose file (`docker-compose.prod.yml`) that differs from the dev
stack in the ways that matter on a real host.

## What the production stack looks like

```
                 :80 / :443
internet ──────────────────▶  Caddy  ──▶  frontend (Streamlit)
                              (TLS)              │  server-side
                                                 ▼
                                               api (FastAPI)
                                                 │
                    ┌────────────┬───────────────┼───────────────┐
                    ▼            ▼               ▼               ▼
                 postgres      redis          qdrant           neo4j
                 (+ classification-worker, resolution-worker)
```

- **Only Caddy is published** (80/443). The API and every datastore stay on the
  internal Docker network with no host ports. The browser never talks to the API
  directly — Streamlit calls it server-side — so this isn't a limitation, it's
  the whole security posture. (`ENVIRONMENT=production` also leaves the API's
  CORS allow-list empty, which is correct here.)
- **Caddy provisions and renews TLS automatically** from Let's Encrypt for your
  domain. No certbot, no cron.
- **The stream workers run as services** (`classification-worker`,
  `resolution-worker`) with `restart: unless-stopped`, instead of being started
  by hand like in dev.
- **Ollama is opt-in.** By default we classify through Groq's free tier
  (`LLM_SKIP_LOCAL=true`), which is what a low-RAM host wants. A bigger box can
  run the local 3B with `--profile local-llm` (and `LLM_SKIP_LOCAL=false`).

## Prerequisites

- A server with a public IP — any Ubuntu 22.04 VM works (DigitalOcean is walked
  through below).
- A domain (or subdomain) you can point an `A` record at. The GitHub Student
  Developer Pack includes a free Namecheap `.me` domain for a year.
- A `GROQ_API_KEY` (free tier) for classification.

---

## The server

Any Ubuntu 22.04 host with a public IP runs this stack the same way. A
**DigitalOcean** 4 GB / 2 vCPU droplet is the worked example below — the GitHub
Student Pack's $200 credit covers roughly eight months of it. Treat the VM as
disposable: nothing precious lives on it, because `git` + `seed_all.sh` rebuild
its entire state from scratch.

1. **Create the droplet.** Ubuntu 22.04 (x64), Basic **4 GB / 2 vCPU**, in the
   region nearest you (e.g. Bangalore `BLR1`). Add your SSH public key during
   creation. Ports 80/443 are open by default.

2. **Give a 4 GB box headroom.** It's tighter on RAM than a large VM, so:
   - Keep the default **Groq-only** path (don't enable `local-llm`; Ollama + a
     3B model won't fit).
   - Cap Neo4j's memory so it can't OOM the box — add to the `neo4j` service
     `environment:` in `docker-compose.prod.yml`:
     ```yaml
     NEO4J_server_memory_heap_max__size: 512m
     NEO4J_server_memory_pagecache_size: 512m
     ```
   - Add swap as a safety net:
     ```bash
     sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
     sudo mkswap /swapfile && sudo swapon /swapfile
     echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
     ```

3. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker "$USER" && newgrp docker
   ```

4. **Point DNS at the box.** Create an `A` record for your domain → the droplet's
   public IP, and *wait for it to resolve* (`dig +short your.domain`) before
   bringing Caddy up — the TLS challenge fails if the name doesn't resolve yet.

5. **Clone, configure, launch** (see [Bring-up](#bring-up) below).

---

## Bring-up

From the repo root on the server:

```bash
git clone https://github.com/deeason7/ResolveAI.git && cd ResolveAI

cp .env.production.example .env.production
# Edit .env.production: set DOMAIN + ACME_EMAIL, your GROQ_API_KEY, and a fresh
# random string for every __change_me__ (generate with: openssl rand -hex 32).
# Keep the paired values in sync (POSTGRES_PASSWORD/DATABASE_URL,
# REDIS_PASSWORD/REDIS_URL, NEO4J_PASSWORD/NEO4J_AUTH).

docker compose -f docker-compose.prod.yml up -d --build
```

`--build` builds the `api` and `frontend` images first; the two worker services
reuse the freshly tagged `resolveai/api:latest`, so they need the build to
happen before they start.

Then load the corpus (idempotent, so it's safe to re-run if interrupted):

```bash
./scripts/seed_all.sh
```

That runs migrations → downloads + ingests the CFPB corpus → embeds it into
Qdrant (~90 min on CPU for 200K) → seeds the knowledge graph. Once Postgres has
rows, classification starts the moment you enqueue a batch from the **Workspace**
page (the workers are already running).

Visit `https://your.domain` — Caddy will have issued the certificate on first
request.

## Operations

- **Logs:** `docker compose -f docker-compose.prod.yml logs -f caddy api classification-worker`
- **Update to a new release:**
  ```bash
  git pull
  docker compose -f docker-compose.prod.yml up -d --build
  ```
- **Back up the data volumes** (Postgres is the system of record; Caddy's volume
  holds your certs). Find the exact names with `docker volume ls`, then:
  ```bash
  docker run --rm -v <project>_postgres_data:/v -v "$PWD":/b alpine \
    tar czf /b/postgres-backup.tgz -C /v .
  ```

## Troubleshooting

- **Certificate never issues / site shows a TLS error.** Check, in order: the
  `A` record resolves (`dig +short your.domain`), ports 80 **and** 443 are open
  in your cloud firewall, and `ACME_EMAIL`/`DOMAIN` are set in `.env.production`.
  Watch `docker compose -f docker-compose.prod.yml logs -f caddy`. While testing
  repeatedly, switch to Let's Encrypt staging to avoid the 5-duplicate-certs-per-week
  rate limit — add `acme_ca https://acme-staging-v02.api.letsencrypt.org/directory`
  to the global options block in `Caddyfile`, then remove it for the real cert.
- **App loads but hangs on "Please wait…".** That's the Streamlit WebSocket
  being blocked. The prod frontend already disables Streamlit's own CORS/XSRF
  checks for exactly this reason; confirm Caddy is proxying `frontend:8501` and
  the frontend container is healthy.
- **A container is killed with exit code 137.** Out of memory — apply the Neo4j
  memory caps + swap above, and stay on the Groq-only default (don't enable
  `local-llm`).
