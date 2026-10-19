# Deployment

This is the runbook for putting ResolveAI on a public URL. It uses a separate
production Compose file (`docker-compose.prod.yml`) that differs from the dev
stack. This describes configuration, not a production-readiness certification.
Before a public release, verify access roles, session transport, recovery and the
known workflow limitations in [architecture.md](architecture.md).

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
  directly — Streamlit calls it server-side. Network isolation is only one
  part of access control; it does not validate API authorization or session transport. (`ENVIRONMENT=production` also leaves the API's
  CORS allow-list empty, which is correct here.)
- **Caddy provisions and renews TLS automatically** from Let's Encrypt for your
  domain. No certbot, no cron.
- **The stream workers run as services** (`classification-worker`,
  `resolution-worker`) with `restart: unless-stopped`, instead of being started
  by hand like in dev.
- **Ollama is opt-in.** The production example uses cloud classification
  (`LLM_SKIP_LOCAL=true`). Provider pricing, quotas and data handling need review. A bigger box can
  run the local 3B with `--profile local-llm` (and `LLM_SKIP_LOCAL=false`).

## Prerequisites

- A Linux server with a public IP and enough capacity for the selected corpus
  and services. Ubuntu 22.04 on DigitalOcean is the historical example below.
- A domain (or subdomain) you can point an `A` record at.
- A `GROQ_API_KEY` for cloud classification/drafting; verify current provider terms.

---

## The server

The historical example uses a **DigitalOcean** 4 GB / 2 vCPU VM. This is not a
measured capacity recommendation. Check current pricing and workload memory use.
Source and public corpus scripts cannot recreate accounts, complaint UUIDs, labels,
drafts or review history. Keep verified database backups outside the VM.

1. **Create the droplet.** Ubuntu 22.04 (x64), Basic **4 GB / 2 vCPU**, in the
   region nearest you (e.g. Bangalore `BLR1`). Add your SSH public key during
   creation. Configure the firewall explicitly for the intended exposed ports.

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

For a new disposable installation, inspect and run the seed script. For an
existing instance, establish a restorable backup before its migration step:

```bash
./scripts/seed_all.sh
```

That runs migrations → downloads + ingests the CFPB corpus → embeds it into
Qdrant → seeds the knowledge graph. Runtime depends on corpus size and hardware;
verify completion before retrying an interrupted stage. Once Postgres has
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
- **Back up PostgreSQL logically.** Do not tar a running PostgreSQL data directory;
  the removed live-volume recipe did not establish a consistent recoverable backup.
  Use `pg_dump` with a client matching the server major version, an identified
  source database and a custom-format output (`--format=custom`). Connect using
  protected libpq credentials/service configuration rather than pasting secrets
  into shell history. See the [PostgreSQL pg_dump documentation](https://www.postgresql.org/docs/current/app-pgdump.html).
- Record server/client versions, application revision, schema revision, completion
  status and an archive checksum. Store the archive securely outside the host;
  it contains account data, password hashes and complaint content.
- Prove recovery with `pg_restore --exit-on-error` into a separate empty database,
  then verify complaint IDs, foreign keys, draft history and representative reads.
  Rebuild derived indexes only after database recovery. Record duration; no
  recovery-time guarantee is established yet. Never test restore over live data.
- Old `backup_state.py` exports may omit complaints and their UUIDs. Re-importing
  the public corpus creates new identities and does not repair those archives.
  Preserve them as partial evidence, not a full recovery source.

## Troubleshooting

- **Certificate never issues / site shows a TLS error.** Check, in order: the
  `A` record resolves (`dig +short your.domain`), ports 80 **and** 443 are open
  in your cloud firewall, and `ACME_EMAIL`/`DOMAIN` are set in `.env.production`.
  Watch `docker compose -f docker-compose.prod.yml logs -f caddy`. While testing
  repeatedly, switch to Let's Encrypt staging to avoid consuming issuance quotas — add `acme_ca https://acme-staging-v02.api.letsencrypt.org/directory`
  to the global options block in `Caddyfile`, then remove it for the real cert.
- **App loads but hangs on "Please wait…".** That's the Streamlit WebSocket
  being blocked. Confirm Caddy proxies WebSockets to `frontend:8501` and inspect
  logs. The current image disables CORS/XSRF protections; treat this as an
  outstanding configuration review, not a recommended blanket fix.
- **A container is killed with exit code 137.** Out of memory — apply the Neo4j
  memory caps + swap above, and stay on the Groq-only default (don't enable
  `local-llm`).

## Revised account and archive procedures

Before enabling a public deployment, follow [account provisioning](access.md)
for the viewer demo and explicitly selected staff users. Existing roles are not
automatically remediated by changing the default registration role.

Use the [version 2 backup/restore procedure](backup_recovery.md) for complete
application archives. It includes complaints by default, requires a matching
empty restore target, and removes the legacy --force/partial-restore behavior.
Hosted access, backup scheduling and production recovery still need verification.
