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
  (`LLM_SKIP_LOCAL=true`), which is what an ARM or low-RAM host wants. A bigger
  box can run the local 3B with `--profile local-llm` (and `LLM_SKIP_LOCAL=false`).

## Prerequisites

- A server with a public IP (targets below).
- A domain (or subdomain) you can point an `A` record at. The GitHub Student
  Developer Pack includes a free Namecheap `.me` domain for a year.
- A `GROQ_API_KEY` (free tier) for classification.

---

## Target A — Oracle Cloud Always Free (recommended)

The Ampere A1 shape (up to **4 ARM cores / 24 GB**, always free) is the only
forever-free tier big enough for the whole stack. Treat the VM as disposable:
nothing precious lives on it, because `git` + `seed_all.sh` rebuild it.

1. **Create the instance.** VM.Standard.A1.Flex, Ubuntu 22.04 (aarch64),
   4 OCPU / 24 GB. If you hit *"Out of host capacity"*, that's the known
   always-free lottery — retry in another availability domain or a few hours
   later.

2. **Open the ports — in BOTH places.** This is the #1 Oracle gotcha: the cloud
   VCN *and* the instance firewall block by default.
   - In the VCN **Security List** (or an NSG on the VNIC), add ingress rules for
     TCP **80** and **443** from `0.0.0.0/0`.
   - On the instance, Oracle's Ubuntu image ships locked-down iptables. Open and
     persist 80/443:
     ```bash
     sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
     sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
     sudo netfilter-persistent save
     ```

3. **Install Docker** (the convenience script handles arm64):
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker "$USER" && newgrp docker
   ```

4. **Point DNS at the box.** Create an `A` record for your domain → the
   instance's public IP, and *wait for it to resolve* (`dig +short your.domain`)
   before bringing Caddy up — the TLS challenge fails if the name doesn't
   resolve yet.

5. **Clone, configure, launch** (see [Bring-up](#bring-up) below).

> **ARM note:** every image in the stack (postgres, redis, qdrant, neo4j,
> caddy, ollama, `python:3.11-slim`) is multi-arch, so it all runs natively on
> aarch64. First build is slower than x86 because torch / sentence-transformers
> compile/download arm64 wheels — that's a one-time cost.

---

## Target B — DigitalOcean droplet (fallback)

A 4 GB / 2 vCPU droplet (x86) on the Student Pack's $200 credit lasts ~8 months.
It's tighter on RAM, so:

- Keep the default **Groq-only** path (don't enable `local-llm`; Ollama + a 3B
  won't fit).
- Cap Neo4j's memory so it doesn't OOM the box — add to the `neo4j` service
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

Ports 80/443 are open by default on DO; otherwise the bring-up is identical.

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
  in *both* the cloud firewall and the host iptables, and `ACME_EMAIL`/`DOMAIN`
  are set in `.env.production`. Watch `docker compose -f docker-compose.prod.yml
  logs -f caddy`. While testing repeatedly, switch to Let's Encrypt staging to
  avoid the 5-duplicate-certs-per-week rate limit — add `acme_ca
  https://acme-staging-v02.api.letsencrypt.org/directory` to the global options
  block in `Caddyfile`, then remove it for the real cert.
- **App loads but hangs on "Please wait…".** That's the Streamlit WebSocket
  being blocked. The prod frontend already disables Streamlit's own CORS/XSRF
  checks for exactly this reason; confirm Caddy is proxying `frontend:8501` and
  the frontend container is healthy.
- **A container is killed with exit code 137.** Out of memory — you're on a
  small VM with Ollama or an uncapped Neo4j. Stay on the Groq-only default and
  apply the Neo4j memory caps + swap from Target B.
