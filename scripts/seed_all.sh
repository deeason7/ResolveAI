#!/usr/bin/env bash
#
# Seed a fresh ResolveAI deployment from zero: migrations -> CFPB corpus ->
# embeddings -> knowledge graph.
#
# Safe to re-run. Every step is idempotent — alembic tracks applied revisions,
# the ingest is ON CONFLICT DO NOTHING, the vector upsert is last-writer-wins,
# and the graph writes are MERGE — so a run that dies halfway just resumes on
# the next invocation. That's the point: treat the VM as disposable and let this
# script rebuild its entire state.
#
# Run from the repo root on the host, with the prod stack already up:
#   docker compose -f docker-compose.prod.yml up -d --build
#   ./scripts/seed_all.sh
#
# Knobs (environment variables):
#   SKIP_DOWNLOAD=1            reuse an existing CSV instead of re-fetching ~1.8 GB
#   TOP_COMPANIES=N            companies to load into the graph (default 500)
#   RESET_GRAPH=1             wipe the graph before seeding
#   COMPOSE_FILE=docker-compose.yml   seed the dev stack instead of prod

set -euo pipefail

COMPOSE="docker compose -f ${COMPOSE_FILE:-docker-compose.prod.yml}"
CSV="/fine_tuning/data/raw/cfpb_200k.csv"
TOP_COMPANIES="${TOP_COMPANIES:-500}"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

# The whole script drives the api container, so fail fast (with the fix) if it
# isn't running rather than erroring five different ways downstream.
if ! $COMPOSE exec -T api true 2>/dev/null; then
	echo "The api container isn't running. Bring the stack up first:" >&2
	echo "  $COMPOSE up -d --build" >&2
	exit 1
fi

log "1/5  Database migrations (alembic upgrade head)"
$COMPOSE exec -T api alembic upgrade head

log "2/5  CFPB corpus download"
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
	echo "SKIP_DOWNLOAD set — leaving any existing CSV in place."
elif $COMPOSE exec -T api test -f "$CSV"; then
	echo "CSV already present at $CSV — skipping the ~1.8 GB download."
else
	$COMPOSE exec -T api python /scripts/download_cfpb_data.py \
		--output "$CSV" \
		--workdir /fine_tuning/data/raw/_cfpb_workdir
fi

log "3/5  Ingest into Postgres (idempotent: ON CONFLICT DO NOTHING)"
# No CLI wrapper exists for the ingest service, so call it inline. -T turns off
# the pseudo-TTY so the heredoc reaches python's stdin.
$COMPOSE exec -T api python - <<PY
import asyncio
from app.services.data_ingestion import ingest_cfpb_csv

r = asyncio.run(ingest_cfpb_csv("$CSV"))
print(f"ingest: read={r.rows_read} inserted={r.rows_inserted} "
      f"skipped={r.rows_skipped} in {r.elapsed_seconds:.1f}s")
PY

log "4/5  Embed narratives into Qdrant (the long pole — ~90 min on CPU for 200K)"
$COMPOSE exec -T api python /scripts/populate_vector_db.py

log "5/5  Seed the Neo4j knowledge graph (top ${TOP_COMPANIES} companies)"
if [ "${RESET_GRAPH:-0}" = "1" ]; then
	$COMPOSE exec -T api python /graph_seed/seed_graph.py --limit "$TOP_COMPANIES" --reset
else
	$COMPOSE exec -T api python /graph_seed/seed_graph.py --limit "$TOP_COMPANIES"
fi

log "Done. Corpus loaded — classify from the Workspace page or enqueue a batch."
