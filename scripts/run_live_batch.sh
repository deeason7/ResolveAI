#!/usr/bin/env bash
# Operator access to PostgreSQL jobs in an already-running stack.
# classify/resolve accept 1..500 cases per invocation; capacity rejection is atomic.
# Usage: run_live_batch.sh {classify [N]|resolve [N]|monitor|wait {classify|resolve} [minutes]}
set -euo pipefail
compose=(docker compose -f "${COMPOSE_FILE:-docker-compose.prod.yml}")
if [[ -n "${GPU_COMPOSE_FILE:-}" ]]; then
  compose+=(-f "$GPU_COMPOSE_FILE")
fi
"${compose[@]}" exec -T api python -m app.services.job_cli "$@"
