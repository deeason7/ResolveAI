#!/usr/bin/env bash
#
# Unattended single-box live run. Runs ON the GPU instance, from the repo root
# (~/Resolve_AI). Chains the whole pipeline end to end and writes the metrics:
#
#   stack up -> load model -> smoke test -> seed corpus -> classify -> wait
#   -> resolve -> wait -> harvest -> ~/metrics.txt
#
# Designed to run detached and survive your SSH session closing:
#   nohup ./scripts/box_bootstrap.sh > ~/bootstrap.log 2>&1 &
#   tail -f ~/bootstrap.log          # watch progress
#   cat ~/metrics.txt                # results when done
#
# Knobs (env vars):
#   SEED_ROWS   corpus size to seed     (default 20000)
#   CLASSIFY_N  complaints to classify  (default 10000)
#   RESOLVE_N   resolutions to draft    (default 500)
#   WORKERS     classification workers  (default 4, match OLLAMA_NUM_PARALLEL)
#   MAX_HOURS   dead-man's switch: schedule OS shutdown after N hours so a hung
#               run can't bleed credits. Unset = no watchdog. Only terminates if
#               the instance was launched with shutdown-behavior=terminate;
#               otherwise it stops the instance.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SEED_ROWS="${SEED_ROWS:-20000}"
CLASSIFY_N="${CLASSIFY_N:-10000}"
RESOLVE_N="${RESOLVE_N:-500}"
WORKERS="${WORKERS:-4}"

# OVERLAY selects the gpu (default) or cpu compose overlay. Exported as
# GPU_COMPOSE_FILE so run_live_batch.sh drives the same stack.
OVERLAY="${OVERLAY:-docker-compose.gpu.yml}"
export GPU_COMPOSE_FILE="$OVERLAY"
DC="docker compose -f docker-compose.prod.yml -f $OVERLAY --profile local-llm"
RUN="./scripts/run_live_batch.sh"
say() { printf '\n\033[1;36m== %s ==\033[0m %s\n' "$1" "$(date -u +%H:%M:%SZ)"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$1" >&2; exit 1; }

# 0. Optional credit watchdog.
if [ -n "${MAX_HOURS:-}" ]; then
  say "watchdog: scheduling shutdown in ${MAX_HOURS}h"
  sudo shutdown -h "+$(( MAX_HOURS * 60 ))" || echo "(could not schedule shutdown; continuing)"
fi

# 1. Preconditions: env file must exist with a Groq key and local model enabled.
say "preflight"
[ -f .env.production ] || die ".env.production missing — copy .env.production.example and fill secrets + GROQ_API_KEY first."
grep -q '^GROQ_API_KEY=.\+' .env.production || die "GROQ_API_KEY is empty in .env.production (needed for the resolution agent)."
if grep -qiE '^LLM_SKIP_LOCAL=(true|1)' .env.production; then
  die "LLM_SKIP_LOCAL is true — that skips your fine-tuned model. Set it false/remove it."
fi
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(no nvidia-smi on host — make sure this is a GPU instance)"

# 2. Bring up only what the batch needs (datastores + api + workers + GPU
#    ollama). Caddy/frontend are skipped — a headless metrics run needs no TLS
#    or UI, and a placeholder DOMAIN would just spam ACME failures.
say "compose up (build)"
$DC up -d --build postgres redis qdrant neo4j ollama api classification-worker resolution-worker

# 3. Load the fine-tuned model into Ollama from the mounted GGUF + Modelfile.
say "ollama create resolveai-sentiment"
for i in $(seq 1 30); do
  $DC exec -T ollama ollama list >/dev/null 2>&1 && break
  echo "waiting for ollama... ($i)"; sleep 5
done
$DC exec -T ollama ollama create resolveai-sentiment -f /models/Modelfile

# 4. Smoke test — don't burn GPU-hours on a model that won't emit JSON.
say "smoke test"
SMOKE="$($DC exec -T ollama ollama run resolveai-sentiment \
  'COMPLAINT: my credit card was charged twice for the same purchase' 2>/dev/null || true)"
echo "$SMOKE"
echo "$SMOKE" | grep -qi 'sentiment' || die "smoke test did not return a classification — check the model/template before spending GPU time."

# 5. Seed the corpus (migrations -> CFPB sample -> embeddings -> graph).
say "seed corpus (SEED_ROWS=$SEED_ROWS)"
SEED_ROWS="$SEED_ROWS" ./scripts/seed_all.sh

# 6. Scale classification consumers to match Ollama's parallelism, then run.
say "scale classification-worker=$WORKERS"
$DC up -d --scale classification-worker="$WORKERS"

say "classify $CLASSIFY_N"
$RUN classify "$CLASSIFY_N"
$RUN wait classify 720      # block until the classification stream drains

say "resolve $RESOLVE_N"
$RUN resolve "$RESOLVE_N"
$RUN wait resolve 720       # block until the resolution stream drains

# 7. Harvest the grounded metrics.
say "harvest -> ~/metrics.txt"
$DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < scripts/harvest_metrics.sql | tee "$HOME/metrics.txt"

say "DONE"
echo "Metrics written to ~/metrics.txt. Send that file back to rebuild the resume bullets."
echo "When finished, TERMINATE the instance to stop charges."
