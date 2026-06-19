#!/usr/bin/env bash
#
# Drive a live classification / resolution batch against an already-seeded,
# running stack. Operator path: enqueues straight onto the Redis streams from
# inside the api container (no auth, no HTTP rate limit), reusing the exact
# pending / escalated selection the Workspace routes use.
#
# Prereqs:
#   - stack up:        docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml --profile local-llm up -d
#   - corpus seeded:   SEED_ROWS=20000 ./scripts/seed_all.sh
#   - model loaded:    docker compose ... exec ollama ollama create resolveai-sentiment -f /models/Modelfile
#
# Usage:
#   ./scripts/run_live_batch.sh classify [N]   # enqueue up to N pending complaints (default 10000)
#   ./scripts/run_live_batch.sh resolve  [N]   # enqueue up to N escalated complaints (default 500)
#   ./scripts/run_live_batch.sh monitor        # status counts + stream depths
#
# Run `classify` first; watch `monitor` until the classification stream drains
# and you have `escalated` complaints; then run `resolve`.
set -euo pipefail

COMPOSE="docker compose -f ${COMPOSE_FILE:-docker-compose.prod.yml} -f ${GPU_COMPOSE_FILE:-docker-compose.gpu.yml}"
CMD="${1:-}"
N="${2:-}"

case "$CMD" in
  classify)
    N="${N:-10000}"
    $COMPOSE exec -T api python - "$N" <<'PY'
import asyncio, sys
from sqlmodel import select
from app.database import AsyncSessionLocal
from app.core.deps import get_default_redis
from app.models.complaint import Complaint, ComplaintStatus
from app.workers.classification_worker import enqueue_complaint

async def main(n: int) -> None:
    redis = get_default_redis()
    async with AsyncSessionLocal() as session:
        ids = (
            await session.exec(
                select(Complaint.id)
                .where(Complaint.status == ComplaintStatus.pending)
                .limit(n)
            )
        ).all()
    for cid in ids:
        await enqueue_complaint(redis, cid)
    print(f"enqueued {len(ids)} complaints for classification")

asyncio.run(main(int(sys.argv[1])))
PY
    ;;

  resolve)
    N="${N:-500}"
    $COMPOSE exec -T api python - "$N" <<'PY'
import asyncio, sys
from datetime import datetime
from sqlmodel import select
from app.database import AsyncSessionLocal
from app.core.deps import get_default_redis
from app.models.complaint import Complaint, ComplaintStatus
from app.workers.resolution_worker import enqueue_resolution

async def main(n: int) -> None:
    redis = get_default_redis()
    async with AsyncSessionLocal() as session:
        complaints = (
            await session.exec(
                select(Complaint)
                .where(Complaint.status == ComplaintStatus.escalated)
                .limit(n)
            )
        ).all()
        # Flip escalated -> agent_triggered (mirrors the Workspace route) so a
        # re-run can't re-select and double-draft the same complaint.
        for c in complaints:
            c.status = ComplaintStatus.agent_triggered
            c.updated_at = datetime.utcnow()
            session.add(c)
        await session.flush()
        for c in complaints:
            await enqueue_resolution(redis, c.id)
        await session.commit()
    print(f"enqueued {len(complaints)} complaints for resolution")

asyncio.run(main(int(sys.argv[1])))
PY
    ;;

  monitor)
    $COMPOSE exec -T api python - <<'PY'
import asyncio
from sqlmodel import select, func
from app.database import AsyncSessionLocal
from app.core.deps import get_default_redis
from app.models.complaint import Complaint
from app.config import settings

async def main() -> None:
    async with AsyncSessionLocal() as s:
        rows = (
            await s.exec(select(Complaint.status, func.count()).group_by(Complaint.status))
        ).all()
    print("== complaint status ==")
    for st, n in rows:
        print(f"  {getattr(st, 'value', st):16} {n}")
    redis = get_default_redis()
    print("== stream depth ==")
    for q in (settings.classification_queue, settings.resolution_queue):
        try:
            print(f"  {q}: {await redis.xlen(q)} msgs")
        except Exception as exc:  # noqa: BLE001 - monitor must never crash
            print(f"  {q}: {exc}")

asyncio.run(main())
PY
    ;;

  wait)
    # Block until a stream's consumer-group backlog (undelivered lag + unacked
    # pending) hits zero. WHICH is "classify" or "resolve"; TIMEOUT in minutes.
    WHICH="${2:-classify}"
    TIMEOUT_MIN="${3:-720}"
    $COMPOSE exec -T api python - "$WHICH" "$TIMEOUT_MIN" <<'PY'
import asyncio, sys, time
from app.core.deps import get_default_redis
from app.config import settings

stream = {
    "classify": settings.classification_queue,
    "resolve": settings.resolution_queue,
}[sys.argv[1]]
timeout_s = int(sys.argv[2]) * 60

async def backlog(redis) -> int | None:
    try:
        groups = await redis.xinfo_groups(stream)
    except Exception:
        return None  # group not created yet — worker hasn't started consuming
    total = 0
    for g in groups:
        total += int(g.get("lag") or 0) + int(g.get("pending") or 0)
    return total

async def main() -> int:
    redis = get_default_redis()
    start = time.time()
    clear_reads = 0
    while True:
        b = await backlog(redis)
        elapsed = int(time.time() - start)
        print(f"[{elapsed:5d}s] {stream} backlog={b}", flush=True)
        if b == 0:
            clear_reads += 1
            if clear_reads >= 2:   # two consecutive empties => settled
                print("drained")
                return 0
        else:
            clear_reads = 0
        if time.time() - start > timeout_s:
            print("TIMEOUT waiting for drain", flush=True)
            return 1
        await asyncio.sleep(20)

raise SystemExit(asyncio.run(main()))
PY
    ;;

  *)
    echo "usage: $0 {classify [N] | resolve [N] | wait {classify|resolve} [timeout_min] | monitor}" >&2
    exit 2
    ;;
esac
