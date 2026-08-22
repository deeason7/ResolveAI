"""
Backfill the Qdrant `complaint_embeddings` collection from the complaints
table in Postgres.

Strategy:
  1. Stream complaints from Postgres in chunks (server-side cursor) so RAM
     stays flat regardless of table size.
  2. For each chunk, embed all narratives in one sentence-transformers
     batch — that's where the GPU-style speedup lives even on CPU
     (BLAS vectorization, fewer Python round-trips).
  3. Upsert the embedded chunk into Qdrant in one network call.
  4. Log progress every chunk; report wall-clock and rate at the end.

Usage (inside the api container):
    PYTHONPATH=/app python /scripts/populate_vector_db.py
    PYTHONPATH=/app python /scripts/populate_vector_db.py --limit 1000
    PYTHONPATH=/app python /scripts/populate_vector_db.py --chunk-size 512

Idempotent: re-running re-embeds and overwrites existing points (Qdrant upsert
is last-writer-wins per id). Safe to interrupt and resume — you'll just
re-process from the start, which costs CPU time but won't corrupt anything.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Make `app.*` importable when invoked as `python /scripts/populate_vector_db.py`.
# Inside the docker container `/app` is the bind-mount root for the backend.
_BACKEND_DIR = Path("/app")
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
else:
    # Local fallback: walk up from this file to find backend/
    _local = Path(__file__).resolve().parent.parent / "backend"
    if _local.exists():
        sys.path.insert(0, str(_local))

from app.database import AsyncSessionLocal  # noqa: E402
from app.models.complaint import Complaint  # noqa: E402
from app.services.embedder import embed_batch  # noqa: E402
from app.services.vector_store import ComplaintPoint, get_default_store  # noqa: E402
from sqlmodel import select  # noqa: E402  (after sys.path bootstrap above)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("populate_vector_db")

DEFAULT_CHUNK_SIZE = 256  # matches spec; also the embed batch size
NARRATIVE_PREVIEW_LEN = 200


def _build_payload(c: Complaint) -> dict:
    """Searchable + display fields stored alongside the vector."""
    preview = (c.narrative or "")[:NARRATIVE_PREVIEW_LEN]
    if c.narrative and len(c.narrative) > NARRATIVE_PREVIEW_LEN:
        preview += "…"
    return {
        "product": c.product,
        "company": c.company,
        "company_response": c.company_response,
        "state": c.state,
        "sentiment": c.sentiment,  # None until SLM has run
        "narrative_preview": preview,
    }


async def _process_chunk(
    rows: list[Complaint],
    store,
) -> tuple[int, int]:
    """Embed + upsert one chunk. Returns (embedded, skipped)."""
    valid = [r for r in rows if r.narrative and r.narrative.strip()]
    skipped = len(rows) - len(valid)
    if not valid:
        return 0, skipped

    texts = [r.narrative for r in valid]
    # Embedding is CPU-bound and sync — push it off the event loop so the
    # Postgres cursor can keep streaming the next page in the background.
    vectors = await asyncio.to_thread(embed_batch, texts, len(texts), False)

    points = [
        ComplaintPoint(
            complaint_id=r.id,
            embedding=v,
            payload=_build_payload(r),
        )
        for r, v in zip(valid, vectors, strict=True)
    ]
    store.upsert_batch(points)
    return len(valid), skipped


async def populate(chunk_size: int, limit: int | None) -> None:
    store = get_default_store()
    log.info(
        "starting backfill into collection=%s (chunk_size=%d, limit=%s)",
        store.collection_name,
        chunk_size,
        limit if limit is not None else "all",
    )

    started = time.monotonic()
    total_embedded = 0
    total_skipped = 0
    chunks = 0
    chunk: list[Complaint] = []

    async with AsyncSessionLocal() as session:
        stmt = select(Complaint).order_by(Complaint.id)
        if limit is not None:
            stmt = stmt.limit(limit)
        stmt = stmt.execution_options(yield_per=chunk_size)

        result = await session.stream_scalars(stmt)
        async for row in result:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                embedded, skipped = await _process_chunk(chunk, store)
                total_embedded += embedded
                total_skipped += skipped
                chunks += 1
                elapsed = time.monotonic() - started
                rate = total_embedded / elapsed if elapsed else 0
                log.info(
                    "chunk %d: embedded=%d skipped=%d total=%d elapsed=%.1fs rate=%.0f/s",
                    chunks,
                    embedded,
                    skipped,
                    total_embedded,
                    elapsed,
                    rate,
                )
                chunk = []

        if chunk:
            embedded, skipped = await _process_chunk(chunk, store)
            total_embedded += embedded
            total_skipped += skipped
            chunks += 1
            log.info(
                "final chunk %d: embedded=%d skipped=%d",
                chunks,
                embedded,
                skipped,
            )

    elapsed = time.monotonic() - started
    final_count = store.collection_count()
    log.info(
        "done: embedded=%d skipped=%d chunks=%d in %.1fs (%.0f rows/s); collection now has %d points",
        total_embedded,
        total_skipped,
        chunks,
        elapsed,
        total_embedded / elapsed if elapsed else 0,
        final_count,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Qdrant from the complaints table.")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per embed+upsert batch (default: {DEFAULT_CHUNK_SIZE}).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many rows. Default: all.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(populate(chunk_size=args.chunk_size, limit=args.limit))


if __name__ == "__main__":
    main()
