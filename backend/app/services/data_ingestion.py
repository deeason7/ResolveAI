"""
Bulk ingest a normalized CFPB CSV into the complaints table.

The downloader script writes a CSV with our internal column names so this
service can stay dumb about CFPB-specific quirks. We stream the file row by
row, accumulate batches of `batch_size`, and issue one multi-row INSERT per
batch using Postgres ON CONFLICT DO NOTHING (keyed on cfpb_complaint_id) so
re-running is safe.

Why this shape:
  - csv.DictReader is a generator → memory is bounded regardless of file size.
  - SQLAlchemy Core insert() with values=[...] becomes a single multi-row INSERT,
    which is ~50x faster than ORM session.add_all() for batches of this size.
  - ON CONFLICT DO NOTHING leans on the UNIQUE constraint added in
    migration c8a4d1f5e3b7 — idempotent without a SELECT-then-INSERT roundtrip.
"""

from __future__ import annotations

import csv
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import AsyncSessionLocal
from app.models.complaint import Complaint

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    rows_read: int
    rows_inserted: int
    rows_skipped: int
    batches: int
    elapsed_seconds: float


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    # CFPB ships either YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.sss
    try:
        return datetime.fromisoformat(value.replace("Z", ""))
    except ValueError:
        return None


def _row_to_complaint_dict(row: dict[str, str]) -> dict | None:
    narrative = (row.get("consumer_complaint_narrative") or "").strip()
    if not narrative:
        return None  # narrative is NOT NULL in the schema; skip rows without one

    return {
        "cfpb_complaint_id": (row.get("complaint_id") or "").strip() or None,
        "narrative": narrative,
        "product": (row.get("product") or "").strip() or None,
        "sub_product": (row.get("sub_product") or "").strip() or None,
        "issue": (row.get("issue") or "").strip() or None,
        "sub_issue": (row.get("sub_issue") or "").strip() or None,
        "company": (row.get("company") or "").strip() or None,
        "company_response": (row.get("company_response") or "").strip() or None,
        "state": (row.get("state") or "").strip()[:2] or None,
        "date_received": _parse_date(row.get("date_received")),
    }


@asynccontextmanager
async def _session():
    """Internal session manager — own transactions so we can commit per batch."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def _flush_batch(batch: list[dict]) -> int:
    """Insert one batch with ON CONFLICT DO NOTHING. Returns rows actually inserted."""
    if not batch:
        return 0

    stmt = (
        pg_insert(Complaint)
        .values(batch)
        .on_conflict_do_nothing(
            index_elements=["cfpb_complaint_id"],
        )
    )

    async with _session() as session:
        result = await session.execute(stmt)
        await session.commit()
        # rowcount on Postgres returns rows actually inserted (excludes conflicts)
        return result.rowcount or 0


async def ingest_cfpb_csv(
    csv_path: Path | str,
    batch_size: int = 10_000,
) -> IngestResult:
    """Stream a normalized CFPB CSV into the complaints table."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    log.info("starting ingest from %s (batch_size=%d)", path, batch_size)
    started = time.monotonic()

    rows_read = 0
    rows_inserted = 0
    rows_skipped = 0
    batches = 0
    batch: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_read += 1
            mapped = _row_to_complaint_dict(row)
            if mapped is None:
                rows_skipped += 1
                continue
            batch.append(mapped)

            if len(batch) >= batch_size:
                inserted = await _flush_batch(batch)
                rows_inserted += inserted
                batches += 1
                log.info(
                    "batch %d: read=%d inserted=%d (conflicts=%d) elapsed=%.1fs",
                    batches,
                    rows_read,
                    inserted,
                    len(batch) - inserted,
                    time.monotonic() - started,
                )
                batch = []

    if batch:
        inserted = await _flush_batch(batch)
        rows_inserted += inserted
        batches += 1
        log.info(
            "final batch %d: read=%d inserted=%d (conflicts=%d)",
            batches,
            rows_read,
            inserted,
            len(batch) - inserted,
        )

    elapsed = time.monotonic() - started
    log.info(
        "ingest complete: read=%d inserted=%d skipped=%d batches=%d in %.1fs",
        rows_read,
        rows_inserted,
        rows_skipped,
        batches,
        elapsed,
    )
    return IngestResult(
        rows_read=rows_read,
        rows_inserted=rows_inserted,
        rows_skipped=rows_skipped,
        batches=batches,
        elapsed_seconds=elapsed,
    )
