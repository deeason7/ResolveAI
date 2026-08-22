"""
Export and restore the irreplaceable half of the database.

The corpus is not the valuable part. Complaints come from the CFPB's public
bulk download and can be re-ingested from scratch; the vector and graph stores
are rebuilt from the complaints table by `populate_vector_db.py` and
`seed_graph.py`. What nothing can regenerate is the state the app produced:
accounts, the drafts the agent wrote, the LLM call log the observability
dashboard reads, and the audit trail. On this deployment that's a few hundred
rows against 30K complaints — small enough that JSONL is a perfectly good
archive format and no server-side tooling is needed.

Deliberately not pg_dump: the managed Postgres is v18, so pg_dump would have to
be v18 too on every machine that ever runs a backup. This talks to the database
through the same engine the app already uses.

Usage:
    python scripts/backup_state.py                       # -> backups/<utc-stamp>/
    python scripts/backup_state.py --out-dir /some/where
    python scripts/backup_state.py --include-complaints  # + the 30K corpus
    python scripts/backup_state.py --restore backups/20260820T190000Z

Restores are idempotent — rows whose primary key already exists are skipped, so
re-running never duplicates and never overwrites live data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path("/app")
if _BACKEND_DIR.exists() and str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
else:
    _local = Path(__file__).resolve().parent.parent / "backend"
    if _local.exists():
        sys.path.insert(0, str(_local))

import app.models  # noqa: E402,F401  — registers every table on SQLModel.metadata
import sqlalchemy as sa  # noqa: E402
from app.database import engine  # noqa: E402
from sqlmodel import SQLModel  # noqa: E402

logger = logging.getLogger("backup_state")

# Foreign keys point left to right, so this is also the restore order.
# complaints sits in the middle: it's the parent of the three child tables but
# is itself reproducible, which is why it's opt-in.
CORE_TABLES = ("users", "complaints", "resolutions", "llm_logs", "audit_logs", "complaint_labels")
REPRODUCIBLE = frozenset({"complaints"})

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1


def _tables(include_complaints: bool) -> list[sa.Table]:
    names = [t for t in CORE_TABLES if include_complaints or t not in REPRODUCIBLE]
    return [SQLModel.metadata.tables[n] for n in names]


def _to_json(value: Any) -> Any:
    """Make a driver-returned value JSON-safe without losing precision."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _from_json(column: sa.Column, value: Any) -> Any:
    """Rebuild the Python type a column expects from its JSON form.

    Driven by the column type rather than by guessing at the value: asyncpg is
    strict, and a str where a UUID belongs fails at bind time.
    """
    if value is None:
        return None
    kind = column.type
    if isinstance(kind, sa.Uuid):
        return uuid.UUID(value) if isinstance(value, str) else value
    if isinstance(kind, sa.DateTime):
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    if isinstance(kind, sa.Date):
        return date.fromisoformat(value) if isinstance(value, str) else value
    return value


async def _alembic_revision(conn: sa.ext.asyncio.AsyncConnection) -> str | None:
    """The schema version these rows were written against.

    A restore into a differently-migrated schema is the classic way to turn a
    backup into a corruption, so the revision travels with the data.
    """
    try:
        return await conn.scalar(sa.text("select version_num from alembic_version limit 1"))
    except sa.exc.SQLAlchemyError:
        return None


async def export_state(out_dir: Path, *, include_complaints: bool = False) -> dict:
    """Write one JSONL file per table plus a manifest. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    async with engine.connect() as conn:
        revision = await _alembic_revision(conn)
        for table in _tables(include_complaints):
            path = out_dir / f"{table.name}.jsonl"
            written = 0
            # Stream: the corpus is 30K rows and there's no reason to hold it.
            result = await conn.stream(sa.select(table))
            with path.open("w", encoding="utf-8") as fh:
                async for row in result:
                    payload = {k: _to_json(v) for k, v in row._mapping.items()}
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written += 1
            counts[table.name] = written
            logger.info("exported %-18s %6d rows", table.name, written)

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "alembic_revision": revision,
        "includes_complaints": include_complaints,
        "row_counts": counts,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _insert_ignoring_conflicts(table: sa.Table, rows: list[dict], dialect: str):
    """Build a dialect-appropriate idempotent INSERT.

    Both backends support it, they just spell it differently — and the generic
    Core insert supports neither, so branching here is the price of having the
    tests run on SQLite and production on Postgres.
    """
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert(table).values(rows).on_conflict_do_nothing()
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(table).values(rows).on_conflict_do_nothing()
    raise RuntimeError(f"restore not supported on dialect {dialect!r}")


async def restore_state(src_dir: Path, *, force: bool = False, batch_size: int = 500) -> dict:
    """Load an export back in. Existing primary keys are left untouched."""
    manifest_path = src_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {src_dir} — not a backup directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    inserted: dict[str, int] = {}
    async with engine.begin() as conn:
        current = await _alembic_revision(conn)
        recorded = manifest.get("alembic_revision")
        if recorded and current and recorded != current and not force:
            raise RuntimeError(
                f"schema mismatch: backup was taken at {recorded}, database is at {current}. "
                "Migrate to match, or pass --force if you know the tables are compatible."
            )

        dialect = conn.engine.dialect.name
        for table in _tables(include_complaints=True):
            path = src_dir / f"{table.name}.jsonl"
            if not path.exists():
                continue
            batch: list[dict] = []
            total = 0
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    batch.append({c.name: _from_json(c, raw.get(c.name)) for c in table.columns})
                    if len(batch) >= batch_size:
                        await conn.execute(_insert_ignoring_conflicts(table, batch, dialect))
                        total += len(batch)
                        batch = []
            if batch:
                await conn.execute(_insert_ignoring_conflicts(table, batch, dialect))
                total += len(batch)
            inserted[table.name] = total
            logger.info("restored %-18s %6d rows read", table.name, total)
    return inserted


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export or restore app-generated database state.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the export into (default: backups/<utc timestamp>).",
    )
    p.add_argument(
        "--include-complaints",
        action="store_true",
        help="Also export the complaints corpus (large, and re-ingestable from the CFPB).",
    )
    p.add_argument(
        "--restore",
        type=Path,
        default=None,
        metavar="DIR",
        help="Restore from a backup directory instead of exporting.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Restore even if the backup's alembic revision doesn't match the database.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.restore:
        counts = asyncio.run(restore_state(args.restore, force=args.force))
        logger.info("restore complete: %s", counts)
        return

    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent
        / "backups"
        / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    )
    manifest = asyncio.run(export_state(out_dir, include_complaints=args.include_complaints))
    logger.info("wrote %s", out_dir)
    logger.info("rows: %s", manifest["row_counts"])


if __name__ == "__main__":
    main()
