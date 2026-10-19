"""Complete application-state export and empty-target restore.

Version 2 archives include every registered application table and a checksummed
manifest. Use a database migrated to the recorded revision for restore; merges,
partial archives and schema overrides are intentionally unsupported. The CLI
reads its target from an explicit environment variable, never a dotenv file.

Examples:
    python scripts/backup_state.py --out-dir backups/restore-drill
    python scripts/backup_state.py --database-url-env RESTORE_DATABASE_URL \
        --restore backups/restore-drill

Archives contain credentials hashes and complaint text. Keep them private.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

_LOCAL_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_LOCAL_BACKEND if _LOCAL_BACKEND.exists() else Path("/app")))

import app.models  # noqa: E402,F401
import sqlalchemy as sa  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine  # noqa: E402
from sqlmodel import SQLModel  # noqa: E402

logger = logging.getLogger("backup_state")
MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 2


class BackupError(ValueError):
    """An unsafe/incompatible archive or target, with a non-sensitive message."""


def _to_json(value: Any) -> Any:
    if isinstance(value, uuid.UUID | Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _from_json(column: sa.Column, value: Any) -> Any:
    if value is None:
        return None
    kind = column.type
    if isinstance(kind, sa.Uuid):
        return uuid.UUID(value)
    if isinstance(kind, sa.DateTime):
        return datetime.fromisoformat(value)
    if isinstance(kind, sa.Date):
        return date.fromisoformat(value)
    if isinstance(kind, sa.Numeric) and not isinstance(kind, sa.Float):
        return Decimal(value)
    return value


async def _alembic_revision(conn: AsyncConnection) -> str:
    exists = await conn.run_sync(lambda sync: sa.inspect(sync).has_table("alembic_version"))
    if not exists:
        raise BackupError("database has no Alembic revision; migrate the target explicitly")
    revisions = (
        (await conn.execute(sa.text("SELECT version_num FROM alembic_version"))).scalars().all()
    )
    if len(revisions) != 1 or not isinstance(revisions[0], str) or not revisions[0]:
        raise BackupError("database must have exactly one recorded Alembic revision")
    return revisions[0]


async def _tables(conn: AsyncConnection) -> list[sa.Table]:
    """Reflect actual columns; refuse an unmapped table rather than omit it."""
    expected = set(SQLModel.metadata.tables)
    actual = set(await conn.run_sync(lambda sync: sa.inspect(sync).get_table_names()))
    if actual != expected | {"alembic_version"}:
        raise BackupError("database table inventory differs from this application version")
    metadata = sa.MetaData()
    await conn.run_sync(lambda sync: metadata.reflect(sync, only=sorted(expected)))
    return list(metadata.sorted_tables)


def _schema(table: sa.Table, dialect: sa.engine.Dialect) -> dict:
    return {
        "columns": [
            {
                "name": c.name,
                "type": c.type.compile(dialect=dialect),
                "nullable": c.nullable,
                "primary_key": c.primary_key,
            }
            for c in table.columns
        ],
        "foreign_keys": sorted([[fk.parent.name, fk.target_fullname] for fk in table.foreign_keys]),
        "unique_constraints": sorted(
            [
                sorted(c.name for c in constraint.columns)
                for constraint in table.constraints
                if isinstance(constraint, sa.UniqueConstraint)
            ]
        ),
        "unique_indexes": sorted(
            sorted(c.name for c in index.columns) for index in table.indexes if index.unique
        ),
    }


async def export_state(out_dir: Path, *, db_engine: AsyncEngine) -> dict:
    """Publish a complete manifest last; failed exports never have that marker."""
    out_dir = Path(out_dir)
    # Reserve the target exclusively. Never overwrite a previous archive.
    out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.partial-", dir=out_dir.parent))
    entries: dict[str, dict] = {}
    counts: dict[str, int] = {}
    async with db_engine.connect() as conn:
        if conn.dialect.name == "postgresql":
            conn = await conn.execution_options(isolation_level="REPEATABLE READ")
        elif conn.dialect.name != "sqlite":
            raise BackupError("unsupported database dialect")
        async with conn.begin():
            if conn.dialect.name == "postgresql":
                await conn.execute(sa.text("SET TRANSACTION READ ONLY"))
            revision = await _alembic_revision(conn)
            tables = await _tables(conn)
            for table in tables:
                path = staging / f"{table.name}.jsonl"
                digest = hashlib.sha256()
                count = 0
                result = await conn.stream(sa.select(table))
                try:
                    with path.open("xb") as stream:
                        path.chmod(0o600)
                        async for row in result:
                            payload = {k: _to_json(v) for k, v in row._mapping.items()}
                            line = (
                                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
                            ).encode()
                            stream.write(line)
                            digest.update(line)
                            count += 1
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    await result.close()
                counts[table.name] = count
                entries[table.name] = {
                    "file": path.name,
                    "rows": count,
                    "sha256": digest.hexdigest(),
                    "schema": _schema(table, conn.dialect),
                }
            dialect = conn.dialect.name

    manifest = {
        "format_version": FORMAT_VERSION,
        "complete": True,
        "created_at": datetime.now(UTC).isoformat(),
        "alembic_revision": revision,
        "dialect": dialect,
        "includes_complaints": True,
        "row_counts": counts,
        "tables": entries,
    }
    pending_manifest = staging / f"{MANIFEST_NAME}.partial"
    with pending_manifest.open("x", encoding="utf-8") as stream:
        pending_manifest.chmod(0o600)
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Replace only our reserved empty directory; a nonempty destination fails.
    staging.rename(out_dir)
    (out_dir / pending_manifest.name).rename(out_dir / MANIFEST_NAME)
    logger.info("export complete: %s", counts)
    return manifest


def _read_manifest(src_dir: Path) -> dict:
    path = src_dir / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise BackupError("no completed manifest; not a complete backup directory")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["format_version"] != FORMAT_VERSION or manifest["complete"] is not True:
            raise BackupError(
                "unsupported/incomplete backup format; legacy archives need separate recovery"
            )
        if manifest["includes_complaints"] is not True:
            raise BackupError("backup omits complaints")
        expected = set(SQLModel.metadata.tables)
        if set(manifest["tables"]) != expected or set(manifest["row_counts"]) != expected:
            raise BackupError("backup table inventory differs from this application version")
        if not isinstance(manifest["alembic_revision"], str) or not manifest["alembic_revision"]:
            raise BackupError("backup has no schema revision")
        for name, entry in manifest["tables"].items():
            if entry["file"] != f"{name}.jsonl":
                raise BackupError("invalid backup filename")
            if type(entry["rows"]) is not int or entry["rows"] < 0:
                raise BackupError("invalid backup row count")
            if manifest["row_counts"][name] != entry["rows"]:
                raise BackupError("inconsistent backup row counts")
            if not isinstance(entry["schema"]["columns"], list):
                raise BackupError("invalid backup schema")
            if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
                raise BackupError("invalid backup checksum")
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise BackupError("invalid manifest structure") from error
    return manifest


def _verified_rows(src_dir: Path, entry: dict):
    """Verify on preflight AND while restoring, detecting edits between passes."""
    path = src_dir / entry["file"]
    if not path.is_file() or path.is_symlink():
        raise BackupError(f"missing required regular file: {entry['file']}")
    expected_columns = {c["name"] for c in entry["schema"]["columns"]}
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != expected_columns:
                    raise BackupError(f"row columns do not match schema: {entry['file']}")
                count += 1
                yield row
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BackupError(f"invalid JSONL in {entry['file']}") from error
    if count != entry["rows"] or digest.hexdigest() != entry["sha256"]:
        raise BackupError(f"checksum/row-count mismatch: {entry['file']}")


async def restore_state(
    src_dir: Path, *, db_engine: AsyncEngine, batch_size: int = 500
) -> dict[str, int]:
    """Restore all rows in one transaction; refuse nonempty or incompatible targets."""
    if not 1 <= batch_size <= 1000:
        raise BackupError("batch size must be between 1 and 1000")
    src_dir = Path(src_dir)
    manifest = _read_manifest(src_dir)
    # Validate every file before a database write is possible.
    for entry in manifest["tables"].values():
        for _ in _verified_rows(src_dir, entry):
            pass

    inserted: dict[str, int] = {}
    async with db_engine.begin() as conn:
        if manifest.get("dialect") != conn.dialect.name:
            raise BackupError("archive and target database dialect differ")
        current = await _alembic_revision(conn)
        if current != manifest["alembic_revision"]:
            raise BackupError("schema mismatch: migrate an empty target to the recorded revision")
        tables = await _tables(conn)
        if conn.dialect.name == "postgresql":
            names = [conn.dialect.identifier_preparer.format_table(t) for t in tables]
            await conn.execute(
                sa.text(
                    "LOCK TABLE alembic_version, " + ", ".join(names) + " IN ACCESS EXCLUSIVE MODE"
                )
            )
        for table in tables:
            entry = manifest["tables"][table.name]
            if entry["schema"] != _schema(table, conn.dialect):
                raise BackupError(f"schema mismatch for table {table.name}")
            if await conn.scalar(sa.select(sa.exists().select_from(table))):
                raise BackupError(
                    f"restore requires empty application tables; {table.name} is not empty"
                )
            # Validate conversions for all rows before starting inserts.
            for raw in _verified_rows(src_dir, entry):
                try:
                    for c in table.columns:
                        _from_json(c, raw[c.name])
                except (ValueError, TypeError) as error:
                    raise BackupError(f"invalid typed value in {entry['file']}") from error

        for table in tables:
            batch = []
            total = 0
            for raw in _verified_rows(src_dir, manifest["tables"][table.name]):
                batch.append({c.name: _from_json(c, raw[c.name]) for c in table.columns})
                if len(batch) >= batch_size:
                    await conn.execute(sa.insert(table), batch)
                    total += len(batch)
                    batch.clear()
            if batch:
                await conn.execute(sa.insert(table), batch)
                total += len(batch)
            inserted[table.name] = total
    # A conflict aborts the entire transaction; counts are reported only after commit.
    logger.info("restore committed: %s", inserted)
    return inserted


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--out-dir", type=Path, help="New directory for a complete export.")
    group.add_argument("--restore", type=Path, help="Complete version 2 archive to restore.")
    parser.add_argument(
        "--database-url-env",
        default="DATABASE_URL",
        help="Explicit target URL environment variable; dotenv files are not loaded.",
    )
    parser.add_argument(
        "--require-ssl",
        action="store_true",
        help="Require verified TLS for the PostgreSQL connection.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    raw_url = os.environ.get(args.database_url_env)
    if not raw_url:
        raise BackupError(f"set the {args.database_url_env} environment variable explicitly")
    try:
        url = sa.engine.make_url(raw_url)
    except sa.exc.ArgumentError:
        raise BackupError("invalid database URL") from None
    if url.drivername != "postgresql+asyncpg":
        raise BackupError("CLI requires a postgresql+asyncpg database URL")
    db_engine = create_async_engine(url, connect_args={"ssl": True} if args.require_ssl else {})
    try:
        if args.restore:
            await restore_state(args.restore, db_engine=db_engine)
        else:
            out_dir = args.out_dir or Path("backups") / datetime.now(UTC).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            await export_state(out_dir, db_engine=db_engine)
            logger.info("archive: %s", out_dir)
    finally:
        await db_engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        asyncio.run(_run(_parse_args()))
    except BackupError as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
    except Exception as error:
        # Database exceptions may include credentials or bound complaint text.
        logger.error(
            "backup/restore failed (%s); target was not reported successful", type(error).__name__
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
