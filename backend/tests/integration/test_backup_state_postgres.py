"""Real PostgreSQL snapshot and restoration evidence using synthetic data."""

from __future__ import annotations

import asyncio
import time

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from tests.test_backup_state import backup_state, dump_rows, seed_rows

pytestmark = pytest.mark.postgres


async def test_default_backup_restores_into_fresh_migrated_postgres(postgres_database, tmp_path):
    await postgres_database.upgrade()
    engine = postgres_database.engine
    await seed_rows(engine)
    expected = await dump_rows(engine)
    archive = tmp_path / "complete"
    await backup_state.export_state(archive, db_engine=engine)

    await postgres_database.reset()
    await postgres_database.upgrade()
    started = time.perf_counter()
    inserted = await backup_state.restore_state(archive, db_engine=engine, batch_size=1)
    elapsed = time.perf_counter() - started

    assert inserted == {name: len(rows) for name, rows in expected.items()}
    assert await dump_rows(engine) == expected
    print(f"synthetic PostgreSQL restore: {sum(inserted.values())} rows in {elapsed:.3f}s")


async def test_export_uses_one_snapshot_during_concurrent_writes(pg_engine, tmp_path, monkeypatch):
    complaint_id = await seed_rows(pg_engine)
    expected = await dump_rows(pg_engine)
    snapshot_taken = asyncio.Event()
    writer_done = asyncio.Event()
    original_stream = AsyncConnection.stream
    first = True

    async def paused_stream(self, *args, **kwargs):
        nonlocal first
        # Introspection/revision SELECTs have already established the snapshot.
        if first:
            first = False
            assert await self.scalar(sa.text("SHOW transaction_isolation")) == "repeatable read"
            assert await self.scalar(sa.text("SHOW transaction_read_only")) == "on"
            snapshot_taken.set()
            await asyncio.wait_for(writer_done.wait(), timeout=10)
        return await original_stream(self, *args, **kwargs)

    monkeypatch.setattr(AsyncConnection, "stream", paused_stream)

    async def writer():
        await asyncio.wait_for(snapshot_taken.wait(), timeout=10)
        async with pg_engine.begin() as conn:
            await conn.execute(
                sa.text("UPDATE complaints SET narrative = :text WHERE id = :id"),
                {"text": "Changed after snapshot", "id": complaint_id},
            )
            await conn.execute(
                sa.text("UPDATE resolutions SET draft_text = :text WHERE complaint_id = :id"),
                {"text": "Changed after snapshot", "id": complaint_id},
            )
        writer_done.set()

    archive = tmp_path / "consistent"
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(writer())
        tasks.create_task(backup_state.export_state(archive, db_engine=pg_engine))
    manifest = backup_state._read_manifest(archive)
    for name, rows in expected.items():
        archived = list(backup_state._verified_rows(archive, manifest["tables"][name]))
        assert sorted(archived, key=lambda row: row["id"]) == rows
    assert (await dump_rows(pg_engine))["complaints"] != expected["complaints"]
