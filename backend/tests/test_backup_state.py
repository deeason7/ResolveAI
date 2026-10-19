"""Complete backup tests on isolated SQLite, with FK enforcement.

PostgreSQL snapshot and real-migration recovery tests live in integration/.
No fixture here loads production configuration or resets the application engine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backup_state.py"
spec = importlib.util.spec_from_file_location("backup_state", SCRIPT_PATH)
backup_state = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backup_state
spec.loader.exec_module(backup_state)


@pytest_asyncio.fixture
async def backup_engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'backup-test.db'}")
    async with engine.begin() as conn:
        await conn.execute(sa.text("PRAGMA foreign_keys=ON"))
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        await conn.execute(sa.text("INSERT INTO alembic_version VALUES ('test-schema-v1')"))
    try:
        yield engine
    finally:
        await engine.dispose()


async def seed_rows(engine):
    from app.models.audit_log import AuditLog
    from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
    from app.models.complaint_label import ComplaintLabel
    from app.models.llm_log import LLMLog
    from app.models.resolution import GuardrailStatus, Resolution
    from app.models.user import User, UserRole

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        user = User(
            email="reviewer@example.test",
            full_name="Synthetic reviewer",
            hashed_password="synthetic-not-a-password-hash",
            role=UserRole.analyst,
        )
        historic = Complaint(
            cfpb_complaint_id="synthetic-001",
            narrative="Synthetic duplicate charge.",
            status=ComplaintStatus.resolved,
            sentiment="negative",
            urgency=5,
            classification_outcome=ClassificationState.unavailable,
            classification_source="manual",
            classification_stale=True,
            classification_attempted_at=datetime(2026, 1, 3),
            classified_at=datetime(2026, 1, 2),
            classification_error="providers_unavailable",
            classification_latency_ms=1501,
            classification_was_fallback=False,
            company="Example Bank",
            date_received=datetime(2026, 1, 2),
        )
        live = Complaint(
            narrative="Synthetic live submission.",
            status=ComplaintStatus.classified,
            classification_outcome=ClassificationState.succeeded,
            classification_source="stub",
            classification_was_fallback=True,
        )
        session.add_all([user, historic, live])
        await session.flush()
        session.add_all(
            [
                Resolution(
                    complaint_id=historic.id,
                    version=1,
                    draft_text="Exact reviewed text — preserved.",
                    guardrail_status=GuardrailStatus.passed,
                    guardrail_violations=[],
                ),
                ComplaintLabel(
                    complaint_id=historic.id,
                    label_source="synthetic-fixture",
                    sentiment="negative",
                    intent="dispute",
                    urgency=5,
                    key_entities=[{"type": "company", "value": "Example Bank"}],
                    reasoning="Synthetic label",
                ),
                LLMLog(
                    complaint_id=live.id,
                    operation="classification",
                    model_used="stub",
                    provider="stub",
                    latency_ms=12,
                ),
                AuditLog(
                    user_id=user.id,
                    event="synthetic_review",
                    metadata_={"case_id": str(historic.id), "revision": 1},
                ),
            ]
        )
        await session.commit()
    return historic.id


async def dump_rows(engine):
    async with engine.connect() as conn:
        tables = await backup_state._tables(conn)
        result = {}
        for table in tables:
            rows = (await conn.execute(sa.select(table).order_by(table.c.id))).mappings()
            result[table.name] = [
                {key: backup_state._to_json(value) for key, value in row.items()} for row in rows
            ]
        return result


async def empty_rows(engine):
    async with engine.begin() as conn:
        for table in reversed(await backup_state._tables(conn)):
            await conn.execute(sa.delete(table))


def rewrite_manifest(archive, mutate):
    path = archive / backup_state.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def replace_rows(archive, table_name, rows):
    path = archive / f"{table_name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def update(manifest):
        entry = manifest["tables"][table_name]
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["rows"] = len(rows)
        manifest["row_counts"][table_name] = len(rows)

    rewrite_manifest(archive, update)


async def test_default_export_contains_complete_state(backup_engine, tmp_path):
    await seed_rows(backup_engine)
    archive = tmp_path / "archive"
    manifest = await backup_state.export_state(archive, db_engine=backup_engine)
    assert manifest["format_version"] == 2
    assert manifest["includes_complaints"] is True
    assert manifest["row_counts"]["complaints"] == 2
    assert set(manifest["tables"]) == set(SQLModel.metadata.tables)
    assert manifest["alembic_revision"] == "test-schema-v1"
    assert archive.stat().st_mode & 0o777 == 0o700
    for entry in manifest["tables"].values():
        path = archive / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_mode & 0o777 == 0o600


async def test_default_roundtrip_preserves_exact_ids_text_and_relationships(
    backup_engine, tmp_path
):
    await seed_rows(backup_engine)
    expected = await dump_rows(backup_engine)
    archive = tmp_path / "archive"
    await backup_state.export_state(archive, db_engine=backup_engine)
    await empty_rows(backup_engine)
    inserted = await backup_state.restore_state(archive, db_engine=backup_engine, batch_size=1)
    assert inserted == {name: len(rows) for name, rows in expected.items()}
    assert await dump_rows(backup_engine) == expected


async def test_nonempty_target_refused_without_overwriting(backup_engine, tmp_path):
    await seed_rows(backup_engine)
    before = await dump_rows(backup_engine)
    archive = tmp_path / "archive"
    await backup_state.export_state(archive, db_engine=backup_engine)
    with pytest.raises(backup_state.BackupError, match="requires empty"):
        await backup_state.restore_state(archive, db_engine=backup_engine)
    assert await dump_rows(backup_engine) == before


@pytest.mark.parametrize("damage", ["missing", "checksum", "legacy", "schema", "revision"])
async def test_preflight_rejects_damage_before_insert(backup_engine, tmp_path, damage):
    await seed_rows(backup_engine)
    archive = tmp_path / "archive"
    await backup_state.export_state(archive, db_engine=backup_engine)
    await empty_rows(backup_engine)
    if damage == "missing":
        (archive / "complaints.jsonl").unlink()
    elif damage == "checksum":
        with (archive / "complaints.jsonl").open("ab") as stream:
            stream.write(b"\n")
    elif damage == "legacy":
        rewrite_manifest(archive, lambda m: m.update(format_version=1))
    elif damage == "revision":
        rewrite_manifest(archive, lambda m: m.update(alembic_revision="different"))
    else:
        rewrite_manifest(
            archive, lambda m: m["tables"]["users"]["schema"]["columns"][0].update(type="BLOB")
        )
    with pytest.raises(backup_state.BackupError):
        await backup_state.restore_state(archive, db_engine=backup_engine)
    assert all(not rows for rows in (await dump_rows(backup_engine)).values())


@pytest.mark.parametrize("conflict", ["duplicate", "foreign_key"])
async def test_constraint_failure_rolls_back_entire_restore(backup_engine, tmp_path, conflict):
    await seed_rows(backup_engine)
    archive = tmp_path / "archive"
    await backup_state.export_state(archive, db_engine=backup_engine)
    await empty_rows(backup_engine)
    rows = [json.loads(line) for line in (archive / "resolutions.jsonl").read_text().splitlines()]
    if conflict == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["complaint_id"] = str(uuid.uuid4())
    replace_rows(archive, "resolutions", rows)
    with pytest.raises(sa.exc.IntegrityError):
        await backup_state.restore_state(archive, db_engine=backup_engine, batch_size=1)
    assert all(not rows for rows in (await dump_rows(backup_engine)).values())


async def test_failed_export_has_no_completion_marker(backup_engine, tmp_path, monkeypatch):
    await seed_rows(backup_engine)
    archive = tmp_path / "archive"

    def interrupted(_):
        raise RuntimeError("synthetic interrupted export")

    monkeypatch.setattr(backup_state, "_to_json", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        await backup_state.export_state(archive, db_engine=backup_engine)
    assert not (archive / backup_state.MANIFEST_NAME).exists()
    with pytest.raises(backup_state.BackupError, match="no completed manifest"):
        await backup_state.restore_state(archive, db_engine=backup_engine)


async def test_existing_archive_never_overwritten(backup_engine, tmp_path):
    archive = tmp_path / "archive"
    await backup_state.export_state(archive, db_engine=backup_engine)
    before = (archive / backup_state.MANIFEST_NAME).read_bytes()
    with pytest.raises(FileExistsError):
        await backup_state.export_state(archive, db_engine=backup_engine)
    assert (archive / backup_state.MANIFEST_NAME).read_bytes() == before


async def test_unmapped_table_cannot_be_silently_omitted(backup_engine, tmp_path):
    async with backup_engine.begin() as conn:
        await conn.execute(sa.text("CREATE TABLE unexpected_state (id INTEGER)"))
    with pytest.raises(backup_state.BackupError, match="inventory"):
        await backup_state.export_state(tmp_path / "archive", db_engine=backup_engine)


async def test_revision_required(backup_engine, tmp_path):
    async with backup_engine.begin() as conn:
        await conn.execute(sa.text("DROP TABLE alembic_version"))
    with pytest.raises(backup_state.BackupError, match="no Alembic revision"):
        await backup_state.export_state(tmp_path / "archive", db_engine=backup_engine)


async def test_schema_guard_detects_missing_unique_index(backup_engine, tmp_path):
    archive = tmp_path / "archive"
    await backup_state.export_state(archive, db_engine=backup_engine)
    async with backup_engine.begin() as conn:
        await conn.execute(sa.text("DROP INDEX ix_users_email"))
    with pytest.raises(backup_state.BackupError, match="schema mismatch for table users"):
        await backup_state.restore_state(archive, db_engine=backup_engine)


def test_unsafe_legacy_cli_flags_removed():
    for flag in ["--force", "--include-complaints"]:
        with pytest.raises(SystemExit):
            backup_state._parse_args([flag])
