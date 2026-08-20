"""
Tests for the state export/restore script.

A backup nobody has restored is a guess, so these drive the round trip against
a real (SQLite) database rather than asserting on the files alone: export,
wipe, restore, compare. The script is loaded via importlib because `scripts/`
isn't an installed package — the same approach test_label_prepare.py uses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "backup_state.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("backup_state", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backup_state"] = mod
    spec.loader.exec_module(mod)
    return mod


backup_state = _load_script()


@pytest_asyncio.fixture()
async def fresh_db():
    import app.models  # noqa: F401 — registers tables
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def _seed(factory, *, users: int = 2, resolutions: int = 1) -> dict:
    """Insert a user + complaint + resolution + llm_log + audit_log."""
    from app.models.complaint import Complaint, ComplaintStatus
    from app.models.llm_log import LLMLog
    from app.models.resolution import Resolution
    from app.models.user import User

    ids: dict = {"users": [], "complaints": [], "resolutions": []}
    async with factory() as s:
        for i in range(users):
            u = User(
                email=f"backup{i}@example.com",
                full_name=f"Backup {i}",
                hashed_password="not-a-real-hash",
            )
            s.add(u)
            ids["users"].append(u.id)

        c = Complaint(
            cfpb_complaint_id=f"BK-{uuid.uuid4().hex[:8]}",
            narrative="They charged me twice for the same payment.",
            product="Mortgage",
            issue="Trouble during payment process",
            company="ACME BANK",
            state="TX",
            status=ComplaintStatus.classified,
        )
        s.add(c)
        await s.flush()
        ids["complaints"].append(c.id)

        for v in range(resolutions):
            r = Resolution(
                complaint_id=c.id,
                version=v + 1,
                draft_text="Draft text for the response.",
                guardrail_violations=["structural"],
            )
            s.add(r)
            ids["resolutions"].append(r.id)

        s.add(
            LLMLog(
                complaint_id=c.id,
                operation="classification",
                model_used="openai/gpt-oss-120b",
                provider="groq",
                prompt_tokens=100,
                completion_tokens=20,
                latency_ms=350,
                cost_usd=0.0001,
                was_fallback=False,
            )
        )
        await s.commit()
    return ids


async def _count(factory, table_name: str) -> int:
    async with factory() as s:
        t = SQLModel.metadata.tables[table_name]
        return await s.scalar(sa.select(sa.func.count()).select_from(t))


class TestExport:
    async def test_writes_a_file_per_table_and_a_manifest(self, fresh_db, tmp_path):
        await _seed(fresh_db)
        manifest = await backup_state.export_state(tmp_path / "bk")

        assert (tmp_path / "bk" / backup_state.MANIFEST_NAME).exists()
        for name in ("users", "resolutions", "llm_logs", "audit_logs", "complaint_labels"):
            assert (tmp_path / "bk" / f"{name}.jsonl").exists()
        assert manifest["row_counts"]["users"] == 2
        assert manifest["row_counts"]["resolutions"] == 1

    async def test_skips_the_reproducible_corpus_by_default(self, fresh_db, tmp_path):
        await _seed(fresh_db)
        manifest = await backup_state.export_state(tmp_path / "bk")

        assert "complaints" not in manifest["row_counts"]
        assert not (tmp_path / "bk" / "complaints.jsonl").exists()
        assert manifest["includes_complaints"] is False

    async def test_includes_the_corpus_on_request(self, fresh_db, tmp_path):
        await _seed(fresh_db)
        manifest = await backup_state.export_state(tmp_path / "bk", include_complaints=True)

        assert manifest["row_counts"]["complaints"] == 1
        assert manifest["includes_complaints"] is True

    async def test_rows_are_json_serialisable_with_types_flattened(self, fresh_db, tmp_path):
        await _seed(fresh_db)
        await backup_state.export_state(tmp_path / "bk")

        line = (tmp_path / "bk" / "resolutions.jsonl").read_text().splitlines()[0]
        row = json.loads(line)
        uuid.UUID(row["id"])  # UUID survived as a parseable string
        datetime.fromisoformat(row["created_at"])  # datetime survived as ISO-8601
        assert row["guardrail_violations"] == ["structural"]  # JSON column kept its shape


class TestRoundTrip:
    async def test_export_wipe_restore_recovers_every_row(self, fresh_db, tmp_path):
        ids = await _seed(fresh_db)
        await backup_state.export_state(tmp_path / "bk", include_complaints=True)

        from app.database import engine

        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
            await conn.run_sync(SQLModel.metadata.create_all)
        assert await _count(fresh_db, "users") == 0

        await backup_state.restore_state(tmp_path / "bk")

        assert await _count(fresh_db, "users") == 2
        assert await _count(fresh_db, "complaints") == 1
        assert await _count(fresh_db, "resolutions") == 1
        assert await _count(fresh_db, "llm_logs") == 1

        async with fresh_db() as s:
            from app.models.user import User

            restored = await s.get(User, ids["users"][0])
            assert restored is not None
            assert restored.email == "backup0@example.com"
            assert restored.role.value == "analyst"

    async def test_restore_is_idempotent(self, fresh_db, tmp_path):
        await _seed(fresh_db)
        await backup_state.export_state(tmp_path / "bk", include_complaints=True)

        await backup_state.restore_state(tmp_path / "bk")
        await backup_state.restore_state(tmp_path / "bk")

        assert await _count(fresh_db, "users") == 2  # not 4

    async def test_restore_never_overwrites_a_live_row(self, fresh_db, tmp_path):
        ids = await _seed(fresh_db)
        await backup_state.export_state(tmp_path / "bk")

        from app.models.user import User

        async with fresh_db() as s:
            u = await s.get(User, ids["users"][0])
            u.full_name = "Renamed Since The Backup"
            s.add(u)
            await s.commit()

        await backup_state.restore_state(tmp_path / "bk")

        async with fresh_db() as s:
            assert (await s.get(User, ids["users"][0])).full_name == "Renamed Since The Backup"


class TestSchemaGuard:
    """A restore into a differently-migrated schema is how backups corrupt data."""

    @staticmethod
    async def _set_revision(revision: str) -> None:
        from app.database import engine

        async with engine.begin() as conn:
            await conn.execute(
                sa.text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32))")
            )
            await conn.execute(sa.text("DELETE FROM alembic_version"))
            await conn.execute(
                sa.text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": revision}
            )

    async def test_revision_is_recorded_in_the_manifest(self, fresh_db, tmp_path):
        await self._set_revision("abc123")
        await _seed(fresh_db)
        manifest = await backup_state.export_state(tmp_path / "bk")
        assert manifest["alembic_revision"] == "abc123"

    async def test_mismatched_revision_refuses_to_restore(self, fresh_db, tmp_path):
        await self._set_revision("abc123")
        await _seed(fresh_db)
        await backup_state.export_state(tmp_path / "bk")

        await self._set_revision("def456")
        with pytest.raises(RuntimeError, match="schema mismatch"):
            await backup_state.restore_state(tmp_path / "bk")

    async def test_force_overrides_the_guard(self, fresh_db, tmp_path):
        await self._set_revision("abc123")
        await _seed(fresh_db)
        await backup_state.export_state(tmp_path / "bk")
        await self._set_revision("def456")

        await backup_state.restore_state(tmp_path / "bk", force=True)  # does not raise

    async def test_missing_manifest_is_rejected(self, fresh_db, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="not a backup directory"):
            await backup_state.restore_state(tmp_path / "empty")
