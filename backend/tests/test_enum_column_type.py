"""Round-trip tests for the EnumString column type.

A row loaded fresh from the DB must carry the Enum (with .value), not a bare
str — that's the whole reason EnumString exists over a plain String(50). The
unit tests pin the bind/result contract directly; the integration tests prove
it through a real persist-then-reload in a *separate* session (so the value
comes back through the column type, not the identity map).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint, ComplaintStatus
from app.models.resolution import GuardrailStatus, Resolution
from app.models.types import EnumString
from app.models.user import User, UserRole


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401 — registers tables on SQLModel.metadata
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


class TestEnumStringContract:
    def test_bind_param_enum_to_value(self):
        t = EnumString(ComplaintStatus)
        assert t.process_bind_param(ComplaintStatus.pending, None) == "pending"

    def test_bind_param_raw_string_is_normalized(self):
        t = EnumString(ComplaintStatus)
        assert t.process_bind_param("escalated", None) == "escalated"

    def test_bind_param_unknown_value_rejected(self):
        t = EnumString(ComplaintStatus)
        with pytest.raises(ValueError):
            t.process_bind_param("not_a_real_status", None)

    def test_bind_and_result_handle_none(self):
        t = EnumString(ComplaintStatus)
        assert t.process_bind_param(None, None) is None
        assert t.process_result_value(None, None) is None

    def test_result_value_returns_enum_singleton(self):
        t = EnumString(ComplaintStatus)
        assert t.process_result_value("classified", None) is ComplaintStatus.classified


async def _reload(factory, obj, obj_id, model):
    async with factory() as s:
        s.add(obj)
        await s.commit()
    async with factory() as s:
        return await s.get(model, obj_id)


class TestRoundTrip:
    async def test_complaint_status_is_enum_after_reload(self, factory):
        c = Complaint(narrative="x", status=ComplaintStatus.escalated)
        loaded = await _reload(factory, c, c.id, Complaint)
        assert loaded.status is ComplaintStatus.escalated
        assert loaded.status.value == "escalated"  # .value works, no guard

    async def test_column_default_is_enum_after_reload(self, factory):
        c = Complaint(narrative="x")  # status defaults to pending
        loaded = await _reload(factory, c, c.id, Complaint)
        assert loaded.status is ComplaintStatus.pending

    async def test_guardrail_status_is_enum_after_reload(self, factory):
        c = Complaint(narrative="x", status=ComplaintStatus.escalated)
        async with factory() as s:
            s.add(c)
            await s.commit()
            cid = c.id
        r = Resolution(complaint_id=cid, draft_text="d", guardrail_status=GuardrailStatus.passed)
        loaded = await _reload(factory, r, r.id, Resolution)
        assert loaded.guardrail_status is GuardrailStatus.passed

    async def test_user_role_is_enum_after_reload(self, factory):
        u = User(email="a@b.com", full_name="A", hashed_password="x", role=UserRole.admin)
        loaded = await _reload(factory, u, u.id, User)
        assert loaded.role is UserRole.admin
