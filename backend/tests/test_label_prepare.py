"""
Tests for fine_tuning/01_prepare_labels.py helpers.

The script's filename starts with a digit, so we load it via importlib
rather than `import`. Tests cover the idempotency contract (skip
already-labeled), candidate filtering (short narratives, exclusions),
persistence (DB row + JSONL line), and provider-aware token-usage
parsing across the Groq and Bedrock response shapes.

Live teacher calls (Groq, Bedrock Converse) are not mocked here — those
are integration territory and we verify them with a real `--limit 2 --provider X`
smoke run before any long backfill.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "fine_tuning" / "01_prepare_labels.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_labels", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_labels"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest_asyncio.fixture()
async def fresh_db():
    """Wipe + recreate schema; yield a sessionmaker bound to the engine."""
    import app.models  # noqa: F401 — registers tables
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory


async def _seed_complaint(
    factory,
    *,
    narrative: str = "I am writing to dispute a charge on my credit card.",
    product: str = "Credit card",
    issue: str = "Fees",
    company: str = "Chase",
) -> uuid.UUID:
    from app.models.complaint import Complaint

    async with factory() as s:
        c = Complaint(
            narrative=narrative,
            product=product,
            issue=issue,
            company=company,
        )
        s.add(c)
        await s.commit()
        await s.refresh(c)
        return c.id


async def _seed_label(factory, complaint_id: uuid.UUID, label_source: str) -> None:
    from app.models.complaint_label import ComplaintLabel

    async with factory() as s:
        s.add(
            ComplaintLabel(
                complaint_id=complaint_id,
                label_source=label_source,
                sentiment="neutral",
                intent="information_request",
                urgency=1,
                key_entities=[],
                reasoning="seeded for tests",
            )
        )
        await s.commit()


class TestLoadAlreadyLabeled:
    async def test_returns_only_matching_source(self, fresh_db):
        script = _load_script()
        cid_a = await _seed_complaint(fresh_db)
        cid_b = await _seed_complaint(fresh_db, narrative="X" * 100)
        await _seed_label(fresh_db, cid_a, "groq:llama-3.3-70b-versatile")
        await _seed_label(fresh_db, cid_b, "openai:gpt-4o")

        ids = await script._load_already_labeled("groq:llama-3.3-70b-versatile")
        assert ids == {str(cid_a)}

    async def test_empty_when_no_labels(self, fresh_db):
        script = _load_script()
        await _seed_complaint(fresh_db)
        ids = await script._load_already_labeled("groq:llama-3.3-70b-versatile")
        assert ids == set()


class TestFetchCandidates:
    async def test_excludes_already_labeled(self, fresh_db):
        script = _load_script()
        cid_a = await _seed_complaint(fresh_db)
        cid_b = await _seed_complaint(fresh_db, narrative="A" * 100)

        candidates = await script._fetch_candidates(limit=10, exclude={str(cid_a)})
        ids = {str(c.id) for c in candidates}
        assert cid_b in {c.id for c in candidates}
        assert str(cid_a) not in ids

    async def test_drops_short_narratives(self, fresh_db):
        script = _load_script()
        await _seed_complaint(fresh_db, narrative="too short")  # < 50 chars
        long_id = await _seed_complaint(
            fresh_db, narrative="This is a much longer narrative that easily clears the floor."
        )

        candidates = await script._fetch_candidates(limit=10, exclude=set())
        assert [c.id for c in candidates] == [long_id]

    async def test_respects_limit(self, fresh_db):
        script = _load_script()
        for _ in range(5):
            await _seed_complaint(fresh_db, narrative="X" * 100)

        candidates = await script._fetch_candidates(limit=3, exclude=set())
        assert len(candidates) == 3


class TestPersist:
    async def test_writes_db_and_jsonl(self, fresh_db, tmp_path):
        script = _load_script()
        from sqlmodel import select

        from app.models.complaint_label import ComplaintLabel
        from app.schemas.classification import ComplaintClassification, Entity

        cid = await _seed_complaint(fresh_db)
        async with fresh_db() as s:
            complaint = (
                await s.execute(
                    select(__import__("app.models.complaint", fromlist=["Complaint"]).Complaint)
                )
            ).scalar_one()

        classification = ComplaintClassification(
            sentiment="negative",
            intent="dispute_resolution",
            urgency=3,
            key_entities=[Entity(entity="Chase", type="company")],
            reasoning="Consumer disputing a charge with clear frustration.",
        )
        meta = {"input_tokens": 500, "output_tokens": 100, "latency_ms": 800}
        jsonl_path = tmp_path / "out.jsonl"

        await script._persist(
            complaint,
            "groq:test",
            classification,
            meta,
            jsonl_path,
        )

        async with fresh_db() as s:
            rows = (await s.execute(select(ComplaintLabel))).scalars().all()
        assert len(rows) == 1
        assert rows[0].complaint_id == cid
        assert rows[0].label_source == "groq:test"
        assert rows[0].sentiment == "negative"
        assert rows[0].urgency == 3
        assert rows[0].input_tokens == 500

        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["complaint_id"] == str(cid)
        assert record["label_source"] == "groq:test"
        assert record["classification"]["sentiment"] == "negative"
        assert record["latency_ms"] == 800

    async def test_unique_constraint_blocks_duplicate(self, fresh_db, tmp_path):
        script = _load_script()
        from sqlalchemy.exc import IntegrityError
        from sqlmodel import select

        from app.schemas.classification import ComplaintClassification

        await _seed_complaint(fresh_db)
        async with fresh_db() as s:
            complaint = (
                await s.execute(
                    select(__import__("app.models.complaint", fromlist=["Complaint"]).Complaint)
                )
            ).scalar_one()

        classification = ComplaintClassification(
            sentiment="neutral",
            intent="information_request",
            urgency=1,
            key_entities=[],
            reasoning="First insert should succeed.",
        )
        jsonl_path = tmp_path / "out.jsonl"

        await script._persist(complaint, "groq:test", classification, {}, jsonl_path)
        with pytest.raises(IntegrityError):
            await script._persist(complaint, "groq:test", classification, {}, jsonl_path)


class TestTokenExtraction:
    """Token-usage field names differ between OpenAI-style and Bedrock responses;
    `_extract_input_tokens` / `_extract_output_tokens` must normalize both."""

    def test_groq_shape_object_with_usage_attribute(self):
        script = _load_script()
        raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=512, completion_tokens=48))
        assert script._extract_input_tokens(raw, "groq") == 512
        assert script._extract_output_tokens(raw, "groq") == 48

    def test_bedrock_shape_dict_with_camelcase_usage(self):
        script = _load_script()
        raw = {"usage": {"inputTokens": 3000, "outputTokens": 240, "totalTokens": 3240}}
        assert script._extract_input_tokens(raw, "bedrock") == 3000
        assert script._extract_output_tokens(raw, "bedrock") == 240

    def test_bedrock_shape_object_fallback(self):
        """instructor may return a Pydantic-ish object instead of a raw dict —
        the extractor should still find `inputTokens`/`outputTokens`."""
        script = _load_script()
        raw = SimpleNamespace(usage=SimpleNamespace(inputTokens=2048, outputTokens=128))
        assert script._extract_input_tokens(raw, "bedrock") == 2048
        assert script._extract_output_tokens(raw, "bedrock") == 128

    def test_handles_none_raw(self):
        script = _load_script()
        assert script._extract_input_tokens(None, "groq") is None
        assert script._extract_output_tokens(None, "bedrock") is None


class TestRetryPredicate:
    """`_is_retryable` is the single tenacity predicate covering both providers."""

    def test_openai_rate_limit_error_is_retryable(self):
        script = _load_script()
        from openai import RateLimitError

        # RateLimitError constructor signature varies by openai-python version,
        # so build a minimal stand-in via __new__ to avoid version-coupling.
        err = RateLimitError.__new__(RateLimitError)
        BaseException.__init__(err, "rate limit")
        assert script._is_retryable(err)

    def test_bedrock_throttling_exception_is_retryable(self):
        script = _load_script()
        from botocore.exceptions import ClientError

        err = ClientError(
            error_response={"Error": {"Code": "ThrottlingException", "Message": "Slow down"}},
            operation_name="Converse",
        )
        assert script._is_retryable(err)
        assert script._is_bedrock_throttle(err)

    def test_bedrock_validation_exception_is_not_retryable(self):
        """ValidationException means our request was malformed — retry won't help."""
        script = _load_script()
        from botocore.exceptions import ClientError

        err = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "Bad input"}},
            operation_name="Converse",
        )
        assert not script._is_retryable(err)
        assert not script._is_bedrock_throttle(err)

    def test_plain_value_error_is_not_retryable(self):
        script = _load_script()
        assert not script._is_retryable(ValueError("nope"))


class TestMakeClient:
    """`_make_client` returns (client, is_async). Verify the Bedrock path
    constructs the boto3 client with the configured region and reports sync."""

    def test_bedrock_returns_sync_flag(self, monkeypatch):
        script = _load_script()
        import boto3

        from app.config import settings

        captured: dict[str, object] = {}

        def fake_client(service_name: str, region_name: str | None = None, **_kwargs):
            captured["service_name"] = service_name
            captured["region_name"] = region_name
            return SimpleNamespace(_service_name=service_name)

        # Pin a known region so the assertion isn't fragile to host env.
        monkeypatch.setattr(settings, "aws_region", "us-west-2", raising=False)
        monkeypatch.setattr(boto3, "client", fake_client)

        # instructor.from_bedrock would try to wrap the fake client; intercept it.
        import instructor

        def fake_from_bedrock(client, mode=None):
            return SimpleNamespace(_wrapped=client, _mode=mode)

        monkeypatch.setattr(instructor, "from_bedrock", fake_from_bedrock)

        client, is_async = script._make_client("bedrock")
        assert is_async is False
        assert captured == {"service_name": "bedrock-runtime", "region_name": "us-west-2"}
        # Ensure instructor wrapped our fake client, not a real one.
        assert client._wrapped._service_name == "bedrock-runtime"

    def test_unknown_provider_exits(self):
        script = _load_script()
        with pytest.raises(SystemExit) as exc:
            script._make_client("anthropic-direct")
        assert exc.value.code == 2
