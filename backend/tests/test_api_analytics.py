"""
Analytics endpoint tests — aggregates over directly-seeded complaint rows.

Rows are inserted straight through the engine (not the submit API) so each
test controls date_received / sentiment / urgency exactly — the aggregate
math is the thing under test, not the intake path.
"""

from datetime import datetime, timedelta

import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint, ComplaintStatus

REGISTER_URL = "/api/v1/auth/register"
TRENDS_URL = "/api/v1/analytics/sentiment/trends"
BREAKDOWN_URL = "/api/v1/analytics/products/breakdown"
RISK_URL = "/api/v1/analytics/companies/risk"

VALID_USER = {
    "email": "analyst@test.com",
    "full_name": "Test Analyst",
    "password": "securepassword123",
}


async def _register_and_token(client: AsyncClient, user: dict) -> str:
    r = await client.post(REGISTER_URL, json=user)
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


@pytest_asyncio.fixture()
async def analyst_token(client: AsyncClient) -> str:
    return await _register_and_token(client, VALID_USER)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _complaint(**overrides) -> Complaint:
    fields = {
        "narrative": "Test narrative long enough to be a valid complaint body.",
        "product": "Checking account",
        "company": "BigBank",
        "status": ComplaintStatus.classified,
    }
    fields.update(overrides)
    return Complaint(**fields)


async def _seed(*complaints: Complaint) -> None:
    from app.database import engine

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        for c in complaints:
            s.add(c)
        await s.commit()


class TestSentimentTrends:
    async def test_counts_grouped_by_day_and_sentiment(
        self, client: AsyncClient, analyst_token: str
    ):
        day = datetime.utcnow() - timedelta(days=2)
        await _seed(
            _complaint(sentiment="negative", date_received=day),
            _complaint(sentiment="negative", date_received=day),
            _complaint(sentiment="neutral", date_received=day),
        )

        r = await client.get(TRENDS_URL, headers=_auth(analyst_token))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 3
        counts = {(p["day"], p["sentiment"]): p["count"] for p in body["points"]}
        key_day = day.strftime("%Y-%m-%d")
        assert counts[(key_day, "negative")] == 2
        assert counts[(key_day, "neutral")] == 1

    async def test_unlabeled_rows_reported_as_unclassified(
        self, client: AsyncClient, analyst_token: str
    ):
        await _seed(_complaint(sentiment=None, status=ComplaintStatus.pending))

        r = await client.get(TRENDS_URL, headers=_auth(analyst_token))
        assert r.status_code == 200
        sentiments = {p["sentiment"] for p in r.json()["points"]}
        assert sentiments == {"unclassified"}

    async def test_window_excludes_rows_older_than_days(
        self, client: AsyncClient, analyst_token: str
    ):
        await _seed(
            _complaint(sentiment="negative", date_received=datetime.utcnow() - timedelta(days=90)),
            _complaint(sentiment="neutral", date_received=datetime.utcnow() - timedelta(days=1)),
        )

        r = await client.get(TRENDS_URL, params={"days": 30}, headers=_auth(analyst_token))
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["points"][0]["sentiment"] == "neutral"

    async def test_date_received_wins_over_created_at(
        self, client: AsyncClient, analyst_token: str
    ):
        # created_at is "now" (ingest time) but the complaint event happened 90
        # days ago — the window must judge it by date_received and exclude it.
        await _seed(
            _complaint(sentiment="negative", date_received=datetime.utcnow() - timedelta(days=90))
        )

        r = await client.get(TRENDS_URL, params={"days": 30}, headers=_auth(analyst_token))
        assert r.status_code == 200
        assert r.json()["total"] == 0

    async def test_requires_auth(self, client: AsyncClient):
        r = await client.get(TRENDS_URL)
        assert r.status_code == 401


class TestProductsBreakdown:
    async def test_pivots_urgency_counts_per_product(self, client: AsyncClient, analyst_token: str):
        await _seed(
            _complaint(product="Mortgage", urgency=5),
            _complaint(product="Mortgage", urgency=5),
            _complaint(product="Mortgage", urgency=3),
            _complaint(product="Mortgage", urgency=None, status=ComplaintStatus.pending),
        )

        r = await client.get(BREAKDOWN_URL, headers=_auth(analyst_token))
        assert r.status_code == 200, r.text
        row = next(i for i in r.json()["items"] if i["product"] == "Mortgage")
        assert row["total"] == 4
        assert row["urgency_counts"] == {"5": 2, "3": 1}
        assert row["unclassified"] == 1

    async def test_null_product_bucketed_as_unspecified(
        self, client: AsyncClient, analyst_token: str
    ):
        await _seed(_complaint(product=None))

        r = await client.get(BREAKDOWN_URL, headers=_auth(analyst_token))
        assert r.status_code == 200
        assert [i["product"] for i in r.json()["items"]] == ["Unspecified"]

    async def test_sorted_by_total_desc(self, client: AsyncClient, analyst_token: str):
        await _seed(
            _complaint(product="Mortgage"),
            _complaint(product="Credit card"),
            _complaint(product="Credit card"),
        )

        r = await client.get(BREAKDOWN_URL, headers=_auth(analyst_token))
        assert r.status_code == 200
        assert [i["product"] for i in r.json()["items"]] == ["Credit card", "Mortgage"]

    async def test_requires_auth(self, client: AsyncClient):
        r = await client.get(BREAKDOWN_URL)
        assert r.status_code == 401


class TestCompaniesRisk:
    async def test_ordered_by_volume_and_limited(self, client: AsyncClient, analyst_token: str):
        await _seed(
            *[_complaint(company="MegaBank") for _ in range(3)],
            *[_complaint(company="MidBank") for _ in range(2)],
            _complaint(company="TinyBank"),
        )

        r = await client.get(RISK_URL, params={"limit": 2}, headers=_auth(analyst_token))
        assert r.status_code == 200, r.text
        body = r.json()
        assert [i["company"] for i in body["items"]] == ["MegaBank", "MidBank"]
        assert body["items"][0]["total_complaints"] == 3
        assert body["limit"] == 2

    async def test_severity_columns(self, client: AsyncClient, analyst_token: str):
        await _seed(
            _complaint(company="MegaBank", urgency=5, sentiment="extreme_negative"),
            _complaint(company="MegaBank", urgency=4, sentiment="negative"),
            _complaint(company="MegaBank", urgency=1, sentiment="neutral"),
        )

        r = await client.get(RISK_URL, headers=_auth(analyst_token))
        assert r.status_code == 200
        row = r.json()["items"][0]
        assert row["avg_urgency"] == 3.33
        assert row["urgent_count"] == 2
        assert row["extreme_negative_count"] == 1

    async def test_null_company_rows_excluded(self, client: AsyncClient, analyst_token: str):
        await _seed(_complaint(company=None), _complaint(company="BigBank"))

        r = await client.get(RISK_URL, headers=_auth(analyst_token))
        assert r.status_code == 200
        assert [i["company"] for i in r.json()["items"]] == ["BigBank"]

    async def test_requires_auth(self, client: AsyncClient):
        r = await client.get(RISK_URL)
        assert r.status_code == 401
