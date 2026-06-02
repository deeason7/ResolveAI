"""Tests for the knowledge-graph API routes.

The Neo4j-backed GraphStore is replaced with an AsyncMock via dependency
override, and auth is bypassed by overriding get_current_user — so these
exercise routing, status codes, query-param forwarding, and response shaping
in isolation, with no live Neo4j and no token plumbing. (Auth enforcement
itself lives in test_api_auth.py; one test here just asserts the gate is wired.)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.schemas.graph import CompanyProfile, ProductBreakdown, Regulation
from app.services.graph_store import GraphStore

COMPANY_URL = "/api/v1/graph/company/{name}"
PRODUCT_REGS_URL = "/api/v1/graph/product/{name}/regulations"
EXPLORE_URL = "/api/v1/graph/explore"

# The routes bind the user to `_` and never read it, so any object works.
_DUMMY_USER = SimpleNamespace(id="test", email="analyst@test.com")


async def _dummy_session():
    # Graph routes never touch Postgres; this keeps the test app off the engine.
    yield None


def _build_app(store, *, bypass_auth=True):
    from app.core.deps import get_current_user, get_graph_store
    from app.database import get_session
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_graph_store] = lambda: store
    app.dependency_overrides[get_session] = _dummy_session
    if bypass_auth:
        app.dependency_overrides[get_current_user] = lambda: _DUMMY_USER
    return app


@pytest.fixture()
def store():
    # spec=GraphStore => only real methods are mockable, and async ones come back
    # as AsyncMock automatically, so `await store.method()` works out of the box.
    return AsyncMock(spec=GraphStore)


@pytest_asyncio.fixture()
async def client(store):
    app = _build_app(store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestCompanyProfile:
    async def test_ok(self, client, store):
        store.get_company_profile.return_value = CompanyProfile(
            name="Equifax",
            total_complaints=120,
            risk_score=0.5,
            violations=["Fair Credit Reporting Act (FCRA)"],
            product_breakdown=[ProductBreakdown(product="Credit reporting", count=80)],
        )
        r = await client.get(COMPANY_URL.format(name="Equifax"))

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "Equifax"
        assert body["risk_score"] == 0.5
        assert body["product_breakdown"][0]["product"] == "Credit reporting"
        store.get_company_profile.assert_awaited_once_with("Equifax")

    async def test_unknown_company_is_404(self, client, store):
        store.get_company_profile.return_value = None
        r = await client.get(COMPANY_URL.format(name="No Such Co"))
        assert r.status_code == 404


class TestProductRegulations:
    async def test_ok(self, client, store):
        store.get_regulations.return_value = [
            Regulation(
                id="FCRA",
                title="Fair Credit Reporting Act (FCRA)",
                cfr_reference="15 U.S.C. § 1681",
                summary="Accuracy and privacy of consumer credit information.",
                key_provisions=["30-day dispute resolution"],
            )
        ]
        r = await client.get(PRODUCT_REGS_URL.format(name="Credit reporting"))

        assert r.status_code == 200, r.text
        body = r.json()
        assert body[0]["id"] == "FCRA"
        # No issue query param => issue forwarded as None.
        store.get_regulations.assert_awaited_once_with("Credit reporting", None)

    async def test_empty_is_200_not_404(self, client, store):
        # A product with no mapped regulations is a valid empty result, not 404.
        store.get_regulations.return_value = []
        r = await client.get(PRODUCT_REGS_URL.format(name="Credit reporting"))
        assert r.status_code == 200
        assert r.json() == []

    async def test_issue_filter_forwarded(self, client, store):
        store.get_regulations.return_value = []
        r = await client.get(
            PRODUCT_REGS_URL.format(name="Credit reporting"),
            params={"issue": "Incorrect information on your report"},
        )
        assert r.status_code == 200
        store.get_regulations.assert_awaited_once_with(
            "Credit reporting", "Incorrect information on your report"
        )


class TestExplore:
    async def test_ok(self, client, store):
        store.get_graph_neighborhood.return_value = {
            "nodes": [{"id": "Equifax", "labels": ["Company"], "props": {}}],
            "edges": [
                {
                    "source": "Equifax",
                    "target": "Credit reporting",
                    "type": "HAS_COMPLAINTS_ABOUT",
                    "props": {"count": 80},
                }
            ],
        }
        r = await client.get(EXPLORE_URL, params={"node_id": "Equifax"})

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["nodes"][0]["id"] == "Equifax"
        assert body["edges"][0]["type"] == "HAS_COMPLAINTS_ABOUT"
        # Default depth is 2.
        store.get_graph_neighborhood.assert_awaited_once_with("Equifax", 2)

    async def test_depth_forwarded(self, client, store):
        store.get_graph_neighborhood.return_value = {"nodes": [{"id": "x"}], "edges": []}
        await client.get(EXPLORE_URL, params={"node_id": "x", "depth": 3})
        store.get_graph_neighborhood.assert_awaited_once_with("x", 3)

    async def test_unknown_node_is_404(self, client, store):
        store.get_graph_neighborhood.return_value = {"nodes": [], "edges": []}
        r = await client.get(EXPLORE_URL, params={"node_id": "ghost"})
        assert r.status_code == 404

    async def test_depth_over_cap_is_422(self, client, store):
        # depth is bounded to <= 3 to stop a traversal from pulling the whole graph.
        r = await client.get(EXPLORE_URL, params={"node_id": "x", "depth": 9})
        assert r.status_code == 422
        store.get_graph_neighborhood.assert_not_awaited()


class TestAuthGate:
    async def test_requires_auth(self, store):
        # Build the app WITHOUT bypassing auth; no token => 401 from get_current_user.
        app = _build_app(store, bypass_auth=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get(COMPANY_URL.format(name="Equifax"))
        assert r.status_code == 401
        store.get_company_profile.assert_not_awaited()
