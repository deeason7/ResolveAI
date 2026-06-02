"""
Tests for the Neo4j graph store + the seed script's pure helpers.

Neo4j has no in-memory mode (unlike Qdrant's :memory:), so we mock at the
driver's execute_query seam: a fake driver returns canned records, and we
assert both the shape we build AND the Cypher/params we send. The seed
helpers are pure functions, tested directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.schemas.graph import CompanyProfile, Regulation, ResolutionPattern
from app.services.graph_store import GraphStore

# ---------------- Fakes ----------------


class _FakeRecord:
    """Stand-in for neo4j.Record — only .data() is used by GraphStore._run."""

    def __init__(self, data: dict):
        self._data = data

    def data(self) -> dict:
        return self._data


def _eager(records: list[dict]):
    """Build a fake EagerResult tuple: (records, summary, keys)."""
    return ([_FakeRecord(r) for r in records], None, None)


@pytest.fixture()
def driver():
    d = AsyncMock()
    d.execute_query = AsyncMock(return_value=_eager([]))
    return d


@pytest.fixture()
def store(driver) -> GraphStore:
    return GraphStore(driver)


# ---------------- get_company_profile ----------------


class TestCompanyProfile:
    async def test_maps_record_to_profile(self, store, driver):
        driver.execute_query.return_value = _eager(
            [
                {
                    "name": "Wells Fargo",
                    "total_complaints": 1200,
                    "risk_score": 0.82,
                    "violations": ["Fair Credit Reporting Act"],
                    "product_breakdown": [{"product": "Mortgage", "count": 800}],
                }
            ]
        )
        profile = await store.get_company_profile("Wells Fargo")
        assert isinstance(profile, CompanyProfile)
        assert profile.total_complaints == 1200
        assert profile.violations == ["Fair Credit Reporting Act"]
        assert profile.product_breakdown[0].product == "Mortgage"
        assert profile.product_breakdown[0].count == 800

    async def test_missing_company_returns_none(self, store):
        # default fixture returns no records
        assert await store.get_company_profile("Nope Inc") is None

    async def test_company_with_no_edges(self, store, driver):
        driver.execute_query.return_value = _eager(
            [
                {
                    "name": "Quiet Co",
                    "total_complaints": None,
                    "risk_score": None,
                    "violations": [],
                    "product_breakdown": [],
                }
            ]
        )
        profile = await store.get_company_profile("Quiet Co")
        assert profile.total_complaints == 0  # None coalesced to 0
        assert profile.product_breakdown == []

    async def test_passes_company_name_param(self, store, driver):
        await store.get_company_profile("ACME")
        _, kwargs = driver.execute_query.call_args
        assert kwargs["company_name"] == "ACME"

    async def test_product_breakdown_query_is_ordered(self, store, driver):
        # collect() is unordered; the query must sort by edge count so the
        # caller's product_breakdown leads with the dominant product.
        await store.get_company_profile("ACME")
        query = driver.execute_query.call_args[0][0]
        assert "ORDER BY h.count" in query


# ---------------- get_regulations ----------------


class TestRegulations:
    async def test_with_issue_returns_regulations(self, store, driver):
        driver.execute_query.return_value = _eager(
            [
                {
                    "regulation": {
                        "id": "FCRA",
                        "title": "Fair Credit Reporting Act",
                        "cfr_reference": "15 U.S.C. § 1681",
                        "summary": "x",
                        "key_provisions": ["a", "b"],
                    }
                }
            ]
        )
        regs = await store.get_regulations("Credit reporting", "Incorrect information")
        assert len(regs) == 1
        assert isinstance(regs[0], Regulation)
        assert regs[0].id == "FCRA"

    async def test_issue_branch_sends_both_params(self, store, driver):
        await store.get_regulations("Mortgage", "Loan servicing")
        query, kwargs = driver.execute_query.call_args[0][0], driver.execute_query.call_args[1]
        assert "HAS_ISSUE" in query and "GOVERNED_BY" in query
        # product-context guard: shared Issue nodes would otherwise leak in regs
        # that only apply to a different product.
        assert "APPLIES_TO" in query
        assert kwargs["product"] == "Mortgage"
        assert kwargs["issue"] == "Loan servicing"

    async def test_no_issue_uses_applies_to_branch(self, store, driver):
        await store.get_regulations("Mortgage")
        query = driver.execute_query.call_args[0][0]
        assert "APPLIES_TO" in query
        assert "issue" not in driver.execute_query.call_args[1]


# ---------------- get_resolution_patterns ----------------


class TestResolutionPatterns:
    async def test_maps_and_includes_company_usage(self, store, driver):
        driver.execute_query.return_value = _eager(
            [
                {
                    "pattern_type": "Closed with monetary relief",
                    "description": "d",
                    "success_rate": 0.95,
                    "company_usage_count": 12,
                }
            ]
        )
        patterns = await store.get_resolution_patterns("Fees", company="ACME")
        assert isinstance(patterns[0], ResolutionPattern)
        assert patterns[0].company_usage_count == 12

    async def test_null_company_usage(self, store, driver):
        driver.execute_query.return_value = _eager(
            [
                {
                    "pattern_type": "Closed with explanation",
                    "description": "d",
                    "success_rate": 0.55,
                    "company_usage_count": None,
                }
            ]
        )
        patterns = await store.get_resolution_patterns("Fees")
        assert patterns[0].company_usage_count is None


# ---------------- get_graph_neighborhood ----------------


class TestNeighborhood:
    async def test_returns_nodes_and_edges(self, store, driver):
        driver.execute_query.return_value = _eager(
            [
                {
                    "nodes": [{"id": "FCRA", "labels": ["Regulation"], "props": {}}],
                    "edges": [
                        {"source": "Mortgage", "target": "FCRA", "type": "GOVERNED_BY", "props": {}}
                    ],
                }
            ]
        )
        result = await store.get_graph_neighborhood("FCRA", depth=2)
        assert result["nodes"][0]["id"] == "FCRA"
        assert result["edges"][0]["type"] == "GOVERNED_BY"

    async def test_empty_neighborhood(self, store):
        assert await store.get_graph_neighborhood("ghost") == {"nodes": [], "edges": []}

    async def test_depth_param_forwarded(self, store, driver):
        await store.get_graph_neighborhood("FCRA", depth=3)
        assert driver.execute_query.call_args[1]["depth"] == 3


# ---------------- upsert_complaint_entities ----------------


class TestUpsert:
    async def test_company_and_product_increment(self, store, driver):
        await store.upsert_complaint_entities("cid", "ACME", "Mortgage", None)
        # one write for the company/product edge
        query = driver.execute_query.call_args_list[0][0][0]
        assert "HAS_COMPLAINTS_ABOUT" in query
        assert driver.execute_query.call_args_list[0][1]["company"] == "ACME"

    async def test_product_and_issue_increment(self, store, driver):
        await store.upsert_complaint_entities("cid", None, "Mortgage", "Servicing")
        query = driver.execute_query.call_args_list[0][0][0]
        assert "HAS_ISSUE" in query

    async def test_regulations_create_violated_edges(self, store, driver):
        await store.upsert_complaint_entities(
            "cid", "ACME", None, None, regulations=["FCRA", "TILA"]
        )
        queries = [c[0][0] for c in driver.execute_query.call_args_list]
        assert sum("VIOLATED" in q for q in queries) == 2

    async def test_all_none_is_noop(self, store, driver):
        await store.upsert_complaint_entities("cid", None, None, None)
        driver.execute_query.assert_not_called()

    async def test_writes_use_write_routing(self, store, driver):
        from neo4j import RoutingControl

        await store.upsert_complaint_entities("cid", "ACME", "Mortgage", None)
        assert driver.execute_query.call_args[1]["routing_"] == RoutingControl.WRITE


# ---------------- seed_graph pure helpers ----------------


def _load_seed_module():
    path = Path(__file__).resolve().parents[2] / "graph_seed" / "seed_graph.py"
    spec = importlib.util.spec_from_file_location("seed_graph", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seed_graph = _load_seed_module()


class TestRiskScore:
    def test_zero_complaints_is_zero(self):
        assert seed_graph.compute_risk_score(0, 0, 100) == 0.0

    def test_bounded_zero_to_one(self):
        s = seed_graph.compute_risk_score(500, 500, 1000)  # 100% adverse
        assert 0.0 <= s <= 1.0

    def test_volume_is_log_scaled(self):
        # 10x the volume should not 10x the score (log compression)
        small = seed_graph.compute_risk_score(100, 0, 100_000)
        big = seed_graph.compute_risk_score(1000, 0, 100_000)
        assert big > small
        assert big < small * 10

    def test_adverse_rate_raises_score(self):
        clean = seed_graph.compute_risk_score(1000, 0, 10_000)
        dirty = seed_graph.compute_risk_score(1000, 1000, 10_000)
        assert dirty > clean


class TestFuzzyMatch:
    def test_substring_both_directions(self):
        prods = ["Credit reporting, credit repair services, or other personal consumer reports"]
        assert seed_graph.match_products_to_regulation(["Credit reporting"], prods) == prods

    def test_no_match_returns_empty(self):
        assert seed_graph.match_products_to_regulation(["Mortgage"], ["Student loan"]) == []

    def test_issue_keyword_match(self):
        issues = ["Incorrect information on your report", "Loan servicing"]
        out = seed_graph.match_issues_to_pattern(["report"], issues)
        assert out == ["Incorrect information on your report"]

    def test_empty_addresses_matches_nothing(self):
        assert seed_graph.match_issues_to_pattern([], ["anything"]) == []

    def test_response_to_pattern_mapping(self):
        patterns = [
            {"id": "monetary_relief", "response_values": ["Closed with monetary relief"]},
            {"id": "untimely", "response_values": ["Untimely response"]},
        ]
        m = seed_graph.build_response_to_pattern(patterns)
        assert m["Untimely response"] == "untimely"
        assert m["Closed with monetary relief"] == "monetary_relief"
