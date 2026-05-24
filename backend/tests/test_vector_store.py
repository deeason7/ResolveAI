"""
Tests for the embedder + vector store services.

We use Qdrant's :memory: mode (an in-process implementation that mimics the
server) so tests are self-contained, fast, and don't need docker. The embedder
is stubbed out where the test only cares about the wrapper logic — actually
downloading the 80MB MiniLM model would make CI miserable.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.services import embedder, vector_store
from app.services.vector_store import (
    COLLECTION_NAME,
    ComplaintPoint,
    VectorStore,
)

# ---------------- Fixtures ----------------


@pytest.fixture()
def store() -> VectorStore:
    """A fresh in-memory Qdrant collection per test."""
    client = QdrantClient(":memory:")
    s = VectorStore(client)
    s.ensure_collection()
    return s


@pytest.fixture()
def stub_embedder(monkeypatch):
    """
    Replace the real model with a deterministic stub.

    Hash the text → seed a numpy RNG → emit a normalized 384-vector. Same text
    always produces the same vector, different text produces different vectors,
    and we never touch the real model.
    """

    class _StubModel:
        max_seq_length = 256

        def get_sentence_embedding_dimension(self):
            return embedder.EMBEDDING_DIM

        def encode(self, texts, **kwargs):
            single = isinstance(texts, str)
            items = [texts] if single else list(texts)
            vectors = []
            for t in items:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(embedder.EMBEDDING_DIM).astype(np.float32)
                # normalize so cosine == dot, matching real model's behavior
                v /= np.linalg.norm(v) or 1.0
                vectors.append(v)
            arr = np.stack(vectors)
            return arr[0] if single else arr

    # Clear any cached real model from a prior test, then swap in the stub.
    # monkeypatch reverts the swap automatically when the test ends.
    embedder._get_model.cache_clear()
    monkeypatch.setattr(embedder, "_get_model", lambda: _StubModel())
    yield


# ---------------- Embedder ----------------


class TestEmbedder:
    def test_embed_text_returns_right_shape(self, stub_embedder):
        v = embedder.embed_text("a customer complaint about overdraft fees")
        assert isinstance(v, list)
        assert len(v) == embedder.EMBEDDING_DIM
        assert all(isinstance(x, float) for x in v)

    def test_embed_text_is_deterministic(self, stub_embedder):
        text = "wrongly reported as delinquent"
        assert embedder.embed_text(text) == embedder.embed_text(text)

    def test_embed_text_rejects_empty(self, stub_embedder):
        with pytest.raises(ValueError):
            embedder.embed_text("   ")

    def test_embed_batch_preserves_order(self, stub_embedder):
        texts = ["alpha", "beta", "gamma"]
        batch_vecs = embedder.embed_batch(texts)
        assert len(batch_vecs) == 3
        for text, vec in zip(texts, batch_vecs, strict=True):
            assert vec == embedder.embed_text(text)

    def test_embed_batch_rejects_empty_member(self, stub_embedder):
        with pytest.raises(ValueError, match=r"texts\[1\]"):
            embedder.embed_batch(["ok", "  ", "also ok"])

    def test_embed_batch_empty_list_returns_empty(self, stub_embedder):
        assert embedder.embed_batch([]) == []


# ---------------- Vector store: collection lifecycle ----------------


class TestCollectionLifecycle:
    def test_ensure_collection_creates_when_missing(self):
        client = QdrantClient(":memory:")
        s = VectorStore(client)
        assert s.ensure_collection() is True
        assert client.collection_exists(COLLECTION_NAME)

    def test_ensure_collection_is_idempotent(self, store):
        # store fixture already called ensure_collection once
        assert store.ensure_collection() is False

    def test_empty_collection_count_is_zero(self, store):
        assert store.collection_count() == 0


# ---------------- Vector store: upsert ----------------


class TestUpsert:
    def test_upsert_single_increments_count(self, store):
        cid = uuid.uuid4()
        store.upsert_complaint(cid, [0.1] * 384, {"product": "Mortgage"})
        assert store.collection_count() == 1

    def test_upsert_batch_inserts_all(self, store):
        points = [
            ComplaintPoint(
                complaint_id=uuid.uuid4(),
                embedding=[float(i % 5) / 5.0] * 384,
                payload={"product": "Credit card"},
            )
            for i in range(50)
        ]
        n = store.upsert_batch(points)
        assert n == 50
        assert store.collection_count() == 50

    def test_upsert_empty_batch_is_noop(self, store):
        assert store.upsert_batch([]) == 0
        assert store.collection_count() == 0

    def test_upsert_is_idempotent_on_same_id(self, store):
        cid = uuid.uuid4()
        store.upsert_complaint(cid, [0.5] * 384, {"v": 1})
        store.upsert_complaint(cid, [0.5] * 384, {"v": 2})
        assert store.collection_count() == 1

    def test_dimension_mismatch_raises(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.upsert_complaint(uuid.uuid4(), [0.1] * 100, {})


# ---------------- Vector store: search ----------------


def _seeded_vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v) or 1.0
    return v.tolist()


class TestSearch:
    def test_search_returns_nearest_first(self, store):
        query = _seeded_vector(42)
        # Same-as-query → score should be highest (cosine ~ 1.0)
        same_id = uuid.uuid4()
        store.upsert_complaint(same_id, query, {"label": "same"})
        # Different seeds → unrelated vectors
        for i in range(5):
            store.upsert_complaint(uuid.uuid4(), _seeded_vector(i), {"label": "other"})

        hits = store.search_similar(query, limit=3)
        assert hits, "expected at least one hit"
        assert hits[0].complaint_id == str(same_id)
        assert hits[0].score > 0.99
        # Results are sorted by score descending
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_payload(self, store):
        cid = uuid.uuid4()
        store.upsert_complaint(
            cid,
            _seeded_vector(7),
            {"product": "Mortgage", "company": "Acme Bank"},
        )
        hits = store.search_similar(_seeded_vector(7), limit=1)
        assert hits[0].payload["product"] == "Mortgage"
        assert hits[0].payload["company"] == "Acme Bank"

    def test_filter_narrows_results(self, store):
        # Two clusters: same vector seed, different products
        for _ in range(3):
            store.upsert_complaint(uuid.uuid4(), _seeded_vector(11), {"product": "Mortgage"})
        for _ in range(3):
            store.upsert_complaint(uuid.uuid4(), _seeded_vector(11), {"product": "Credit card"})

        hits = store.search_similar(
            _seeded_vector(11),
            filters={"product": "Mortgage"},
            limit=10,
        )
        assert len(hits) == 3
        assert all(h.payload["product"] == "Mortgage" for h in hits)

    def test_filter_with_none_values_is_ignored(self, store):
        store.upsert_complaint(uuid.uuid4(), _seeded_vector(1), {"product": "X"})
        # All filters are None → treated as unfiltered
        hits = store.search_similar(
            _seeded_vector(1),
            filters={"product": None, "company": None},
            limit=5,
        )
        assert len(hits) == 1

    def test_empty_filter_is_equivalent_to_no_filter(self, store):
        store.upsert_complaint(uuid.uuid4(), _seeded_vector(2), {"product": "X"})
        a = store.search_similar(_seeded_vector(2), filters={}, limit=5)
        b = store.search_similar(_seeded_vector(2), filters=None, limit=5)
        assert len(a) == len(b) == 1


# ---------------- Default-store singleton ----------------


class TestDefaultStore:
    def test_reset_clears_singleton(self, monkeypatch):
        # Avoid hitting a real Qdrant — patch the constructor
        captured = []

        class _FakeClient:
            def __init__(self, **kwargs):
                captured.append(kwargs)

            def collection_exists(self, name):
                return True  # short-circuit ensure_collection

        monkeypatch.setattr(vector_store, "QdrantClient", _FakeClient)
        vector_store.reset_default_store()
        vector_store.get_default_store()
        vector_store.reset_default_store()
        vector_store.get_default_store()
        # Two reset+get cycles → two client instances
        assert len(captured) == 2
