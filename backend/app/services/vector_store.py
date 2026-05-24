"""
Qdrant wrapper for the `complaint_embeddings` collection.

Why this shape:
  - We package the (client, collection_name, dim) triple as a small class
    because they always travel together — passing them around as separate
    function args would force every caller to remember "oh yeah, dim is 384
    and the collection is called complaint_embeddings". The class is the
    natural seam for tests (inject a :memory: client) and for swapping
    collections later (e.g., a separate `regulation_embeddings`).
  - Collection creation is idempotent. `ensure_collection()` checks existence
    first so the populate script and the API process can race-call it without
    fear. Qdrant does have a "create if not exists" pattern via try/except,
    but the explicit `collection_exists` check is clearer and avoids relying
    on exception types as control flow.
  - Filters are exposed as a plain dict (`{"product": "Mortgage"}`) instead
    of Qdrant's FieldCondition DSL. That keeps callers decoupled from the
    Qdrant SDK — if we ever migrate to pgvector or Weaviate, only this file
    changes. The translation happens in `_build_filter`.
  - Point IDs are the Postgres complaint UUID, stringified. Qdrant accepts
    str-UUID natively and stores it efficiently. This gives us a direct,
    stable lookup back to the source row without an auxiliary mapping table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.services.embedder import EMBEDDING_DIM

log = logging.getLogger(__name__)

COLLECTION_NAME = "complaint_embeddings"


@dataclass
class SimilarComplaint:
    """A single search hit. Score is cosine similarity in [-1, 1]; higher is closer."""

    complaint_id: str
    score: float
    payload: dict[str, Any]


@dataclass
class ComplaintPoint:
    """One row to upsert: stable id + the embedding + searchable payload."""

    complaint_id: str | UUID
    embedding: list[float]
    payload: dict[str, Any]


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
        dim: int = EMBEDDING_DIM,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dim = dim

    # --- Collection lifecycle ---

    def ensure_collection(self) -> bool:
        """Create the collection if missing. Returns True if newly created."""
        if self.client.collection_exists(self.collection_name):
            return False
        log.info(
            "creating Qdrant collection %s (dim=%d, distance=COSINE)",
            self.collection_name,
            self.dim,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        return True

    def collection_count(self) -> int:
        """Total points currently in the collection. Useful for verification."""
        info = self.client.count(collection_name=self.collection_name, exact=True)
        return info.count

    # --- Writes ---

    def upsert_complaint(
        self,
        complaint_id: str | UUID,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Upsert a single point. Use upsert_batch for any non-trivial volume."""
        self._validate_vector(embedding)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qm.PointStruct(
                    id=str(complaint_id),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def upsert_batch(self, points: list[ComplaintPoint]) -> int:
        """Upsert many points in one round-trip. Returns the count upserted."""
        if not points:
            return 0
        for p in points:
            self._validate_vector(p.embedding)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qm.PointStruct(
                    id=str(p.complaint_id),
                    vector=p.embedding,
                    payload=p.payload,
                )
                for p in points
            ],
        )
        return len(points)

    # --- Reads ---

    def search_similar(
        self,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[SimilarComplaint]:
        """
        Find the most similar complaints, optionally narrowed by metadata.

        `filters` is a flat dict, e.g. {"product": "Mortgage", "state": "CA"}.
        All conditions are AND-ed. None values are dropped.
        """
        self._validate_vector(embedding)
        query_filter = self._build_filter(filters)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            SimilarComplaint(
                complaint_id=str(p.id),
                score=p.score,
                payload=p.payload or {},
            )
            for p in result.points
        ]

    # --- Internals ---

    def _validate_vector(self, vec: list[float]) -> None:
        if len(vec) != self.dim:
            raise ValueError(f"embedding dimension mismatch: got {len(vec)}, expected {self.dim}")

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> qm.Filter | None:
        if not filters:
            return None
        clean = {k: v for k, v in filters.items() if v is not None}
        if not clean:
            return None
        return qm.Filter(
            must=[qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in clean.items()]
        )


@lru_cache(maxsize=1)
def get_default_store() -> VectorStore:
    """Process-wide singleton wired to the configured Qdrant instance."""
    log.info("connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = VectorStore(client)
    store.ensure_collection()
    return store


def reset_default_store() -> None:
    """Drop the singleton. Test hook only."""
    get_default_store.cache_clear()
