"""
Sentence-transformer wrapper for converting complaint narratives into dense
384-dim vectors suitable for similarity search.

Why this shape:
  - Model load is the expensive thing (~80MB download + ~1s warm-up on CPU).
    We hide that behind `@lru_cache(maxsize=1)` so the model is loaded at most
    once per process, on first use. Subsequent calls reuse the in-memory copy.
  - We expose two entry points: `embed_text` for single strings (used by API
    request handlers) and `embed_batch` for the bulk pipeline. Batching matters
    a lot on CPU — sentence-transformers can saturate the BLAS threads on long
    inputs, and per-call Python overhead is non-trivial for 200K rows.
  - The model name lives in a module-level constant. Switching models later
    means changing one line and re-running the backfill; nothing else cares.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model. Cached for process lifetime."""
    from sentence_transformers import SentenceTransformer

    log.info("loading sentence-transformer model %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info(
        "model loaded: max_seq_length=%d embedding_dim=%d",
        model.max_seq_length,
        model.get_sentence_embedding_dimension(),
    )
    return model


def embed_text(text: str) -> list[float]:
    """Embed a single complaint narrative. Returns a 384-dim list of floats."""
    if not text or not text.strip():
        raise ValueError("cannot embed empty text")
    model = _get_model()
    # encode() returns a numpy array; tolist() makes it JSON/Qdrant-friendly
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Embed many texts. Returns a list of 384-dim vectors in the same order.

    Empty / whitespace-only inputs raise — the caller is expected to filter
    those out, since silently dropping rows would misalign vectors with their
    source records.
    """
    if not texts:
        return []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(f"texts[{i}] is empty; filter empty narratives before embedding")

    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()


def reset_cache() -> None:
    """Clear the model cache. Test hook — production code should not call this."""
    _get_model.cache_clear()
