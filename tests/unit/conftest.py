"""Unit-test conftest.

Two session-scoped autouse fixtures warm the lazily-loaded NLP models
before any test runs:

- `_warm_chunking_nlp` — `unstructured`'s spaCy `en_core_web_sm`,
  needed by `parse_filing`'s element classification.
- `_warm_bge_embeddings` — BAAI/bge-small-en-v1.5, the in-process
  embedding fallback used by every unit test that exercises
  `EmbeddingAdapter` with `VOYAGE_API_KEY` absent.

Both follow the same pattern: warm the cache at session start (when
sockets are still real) so subsequent hermetic tests can monkey-patch
the network without triggering a lazy model download. If the
underlying model isn't installed/cached the warmup raises
`RuntimeError` with a clear remediation pointing at `make setup-nlp`.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _warm_chunking_nlp() -> None:
    """Warm `auto_research.extract.chunking._ensure_nlp_warmup` once per session.

    Safe no-op if the module isn't imported by any test in the session
    (the import itself is cheap; the spaCy load is deferred to the
    warmup function). If `en_core_web_sm` is missing the warmup raises
    `RuntimeError` with a clear remediation message — better than a
    cryptic socket error from inside `partition_html`.
    """
    from auto_research.extract.chunking import _ensure_nlp_warmup

    _ensure_nlp_warmup()


@pytest.fixture(scope="session", autouse=True)
def _warm_bge_embeddings() -> None:
    """Warm `auto_research.extract.embeddings._ensure_bge_warmup` once per session.

    Without this, the first hermetic unit test exercising
    `EmbeddingAdapter` in BGE mode would trigger a `SentenceTransformer`
    HuggingFace download under whatever socket monkey-patch the test
    installed. Loading at session start (before any socket patching)
    lands the model in HF cache during a real-network window; the
    in-process singleton then serves every adapter for the rest of the
    session at zero further network cost.
    """
    from auto_research.extract.embeddings import _ensure_bge_warmup

    _ensure_bge_warmup()
