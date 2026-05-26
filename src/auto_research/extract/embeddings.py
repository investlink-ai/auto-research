"""Embedding adapter for the RAG retrieval layer.

Default model is Voyage `voyage-finance-2` per ADR D1
(`docs/decisions/2026-05-24-rag-enhancements.md`); BGE
`bge-small-en-v1.5` is selectable as a whole-run alternative when
`VOYAGE_API_KEY` is absent or `force_local=True`.

The backend is chosen once at adapter init and locked for its lifetime.
There is no mid-run switch on quota or any other Voyage error — a
single corpus must live in a single vector space, since Voyage's
1024-dim and BGE's 384-dim outputs are not comparable under cosine
similarity (dense retrieval would silently degrade). On
`voyageai.error.RateLimitError` the call propagates; operational
handling (retry-with-backoff, circuit breaking, quota alerting) lives
at the worker layer.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import lancedb
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from auto_research.extract.chunking_contextual import ContextualChildChunk
from auto_research.telemetry import truncate_status_description as _truncate

ALLOWED_VOYAGE_MODELS: frozenset[str] = frozenset({
    "voyage-finance-2",
    "voyage-3",
    "voyage-3-large",
    "voyage-3.5",
    "voyage-4",
    "voyage-4-large",
})
DEFAULT_VOYAGE_MODEL = "voyage-finance-2"

_MODEL_DIM: dict[str, int] = {
    "voyage-finance-2": 1024,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "bge-small-en-v1.5": 384,
}

NARRATIVE_DOC_TYPES: frozenset[str] = frozenset({"10-K", "10-Q", "transcript"})
_BGE_MODEL_NAME = "bge-small-en-v1.5"
_PER_CORPUS_STORE = "_corpus_narrative"
_WORKER = "embeddings"

# Voyage rate-limit posture. This project's Voyage account is on the
# constrained tier — **3 RPM / 10,000 TPM** — not the doc-page Tier 1
# (2000 RPM / millions TPM). 429 is the documented signal across tiers
# and Voyage recommends exponential-with-jitter backoff. The 20s initial
# wait matches the 3-RPM window so the first retry lands at the start of
# the next quota slot; the 120s cap allows a TPM-bound burst to settle.
# Six attempts span ~3-4 minutes worst-case before the RateLimitError
# propagates — at which point the orchestrator (not the adapter) decides
# whether to wait longer, alert, or stop the backfill.
#
# These retries are REACTIVE only. At 3 RPM, sustained throughput needs
# proactive pacing (one call every ~20s) at the orchestrator / backfill
# layer; the adapter is too low-level to coordinate that across workers.
_VOYAGE_RETRY_WAIT_INITIAL = 20.0
_VOYAGE_RETRY_WAIT_MAX = 120.0
_VOYAGE_RETRY_ATTEMPTS = 6

# LanceDB FTS kwargs applied identically to the per-doc and per-corpus
# narrative tables. Pulled to a module constant so the two `create_fts_
# index` callsites stay in sync — diverging tokenizer / stopword / stem
# settings across the two stores would make the same BM25 query produce
# different rankings depending on which surface it hit.
#
# - `use_tantivy=False`: Lance native FTS, not the Tantivy backend.
#   Native supports incremental updates via `table.add()` (needed for
#   the corpus store, which appends per filing), avoids Tantivy's
#   1 GB writer-heap allocation per index, and matches the BM25
#   semantics we need (phrase / fuzzy / regex queries are out of scope).
# - `replace=True`: idempotent against re-embeds — the per-doc table is
#   recreated on every `embed()` (`mode="overwrite"`); defensive on the
#   per-corpus path against future code paths that might re-invoke this.
# - `remove_stop_words=True` and `stem=True`: empirically required for
#   SEC English. Without stopword removal, chunks containing only
#   common-word query overlap rank above lexically-disjoint relevant
#   chunks. Without stemming, BM25 misses morphological variants
#   ("change"/"changed"/"changes" don't collapse). Both calibrated
#   during Issue #16's hybrid-retrieval micro-corpus tuning.
_FTS_INDEX_KWARGS: dict[str, Any] = {
    "use_tantivy": False,
    "replace": True,
    "remove_stop_words": True,
    "stem": True,
}

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema([
        ("text", pa.string()),
        ("vector", pa.list_(pa.float32(), vector_dim)),
        ("ticker", pa.string()),
        ("filing_date", pa.string()),
        ("fiscal_period", pa.string()),
        ("doc_type", pa.string()),
        ("doc_id", pa.string()),
        ("parent_id", pa.string()),
        ("section_name", pa.string()),
    ])


@dataclass(frozen=True)
class QueryHit:
    text: str
    score: float
    parent_id: str
    section_name: str
    ticker: str
    filing_date: date
    doc_type: str
    doc_id: str


@dataclass(frozen=True)
class FallbackDecision:
    backend: str
    model: str
    reason: str  # voyage_used | no_key | explicit_override (all init-time)


class EmbeddingAdapter:
    def __init__(
        self,
        *,
        rag_root: Path = Path("data/rag"),
        voyage_model: str | None = None,
        force_local: bool = False,
    ) -> None:
        resolved = voyage_model or os.environ.get("VOYAGE_MODEL") or DEFAULT_VOYAGE_MODEL
        if resolved not in ALLOWED_VOYAGE_MODELS:
            raise ValueError(
                f"VOYAGE_MODEL={resolved!r} not in {sorted(ALLOWED_VOYAGE_MODELS)}"
            )
        self._rag_root = rag_root
        if force_local:
            self._decision = FallbackDecision("bge", "bge-small-en-v1.5", "explicit_override")
        elif not os.environ.get("VOYAGE_API_KEY"):
            self._decision = FallbackDecision("bge", "bge-small-en-v1.5", "no_key")
        else:
            self._decision = FallbackDecision("voyage", resolved, "voyage_used")
        _log.info(
            "embedding_adapter_init backend=%s model=%s reason=%s",
            self._decision.backend,
            self._decision.model,
            self._decision.reason,
        )

    @property
    def decision(self) -> FallbackDecision:
        return self._decision

    @cached_property
    def _vector_dim(self) -> int:
        return _MODEL_DIM[self._decision.model]

    @cached_property
    def _bge(self) -> Any:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(f"BAAI/{_BGE_MODEL_NAME}")

    def _encode(
        self, texts: list[str], *, input_type: str = "document"
    ) -> NDArray[np.float32]:
        if self._decision.backend == "bge":
            # BGE-small doesn't take an input_type prompt; same encoder for
            # both corpus and query sides.
            arr: NDArray[np.float32] = self._bge.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
            return arr.astype(np.float32)

        from voyageai.error import RateLimitError

        # Retry the Voyage call on 429s with exponential backoff + jitter
        # (matches Voyage's documented recommendation). After the budget is
        # exhausted, RateLimitError propagates — the adapter never switches
        # to BGE mid-run, since a mixed-vector-space corpus is incoherent.
        retrying = Retrying(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential_jitter(
                initial=_VOYAGE_RETRY_WAIT_INITIAL,
                max=_VOYAGE_RETRY_WAIT_MAX,
            ),
            stop=stop_after_attempt(_VOYAGE_RETRY_ATTEMPTS),
            before_sleep=before_sleep_log(_log, logging.WARNING),
            reraise=True,
        )
        resp = retrying(
            self._voyage_client.embed,
            texts,
            model=self._decision.model,
            input_type=input_type,
        )
        return np.asarray(resp.embeddings, dtype=np.float32)

    @cached_property
    def _voyage_client(self) -> Any:
        import voyageai

        return voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])  # type: ignore[attr-defined]

    def _rows(
        self, chunks: Sequence[ContextualChildChunk], vectors: NDArray[np.float32]
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            md = chunk.child.metadata
            rows.append({
                "text": chunk.embedding_text,
                "vector": vec.tolist(),
                "ticker": md.ticker,
                "filing_date": md.filing_date.isoformat(),
                "fiscal_period": md.fiscal_period,
                "doc_type": md.doc_type,
                "doc_id": md.doc_id,
                "parent_id": chunk.child.parent_id,
                "section_name": chunk.child.section_name,
            })
        return rows

    def embed(self, chunks: Sequence[ContextualChildChunk]) -> None:
        if not chunks:
            return
        doc_ids = {c.child.metadata.doc_id for c in chunks}
        if len(doc_ids) != 1:
            raise ValueError(
                f"embed() requires single doc_id per call; got {sorted(doc_ids)}"
            )
        doc_id = next(iter(doc_ids))

        with _tracer.start_as_current_span("extract.embed") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("extract.doc_id", doc_id)
            span.set_attribute("embedding.chunks_count", len(chunks))
            span.set_attribute("embedding.backend", self._decision.backend)
            span.set_attribute("embedding.model", self._decision.model)
            span.set_attribute("embedding.fallback_reason", self._decision.reason)
            try:
                texts = [c.embedding_text for c in chunks]
                vectors = self._encode(texts)
                rows = self._rows(chunks, vectors)

                self._rag_root.mkdir(parents=True, exist_ok=True)
                db = lancedb.connect(self._rag_root)
                schema = _schema(self._vector_dim)

                per_doc_tbl = db.create_table(
                    doc_id, data=rows, schema=schema, mode="overwrite"
                )
                per_doc_tbl.create_fts_index("text", **_FTS_INDEX_KWARGS)

                narrative_rows = [r for r in rows if r["doc_type"] in NARRATIVE_DOC_TYPES]
                if narrative_rows:
                    if _PER_CORPUS_STORE in db.table_names():
                        # LanceDB updates the FTS index incrementally on `.add()`.
                        db.open_table(_PER_CORPUS_STORE).add(narrative_rows)
                    else:
                        corpus_tbl = db.create_table(
                            _PER_CORPUS_STORE, data=narrative_rows, schema=schema
                        )
                        corpus_tbl.create_fts_index("text", **_FTS_INDEX_KWARGS)
                span.set_attribute("embedding.narrative_count", len(narrative_rows))
                span.set_attribute("extract.outcome", "success")
            except Exception as exc:
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

    def bm25_query(
        self,
        text: str,
        *,
        k: int,
        store: Literal["per_doc", "corpus_narrative"] = "per_doc",
        doc_id: str | None = None,
        where: str | None = None,
    ) -> list[QueryHit]:
        """Lexical BM25 search over the LanceDB FTS index on the `text` column.

        Returned hits are ordered descending by `_score` (higher = more
        relevant), packed into the same `QueryHit` shape `query` returns —
        callers treat the list order as the canonical ranking and use
        `score` only for surfacing the raw retriever number.

        Sharing the adapter (and its tables) with `query` keeps BM25 and
        dense over exactly the same corpus, the same metadata columns, and
        the same `where` filter (ADR D7); the hybrid retriever in
        `rag_retrieval.py` composes the two via RRF without orchestrating
        any second source of truth.
        """
        if store == "per_doc" and doc_id is None:
            raise ValueError("doc_id required when store='per_doc'")
        with _tracer.start_as_current_span("extract.bm25_query") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.store", store)
            span.set_attribute("embedding.k", k)
            span.set_attribute("embedding.has_filter", where is not None)
            if doc_id is not None:
                span.set_attribute("extract.doc_id", doc_id)
            try:
                db = lancedb.connect(self._rag_root)
                table_name = doc_id if store == "per_doc" else _PER_CORPUS_STORE
                tbl = db.open_table(table_name)
                builder = tbl.search(text, query_type="fts").limit(k)
                if where:
                    builder = builder.where(where, prefilter=True)
                df = builder.to_pandas()
                hits = [
                    QueryHit(
                        text=row["text"],
                        score=float(row["_score"]),
                        parent_id=row["parent_id"],
                        section_name=row["section_name"],
                        ticker=row["ticker"],
                        filing_date=date.fromisoformat(row["filing_date"]),
                        doc_type=row["doc_type"],
                        doc_id=row["doc_id"],
                    )
                    for _, row in df.iterrows()
                ]
                span.set_attribute("embedding.hits_count", len(hits))
                span.set_attribute("extract.outcome", "success")
                return hits
            except Exception as exc:
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

    def query(
        self,
        text: str,
        *,
        k: int,
        store: Literal["per_doc", "corpus_narrative"] = "per_doc",
        doc_id: str | None = None,
        where: str | None = None,
    ) -> list[QueryHit]:
        if store == "per_doc" and doc_id is None:
            raise ValueError("doc_id required when store='per_doc'")
        with _tracer.start_as_current_span("extract.embed_query") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.backend", self._decision.backend)
            span.set_attribute("embedding.model", self._decision.model)
            span.set_attribute("embedding.store", store)
            span.set_attribute("embedding.k", k)
            span.set_attribute("embedding.has_filter", where is not None)
            if doc_id is not None:
                span.set_attribute("extract.doc_id", doc_id)
            try:
                db = lancedb.connect(self._rag_root)
                table_name = doc_id if store == "per_doc" else _PER_CORPUS_STORE
                tbl = db.open_table(table_name)
                # Voyage benefits from input_type="query" on the query side
                # of the asymmetric encoder; BGE ignores input_type.
                qvec = self._encode([text], input_type="query")[0].tolist()
                builder = tbl.search(qvec).limit(k)
                if where:
                    builder = builder.where(where, prefilter=True)
                df = builder.to_pandas()
                hits = [
                    QueryHit(
                        text=row["text"],
                        score=float(row["_distance"]),
                        parent_id=row["parent_id"],
                        section_name=row["section_name"],
                        ticker=row["ticker"],
                        filing_date=date.fromisoformat(row["filing_date"]),
                        doc_type=row["doc_type"],
                        doc_id=row["doc_id"],
                    )
                    for _, row in df.iterrows()
                ]
                span.set_attribute("embedding.hits_count", len(hits))
                span.set_attribute("extract.outcome", "success")
                return hits
            except Exception as exc:
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise
