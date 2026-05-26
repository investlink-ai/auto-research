"""Embedding adapter for the RAG retrieval layer.

Backend selection is **explicit** — the caller passes
`backend="voyage"` (production default, `voyage-finance-2` per ADR D1)
or `backend="bge"` (in-process `bge-small-en-v1.5`, used by tests and
airgapped dev). There is no env-var-driven implicit fallback: a
missing `VOYAGE_API_KEY` does NOT silently switch the adapter to BGE
— it raises when the Voyage client is first constructed, surfacing
the misconfiguration loudly. Workers / CLI entry points read
`EMBEDDING_BACKEND` themselves and pass the choice in.

The backend is locked for the adapter's lifetime. There is no
mid-run switch on quota or any other Voyage error — a single corpus
must live in a single vector space, since Voyage's 1024-dim and BGE's
384-dim outputs are not comparable under cosine similarity (dense
retrieval would silently degrade). On `voyageai.error.RateLimitError`
the call propagates; operational handling (retry-with-backoff,
circuit breaking, quota alerting) lives at the worker layer.
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
_BGE_HF_ID = f"BAAI/{_BGE_MODEL_NAME}"
_PER_CORPUS_STORE = "_corpus_narrative"
_WORKER = "embeddings"

# One-shot BGE warmup. Mirrors `extract.chunking._nlp_warmup` for the
# embedding side: `SentenceTransformer` lazy-loads its model from
# HuggingFace on first instantiation, a network call that breaks
# hermetic unit tests (socket-monkey-patched) and silently surprises
# fresh CI runners that have no `~/.cache/huggingface/` populated.
#
# `_ensure_bge_warmup` is idempotent via `_BGE_MODEL` (the module-level
# cache). The conftest autouse fixture calls it once at session start —
# before any socket-blocking test runs — so the model lands in cache
# during a "real" network window. `EmbeddingAdapter._bge` reuses the
# same singleton so multiple adapter instances in a single process
# share one loaded model rather than each paying the ~1-2s reload cost.
_BGE_MODEL: Any = None

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


def _ensure_bge_warmup() -> Any:
    """Load BGE once and return the cached `SentenceTransformer`.

    Idempotent via the module-level `_BGE_MODEL` singleton. First call
    instantiates the model (downloading from HuggingFace if absent from
    the local cache, raising a clear `RuntimeError` if both the cache
    is empty AND the network is unavailable — typically a hermetic-test
    socket monkey-patch or an airgapped CI runner). Subsequent calls
    return the cached instance.

    Production code paths (`EmbeddingAdapter._bge`) call this to share
    the singleton; the unit-conftest autouse fixture calls it at session
    start so hermetic tests can monkey-patch sockets without triggering
    a lazy HF download. Mirror of `extract.chunking._nlp_warmup`.
    """
    global _BGE_MODEL
    if _BGE_MODEL is not None:
        return _BGE_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _BGE_MODEL = SentenceTransformer(_BGE_HF_ID)
    except Exception as exc:
        raise RuntimeError(
            f"BGE model {_BGE_HF_ID!r} could not be loaded — likely a "
            "HuggingFace cache miss with no network reachable. Populate "
            "the cache with:\n"
            "    make setup-nlp\n"
            "(CI runs this in the same step as `uv sync` per "
            ".github/workflows/ci.yml.)"
        ) from exc
    return _BGE_MODEL


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


BGE_MODEL_ID = "bge-small-en-v1.5"


def resolve_backend_from_env() -> Literal["voyage", "bge"]:
    """Read `EMBEDDING_BACKEND` from the environment and validate.

    Workers / CLI entry points call this once at startup and pass the
    explicit choice into `EmbeddingAdapter(backend=...)`. The adapter
    itself never reads this env var — selection is the caller's
    responsibility, missing config is a loud error not a silent
    fallback.
    """
    raw = os.environ.get("EMBEDDING_BACKEND")
    if raw not in {"voyage", "bge"}:
        raise RuntimeError(
            "EMBEDDING_BACKEND env var must be set to 'voyage' or 'bge'; "
            f"got {raw!r}. Set explicitly — there is no default fallback."
        )
    return raw  # type: ignore[return-value]


class EmbeddingAdapter:
    def __init__(
        self,
        *,
        backend: Literal["voyage", "bge"],
        rag_root: Path = Path("data/rag"),
        voyage_model: str | None = None,
    ) -> None:
        """Construct an adapter bound to an explicitly-chosen backend.

        `backend` is required — no env-var inference, no default. Pass
        `"voyage"` for production (`voyage-finance-2` by default, or
        the model named in `voyage_model` / `$VOYAGE_MODEL`) or
        `"bge"` for the in-process `bge-small-en-v1.5` fallback.

        `voyage_model` is only honored when `backend="voyage"`; passing
        it alongside `backend="bge"` is rejected so the caller's intent
        stays unambiguous.
        """
        if backend == "voyage":
            resolved = (
                voyage_model
                or os.environ.get("VOYAGE_MODEL")
                or DEFAULT_VOYAGE_MODEL
            )
            if resolved not in ALLOWED_VOYAGE_MODELS:
                raise ValueError(
                    f"VOYAGE_MODEL={resolved!r} not in "
                    f"{sorted(ALLOWED_VOYAGE_MODELS)}"
                )
            self._backend: Literal["voyage", "bge"] = "voyage"
            self._model = resolved
        elif backend == "bge":
            if voyage_model is not None:
                raise ValueError(
                    "voyage_model is only valid when backend='voyage'; "
                    f"got backend='bge' with voyage_model={voyage_model!r}"
                )
            self._backend = "bge"
            self._model = BGE_MODEL_ID
        else:
            raise ValueError(
                f"backend must be 'voyage' or 'bge'; got {backend!r}"
            )
        self._rag_root = rag_root
        _log.info(
            "embedding_adapter_init backend=%s model=%s",
            self._backend,
            self._model,
        )

    @property
    def backend(self) -> Literal["voyage", "bge"]:
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    @cached_property
    def _vector_dim(self) -> int:
        return _MODEL_DIM[self._model]

    @cached_property
    def _bge(self) -> Any:
        # Routes through the module-level singleton so every adapter in
        # the process shares one warm model; surfaces a clear remediation
        # error when the HF cache is empty and the network is blocked.
        return _ensure_bge_warmup()

    def _encode(
        self, texts: list[str], *, input_type: str = "document"
    ) -> NDArray[np.float32]:
        if self._backend == "bge":
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
            model=self._model,
            input_type=input_type,
        )
        return np.asarray(resp.embeddings, dtype=np.float32)

    @cached_property
    def _voyage_client(self) -> Any:
        import voyageai

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            # Loud failure rather than silent fallback to BGE: the
            # caller explicitly chose backend='voyage', so a missing
            # key is a misconfiguration to surface, not a signal to
            # quietly degrade.
            raise RuntimeError(
                "EmbeddingAdapter was constructed with backend='voyage' "
                "but VOYAGE_API_KEY is not set. Provide the key, or "
                "construct the adapter with backend='bge' for the "
                "in-process fallback."
            )
        return voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]

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
            span.set_attribute("embedding.backend", self._backend)
            span.set_attribute("embedding.model", self._model)
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
            # `embedding.backend`/`model` deliberately omitted: BM25 is
            # purely lexical (Lance FTS over the `text` column) and is
            # independent of which embedding backend produced the
            # vector index alongside it.
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
            span.set_attribute("embedding.backend", self._backend)
            span.set_attribute("embedding.model", self._model)
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
