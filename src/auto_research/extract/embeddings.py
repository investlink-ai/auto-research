"""Embedding adapter for the RAG retrieval layer.

Default model is Voyage `voyage-finance-2` per ADR D1
(`docs/decisions/2026-05-24-rag-enhancements.md`); falls back to local
`bge-small-en-v1.5` when `VOYAGE_API_KEY` is absent or `force_local=True`.
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

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


def _corpus_vector_dim(tbl: Any) -> int:
    """Read the existing LanceDB table's vector-column fixed-list size.

    Used to detect a backend switch between embed() calls so the corpus
    narrative index is dropped+recreated (rather than .add()ing rows of a
    different dim, which raises pyarrow ArrowInvalid).
    """
    vector_field = tbl.schema.field("vector")
    return int(vector_field.type.list_size)


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
    reason: str  # voyage_used | no_key | quota | explicit_override


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

    @property
    def _vector_dim(self) -> int:
        # Read from current decision every call: a quota-driven switch from
        # voyage(1024) to bge(384) must not leave the schema at the
        # pre-switch dimension on subsequent embeds. Plain dict lookup, no
        # cost to recomputing.
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

        try:
            resp = self._voyage_client.embed(
                texts, model=self._decision.model, input_type=input_type
            )
        except RateLimitError:
            self._decision = FallbackDecision("bge", "bge-small-en-v1.5", "quota")
            _log.warning(
                "embedding_quota_switch backend=bge model=%s reason=quota",
                _BGE_MODEL_NAME,
            )
            return self._encode(texts, input_type=input_type)
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
            pre_reason = self._decision.reason
            try:
                texts = [c.embedding_text for c in chunks]
                vectors = self._encode(texts)
                rows = self._rows(chunks, vectors)

                self._rag_root.mkdir(parents=True, exist_ok=True)
                db = lancedb.connect(self._rag_root)
                schema = _schema(self._vector_dim)

                db.create_table(doc_id, data=rows, schema=schema, mode="overwrite")

                narrative_rows = [r for r in rows if r["doc_type"] in NARRATIVE_DOC_TYPES]
                if narrative_rows:
                    if _PER_CORPUS_STORE in db.table_names():
                        existing = db.open_table(_PER_CORPUS_STORE)
                        if _corpus_vector_dim(existing) == self._vector_dim:
                            existing.add(narrative_rows)
                        else:
                            # Backend switch (e.g., voyage 1024 → bge 384 after
                            # quota) means the existing corpus narrative index
                            # is dim-incompatible with the new rows. Cross-
                            # backend dense retrieval is incoherent, so we drop
                            # the prior table and start fresh on the new dim.
                            # WARN, not raise — backfill should keep moving.
                            _log.warning(
                                "embedding_corpus_dim_mismatch dropping=%s "
                                "old_dim=%d new_dim=%d new_backend=%s",
                                _PER_CORPUS_STORE,
                                _corpus_vector_dim(existing),
                                self._vector_dim,
                                self._decision.backend,
                            )
                            db.drop_table(_PER_CORPUS_STORE)
                            db.create_table(
                                _PER_CORPUS_STORE, data=narrative_rows, schema=schema
                            )
                    else:
                        db.create_table(
                            _PER_CORPUS_STORE, data=narrative_rows, schema=schema
                        )
                span.set_attribute("embedding.backend", self._decision.backend)
                span.set_attribute("embedding.model", self._decision.model)
                span.set_attribute("embedding.fallback_reason", self._decision.reason)
                span.set_attribute("embedding.narrative_count", len(narrative_rows))
                # quota_fallback = the embed completed, but switched mid-call.
                # success = ran end-to-end on the original backend.
                if pre_reason == "voyage_used" and self._decision.reason == "quota":
                    span.set_attribute("extract.outcome", "quota_fallback")
                else:
                    span.set_attribute("extract.outcome", "success")
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
