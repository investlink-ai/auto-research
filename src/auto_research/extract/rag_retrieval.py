"""Hybrid retrieval — BM25 + dense, fused via Reciprocal Rank Fusion.

Both halves live on `EmbeddingAdapter`: dense via `query` (vector index),
BM25 via `bm25_query` (LanceDB native FTS index built at `embed`-time).
This module orchestrates the two and fuses the rankings at PARENT
granularity (ADR D4) — a parent's per-source rank is the best (smallest)
rank across its children.

ADR D8 reserves `bm25_weight` / `dense_weight` hooks for future tuning;
they default to symmetric (standard RRF). Tuning is deferred until
#20/#21 produce eval data, but the surface is stable now so a later
weight change does not require a callsite refactor.

`rrf_fuse` is a pure function and unit-testable in isolation;
`hybrid_retrieve` is the thin orchestration on top.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from auto_research.extract.chunking import ParentChunk
from auto_research.extract.embeddings import EmbeddingAdapter

DEFAULT_RRF_K: int = 60
DEFAULT_CANDIDATE_K: int = 20


@dataclass(frozen=True)
class HybridHit:
    """One fused retrieval hit, resolved to its `ParentChunk`.

    `bm25_rank` / `dense_rank` are 1-based parent-level ranks (None if
    the parent was not returned by that retriever). `bm25_score` /
    `dense_score` are the raw retriever scores for the best child of
    each parent — BM25's positive relevance score and the dense
    retriever's distance respectively, surfaced for downstream tuning.
    """

    parent: ParentChunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None


def rrf_fuse(
    *,
    bm25_ranking: Sequence[str],
    dense_ranking: Sequence[str],
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion over two ranked id lists.

    For every id present in either ranking, the score is
    `bm25_weight / (rrf_k + r_bm25) + dense_weight / (rrf_k + r_dense)`
    with the missing term set to 0. Returned list is sorted score
    descending; ties broken by id (lexicographic) for determinism.
    """
    bm25_ranks = {doc: i + 1 for i, doc in enumerate(bm25_ranking)}
    dense_ranks = {doc: i + 1 for i, doc in enumerate(dense_ranking)}
    keys = set(bm25_ranks) | set(dense_ranks)
    scored: list[tuple[str, float]] = []
    for k in keys:
        s = 0.0
        if k in bm25_ranks:
            s += bm25_weight / (rrf_k + bm25_ranks[k])
        if k in dense_ranks:
            s += dense_weight / (rrf_k + dense_ranks[k])
        scored.append((k, s))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


def _collapse_to_parent_ranking(
    pairs: Iterable[tuple[str, float]],
) -> tuple[list[str], dict[str, float]]:
    """Reduce ordered (parent_id, score) pairs to a parent-level ranking.

    `pairs` is assumed to be in descending relevance order — the first
    occurrence of each parent_id wins (it's the best child) and later
    duplicates are ignored. Returns the deduplicated parent order plus
    a map from parent_id to its best raw score.
    """
    best: dict[str, float] = {}
    order: list[str] = []
    for pid, score in pairs:
        if pid in best:
            continue
        best[pid] = score
        order.append(pid)
    return order, best


def hybrid_retrieve(
    *,
    query: str,
    adapter: EmbeddingAdapter,
    parents: Sequence[ParentChunk],
    doc_id: str,
    k: int = 5,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    rrf_k: int = DEFAULT_RRF_K,
    where: str | None = None,
) -> list[HybridHit]:
    """Run BM25 and dense in parallel over the adapter's per-doc store
    and fuse via RRF.

    `candidate_k` controls how many children each retriever pulls before
    fusion; the final result is the top-`k` parents after RRF. Setting
    `dense_weight=0` (or `bm25_weight=0`) skips the corresponding side —
    useful for ablation tests, not a production hook.

    `where` applies the same metadata filter to BOTH retrievers (ADR
    D7 — ticker/filing_date scoping is a corpus property, not a
    retriever-specific concern).
    """
    parent_by_id = {
        f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}": p
        for p in parents
    }

    bm25_hits = (
        adapter.bm25_query(
            query, k=candidate_k, store="per_doc", doc_id=doc_id, where=where
        )
        if bm25_weight > 0
        else []
    )
    dense_hits = (
        adapter.query(
            query, k=candidate_k, store="per_doc", doc_id=doc_id, where=where
        )
        if dense_weight > 0
        else []
    )

    bm25_ranking, bm25_best = _collapse_to_parent_ranking(
        (h.parent_id, h.score) for h in bm25_hits if h.parent_id in parent_by_id
    )
    # `adapter.query` packs LanceDB `_distance` into `score` (smaller =
    # closer); flip the sign so the surfaced `dense_score` is monotone
    # with relevance — same direction as BM25's positive score.
    dense_ranking, dense_best = _collapse_to_parent_ranking(
        (h.parent_id, -h.score) for h in dense_hits if h.parent_id in parent_by_id
    )

    fused = rrf_fuse(
        bm25_ranking=bm25_ranking,
        dense_ranking=dense_ranking,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
    )

    bm25_rank_of = {pid: i + 1 for i, pid in enumerate(bm25_ranking)}
    dense_rank_of = {pid: i + 1 for i, pid in enumerate(dense_ranking)}

    return [
        HybridHit(
            parent=parent_by_id[pid],
            score=rrf_score,
            bm25_rank=bm25_rank_of.get(pid),
            dense_rank=dense_rank_of.get(pid),
            bm25_score=bm25_best.get(pid),
            dense_score=dense_best.get(pid),
        )
        for pid, rrf_score in fused[:k]
    ]


__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_RRF_K",
    "HybridHit",
    "hybrid_retrieve",
    "rrf_fuse",
]
