"""Live smoke for Qwen3-Reranker on a hand-built micro-corpus.

Asserts the AC requirement that rerank improves precision@5 over RRF
alone, measured against both the 0.6B (dev/ci-cpu) and 4B (deployment)
backends. The 0.6B path runs on Apple-Silicon MPS (matching the dev
tier); the 4B path is gated by QWEN3_FULL=1 because the weights are
~8 GB.

What this catches that unit tests can't:

- Upstream Qwen3-Reranker repo drift (Alibaba rename, weight re-upload,
  tokenizer change).
- Score distribution shift between 0.6B and 4B — both must still rank
  the known-relevant passages above the distractors.
- Apple-Silicon MPS / fp16 numerical issues that don't surface on the
  fp32-CPU unit test path.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from datetime import date

import pytest

from auto_research.extract.chunking import ChunkMetadata, ParentChunk
from auto_research.extract.rag_retrieval import HybridHit
from auto_research.extract.rerank import Qwen3Reranker, rerank


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# The micro-corpus: 12 passages drawn from the kinds of text the
# extraction layer will see. Index 0-3 are the known-relevant set
# (data-center / Hopper / Blackwell AI-infrastructure capacity); the
# rest are distractors of varying topic distance. Index 11 is mildly
# relevant — included to test the reranker's ordering nuance.
_QUERY = "NVIDIA data-center revenue drivers from Hopper and Blackwell GPU shipments"

_PASSAGES_AND_GOLD: list[tuple[str, bool]] = [
    (
        "Data Center revenue grew 154% year-over-year driven by strong "
        "demand for our Hopper architecture and the ramp of Blackwell.",
        True,
    ),
    (
        "Cloud service providers expanded their Hopper-based GPU "
        "capacity through Q4, with multiple hyperscalers placing orders "
        "for Blackwell training systems.",
        True,
    ),
    (
        "Our Blackwell platform began shipping to leading cloud "
        "customers in the fourth quarter; demand exceeds supply.",
        True,
    ),
    (
        "Hopper architecture (H100, H200) shipments remained a primary "
        "driver of Data Center segment revenue in the period.",
        True,
    ),
    (
        "Gaming revenue increased sequentially as channel inventory "
        "normalized and GeForce RTX 40-series demand stayed solid.",
        False,
    ),
    (
        "Automotive segment revenue grew modestly on continued "
        "DRIVE Orin design wins across passenger vehicles.",
        False,
    ),
    (
        "Cash from operations more than doubled year-over-year on "
        "higher net income and favorable working-capital movements.",
        False,
    ),
    (
        "Stock-based compensation expense was largely consistent with "
        "the prior quarter; no acceleration events occurred.",
        False,
    ),
    (
        "Our China revenue continued to be impacted by U.S. export "
        "controls limiting the products we can ship into that market.",
        False,
    ),
    (
        "Operating expenses grew modestly driven by higher engineering "
        "headcount and continued investment in software platforms.",
        False,
    ),
    (
        "The board authorized an additional $25 billion of share "
        "repurchases with no expiration date.",
        False,
    ),
    (
        "Inventory increased sequentially in preparation for the "
        "Blackwell platform ramp and continued Hopper customer demand.",
        True,
    ),
]


def _make_hit(idx: int, text: str, rrf_score: float) -> HybridHit:
    parent = ParentChunk(
        text=text,
        section_name="MD&A",
        char_span=(0, len(text)),
        token_count=max(1, len(text) // 4),
        table_html=None,
        metadata=ChunkMetadata(
            ticker="NVDA",
            filing_date=date(2025, 2, 26),
            fiscal_period="FY2025-Q4",
            doc_type="10-K",
            doc_id=f"doc-NVDA-{idx:02d}",
        ),
    )
    return HybridHit(
        parent=parent,
        score=rrf_score,
        bm25_rank=idx + 1,
        dense_rank=idx + 1,
        bm25_score=1.0 / (idx + 1),
        dense_score=1.0 / (idx + 1),
    )


@dataclass(frozen=True)
class _Corpus:
    hits: list[HybridHit]
    gold_doc_ids: set[str]


def _build_corpus() -> _Corpus:
    """The RRF ordering deliberately mis-orders the corpus: gold items
    are interleaved with distractors so a top-5 cut from RRF alone
    catches only some of them, and the reranker has room to improve."""
    rrf_order = [4, 0, 5, 1, 6, 11, 7, 2, 8, 3, 9, 10]
    hits: list[HybridHit] = []
    for rank_idx, corpus_idx in enumerate(rrf_order):
        text, _ = _PASSAGES_AND_GOLD[corpus_idx]
        hits.append(_make_hit(corpus_idx, text, rrf_score=1.0 - rank_idx * 0.05))
    gold_doc_ids = {
        f"doc-NVDA-{i:02d}"
        for i, (_, is_gold) in enumerate(_PASSAGES_AND_GOLD)
        if is_gold
    }
    return _Corpus(hits=hits, gold_doc_ids=gold_doc_ids)


def _precision_at_5(doc_ids: list[str], gold: set[str]) -> float:
    return sum(1 for d in doc_ids[:5] if d in gold) / 5.0


@pytest.mark.skipif(
    not _is_apple_silicon(), reason="dev-tier MPS requires Apple Silicon"
)
def test_rerank_06b_improves_precision_at_5_over_rrf() -> None:
    corpus = _build_corpus()
    rrf_top5 = [h.parent.metadata.doc_id for h in corpus.hits[:5]]
    rrf_p5 = _precision_at_5(rrf_top5, corpus.gold_doc_ids)

    reranker = Qwen3Reranker(tier="dev")
    reranked = rerank(
        query=_QUERY,
        hits=corpus.hits,
        top_k=5,
        scorer=lambda q, ps: reranker.score(query=q, passages=ps),
    )
    rerank_top5 = [h.parent.metadata.doc_id for h in reranked]
    rerank_p5 = _precision_at_5(rerank_top5, corpus.gold_doc_ids)

    assert rerank_p5 > rrf_p5, (
        f"rerank precision@5={rerank_p5:.2f} did not exceed RRF "
        f"baseline={rrf_p5:.2f}. RRF top-5={rrf_top5}; "
        f"rerank top-5={rerank_top5}; gold={sorted(corpus.gold_doc_ids)}"
    )
    # Reranker must produce a stable order on the same input.
    reranked2 = rerank(
        query=_QUERY,
        hits=corpus.hits,
        top_k=5,
        scorer=lambda q, ps: reranker.score(query=q, passages=ps),
    )
    assert [h.parent.metadata.doc_id for h in reranked2] == rerank_top5


@pytest.mark.skipif(
    os.environ.get("QWEN3_FULL") != "1",
    reason="QWEN3_FULL=1 required to opt into the 8 GB 4B download",
)
@pytest.mark.skipif(
    not _is_apple_silicon(), reason="deployment-tier MPS requires Apple Silicon"
)
def test_rerank_4b_improves_precision_at_5_over_rrf() -> None:
    corpus = _build_corpus()
    rrf_top5 = [h.parent.metadata.doc_id for h in corpus.hits[:5]]
    rrf_p5 = _precision_at_5(rrf_top5, corpus.gold_doc_ids)

    reranker = Qwen3Reranker(tier="deployment")
    reranked = rerank(
        query=_QUERY,
        hits=corpus.hits,
        top_k=5,
        scorer=lambda q, ps: reranker.score(query=q, passages=ps),
    )
    rerank_top5 = [h.parent.metadata.doc_id for h in reranked]
    rerank_p5 = _precision_at_5(rerank_top5, corpus.gold_doc_ids)

    assert rerank_p5 > rrf_p5, (
        f"rerank-4B precision@5={rerank_p5:.2f} did not exceed RRF "
        f"baseline={rrf_p5:.2f}. RRF top-5={rrf_top5}; "
        f"rerank top-5={rerank_top5}; gold={sorted(corpus.gold_doc_ids)}"
    )
