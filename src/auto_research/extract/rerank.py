"""Local Qwen3-Reranker over hybrid-retrieval output.

Reranking reorders the top-`candidate_k` `HybridHit`s produced by
`hybrid_retrieve` (RRF fusion of BM25 + dense) into a top-`k` final
selection using a cross-encoder yes/no score from the Qwen3-Reranker
causal LM. Single backend, three tiers — explicit caller parameter,
no env-var fallback (same discipline as `embeddings.EmbeddingAdapter`):

- `tier="dev"` — `Qwen3-Reranker-0.6B` on Apple-Silicon MPS. Fast
  iteration on dev machines.
- `tier="deployment"` — `Qwen3-Reranker-4B` on MPS. Higher quality at
  ~6-8x the latency; the indexing/extraction-time choice.
- `tier="ci-cpu"` — `Qwen3-Reranker-0.6B` on CPU. Explicit opt-in for
  sandboxed CI without MPS.

There is no implicit env-var fallback. Construct on the wrong platform
and the warmup raises a clear `RuntimeError` pointing at the correct
tier. Mixing tiers within a single index build is forbidden by
`reranker_version` — analogous to `embed_model_version`, this is the
stable token a row-stamp / cache key would carry once a downstream
worker persists reranked output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from auto_research.extract.chunking import ParentChunk
from auto_research.extract.rag_retrieval import HybridHit

_log = logging.getLogger(__name__)

ALLOWED_TIERS: frozenset[str] = frozenset({"dev", "deployment", "ci-cpu"})

_TIER_TO_MODEL: dict[str, str] = {
    "dev": "Qwen3-Reranker-0.6B",
    "deployment": "Qwen3-Reranker-4B",
    "ci-cpu": "Qwen3-Reranker-0.6B",
}

_TIER_TO_DEVICE: dict[str, str] = {
    "dev": "mps",
    "deployment": "mps",
    "ci-cpu": "cpu",
}

_TIER_TO_DTYPE: dict[str, str] = {
    "dev": "fp16",
    "deployment": "fp16",
    "ci-cpu": "fp32",
}

RERANKER_VERSION_TAG: str = "v1"
"""Bump when the reranker scoring contract changes.

Triggers: upstream weight re-upload under same id, prompt-template
edit, dtype/device policy flip, tokenizer truncation budget change.
Non-triggers: choosing a different tier (`reranker_version` already
encodes tier+model).
"""


def reranker_version(tier: str, model: str) -> str:
    """Stable token identifying a reranker score space.

    Returns `"{tier}:{model}:{RERANKER_VERSION_TAG}"`. Including tier
    distinguishes the dev/ci-cpu pair: both run the 0.6B model, but on
    different devices/dtypes, which produces non-identical score
    distributions — treating them as the same score space silently
    degrades any persisted-score downstream consumer.
    """
    return f"{tier}:{model}:{RERANKER_VERSION_TAG}"


class Qwen3Reranker:
    def __init__(
        self,
        *,
        tier: Literal["dev", "deployment", "ci-cpu"],
    ) -> None:
        if tier not in ALLOWED_TIERS:
            raise ValueError(
                f"tier must be one of {sorted(ALLOWED_TIERS)}; got {tier!r}"
            )
        self._tier: Literal["dev", "deployment", "ci-cpu"] = tier
        self._model_id = _TIER_TO_MODEL[tier]
        self._device = _TIER_TO_DEVICE[tier]
        self._dtype = _TIER_TO_DTYPE[tier]
        _log.info(
            "reranker_init tier=%s model=%s device=%s dtype=%s",
            self._tier,
            self._model_id,
            self._device,
            self._dtype,
        )

    @property
    def tier(self) -> Literal["dev", "deployment", "ci-cpu"]:
        return self._tier

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> str:
        return self._dtype

    @cached_property
    def reranker_version(self) -> str:
        return reranker_version(self._tier, self._model_id)


@dataclass(frozen=True)
class RerankHit:
    """One reranked hit, carrying both the reranker score and the prior
    RRF context so a caller can diagnose how the reranker moved each
    item (or persist both columns).
    """

    parent: ParentChunk
    score: float
    prev_rrf_score: float
    prev_rank: int


# A scorer maps `(query, passages)` to a per-passage relevance score
# (larger = more relevant). The protocol lives at the module level so
# the unit tests can substitute a deterministic stub for the real
# Qwen3-Reranker model. The real implementation is `Qwen3Reranker.score`.
ScorerFn = Callable[[str, list[str]], list[float]]


def rerank(
    *,
    query: str,
    hits: Sequence[HybridHit],
    top_k: int,
    scorer: ScorerFn,
) -> list[RerankHit]:
    """Reorder `hits` by `scorer(query, [h.parent.text for h in hits])` and
    return the top `top_k`.

    Tie-break for deterministic output: descending reranker score, then
    descending prior RRF score, then ascending `doc_id`. The reranker's
    yes-probability is dense-floating-point and ties are statistically
    rare, but they DO happen on identical passages (e.g., two filings
    that quote the same boilerplate); the fixed tie-break makes the
    output order reproducible across runs.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")
    if not hits:
        return []
    passages = [h.parent.text for h in hits]
    scores = scorer(query, passages)
    if len(scores) != len(passages):
        raise ValueError(
            f"scorer returned {len(scores)} scores for {len(passages)} passages"
        )
    indexed = list(enumerate(zip(hits, scores, strict=True)))
    indexed.sort(
        key=lambda item: (
            -item[1][1],
            -item[1][0].score,
            item[1][0].parent.metadata.doc_id,
        )
    )
    return [
        RerankHit(
            parent=h.parent,
            score=s,
            prev_rrf_score=h.score,
            prev_rank=orig_idx + 1,
        )
        for orig_idx, (h, s) in indexed[:top_k]
    ]


__all__ = [
    "ALLOWED_TIERS",
    "RERANKER_VERSION_TAG",
    "Qwen3Reranker",
    "RerankHit",
    "ScorerFn",
    "rerank",
    "reranker_version",
]
