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
import math
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

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

# Tokenizer truncation budget. Qwen3-Reranker is a causal LM with
# 32K-token context; 2048 covers the contextual-chunking pattern
# (LLM-generated context prefix + child text + query + instruction)
# without pinning the device to full 32K on every batch.
_RERANKER_MAX_LENGTH = 2048

# Prompt template per the Qwen3-Reranker model card. The "Given X,
# retrieve Y" framing matches the model's training distribution
# (the model card's default instruction is "Given a web search query,
# retrieve relevant passages that answer the query"), and the corpus
# tail mirrors `embeddings._QWEN3_QUERY_INSTRUCTION` so dense + rerank
# describe the same task. Output direction ("yes"/"no") is already
# pinned by `_RERANKER_PROMPT_PREFIX`; not repeated here. Revisit when
# a Ragas / DeepEval baseline gives a tuning handle.
_RERANKER_INSTRUCTION = (
    "Given a financial research query, retrieve relevant passages "
    "from SEC filings, earnings transcripts, and analyst materials."
)
_RERANKER_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only '
    'be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_RERANKER_PROMPT_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# HF repo ids. Use the upstream Qwen org weights directly (the reranker
# has not been re-quantized by mlx-community at issue pickup time).
_RERANKER_HF_REPOS: dict[str, str] = {
    "Qwen3-Reranker-0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "Qwen3-Reranker-4B": "Qwen/Qwen3-Reranker-4B",
}

# Module-level singleton cache keyed by `(model_id, device, dtype)`.
# Including dtype keeps two tiers that happen to share `(model, device)`
# but use different precision from silently sharing weights — score
# distributions diverge by dtype even on the same hardware.
_RERANKER_MODELS: dict[tuple[str, str, str], tuple[Any, Any]] = {}

_ALLOWED_DEVICES: frozenset[str] = frozenset({"mps", "cpu"})


def _assert_yes_no_single_token(tokenizer: Any) -> None:
    """The yes/no scoring takes `tokenizer.encode(...)[0]`; if the
    tokenizer splits 'yes' or 'no' across tokens (BPE merge update,
    whitespace policy change), the `[0]` slice would silently grab the
    wrong id and scores would be miscalibrated with no error signal.
    Verified at warmup so the failure is loud and tied to its cause.
    """
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(
            f"Qwen3-Reranker tokenizer encodes 'yes' to {yes_ids} and 'no' "
            f"to {no_ids}; expected single tokens. The scoring code takes "
            "[0] from each, which would be wrong for multi-token encodings. "
            "This usually means the upstream tokenizer revision drifted; "
            "pin a known-good revision or update the scoring logic."
        )


def _ensure_qwen3_reranker_warmup(
    model_id: str, device: str, dtype: str
) -> tuple[Any, Any]:
    """Load a Qwen3-Reranker once and return `(model, tokenizer)`.

    Idempotent via `_RERANKER_MODELS` keyed by `(model_id, device, dtype)`.
    Raises with a clear remediation on:

    - `device` not in `_ALLOWED_DEVICES` (e.g., `cuda`, `xla` — not yet
      wired up; fail before torch surfaces a cryptic backend error).
    - `device="mps"` requested on a non-Apple-Silicon host.
    - `transformers` / `torch` import failure.
    - HF cache miss / OSError with no network reachable (point at
      `make setup-reranker`).
    """
    if device not in _ALLOWED_DEVICES:
        raise ValueError(
            f"device must be one of {sorted(_ALLOWED_DEVICES)}; got {device!r}. "
            "Add the device to the tier table and `_ALLOWED_DEVICES` "
            "together if a new backend is wired up."
        )

    key = (model_id, device, dtype)
    cached = _RERANKER_MODELS.get(key)
    if cached is not None:
        return cached

    if device == "mps" and not (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    ):
        raise RuntimeError(
            f"Qwen3-Reranker tier requested device={device!r} but host is "
            f"system={platform.system()!r} machine={platform.machine()!r}. "
            "Construct with tier='ci-cpu' on non-Apple-Silicon hosts."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"Qwen3-Reranker {model_id!r} requested but `transformers` / "
            "`torch` could not be imported. Run `uv sync` to install "
            "core deps."
        ) from exc

    repo = _RERANKER_HF_REPOS.get(model_id)
    if repo is None:
        raise ValueError(
            f"No HF repo mapping for Qwen3-Reranker model {model_id!r}; "
            f"known: {sorted(_RERANKER_HF_REPOS)}"
        )

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    # `from_pretrained` allocates weights to host RAM first, then
    # `.to(device)` copies them to the target device — peak memory is
    # roughly 2x model size during load. The lower-peak alternative
    # (`device_map={"": device}`) would require adding `accelerate` as
    # a runtime dependency, which costs more than the niche
    # memory-constrained-Mac 4B case we'd avoid.
    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(repo)
        model: Any = AutoModelForCausalLM.from_pretrained(repo, dtype=torch_dtype)
    except OSError as exc:
        # OSError is what HF/transformers raise for missing local files
        # and network failures (`LocalEntryNotFoundError`,
        # `OfflineModeIsEnabled`, etc.). Other exception types (ValueError
        # for bad config, RuntimeError for backend issues) propagate
        # unchanged so the user sees the actual cause.
        raise RuntimeError(
            f"Qwen3-Reranker {model_id!r} could not be loaded — likely a "
            "HuggingFace cache miss with no network reachable. Populate "
            "the cache with:\n"
            "    make setup-reranker\n"
            f"(repo: {repo})"
        ) from exc
    model = model.to(device).eval()
    _assert_yes_no_single_token(tokenizer)

    _RERANKER_MODELS[key] = (model, tokenizer)
    return _RERANKER_MODELS[key]


def _truncate_passage_to_budget(
    *, passage: str, tokenizer: Any, budget: int
) -> str:
    """Truncate `passage` so its tokenization is at most `budget` tokens.

    Returns the input unchanged when it already fits — saves a decode
    round-trip on the common path. The Qwen3-Reranker prompt is
    assembled around a fixed prefix + suffix (the chat template);
    right-truncating the WHOLE prompt would drop the assistant suffix
    that carries the yes/no decision token, silently making scores
    uninformative. Pre-truncating only the passage preserves the
    suffix and keeps the model's last-token logits meaningful.
    """
    ids = tokenizer.encode(passage, add_special_tokens=False)
    if len(ids) <= budget:
        return passage
    truncated_ids = ids[:budget]
    decoded: str = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return decoded


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

    @cached_property
    def _model_and_tokenizer(self) -> tuple[Any, Any]:
        return _ensure_qwen3_reranker_warmup(self._model_id, self._device, self._dtype)

    @cached_property
    def _yes_no_token_ids(self) -> tuple[int, int]:
        # Resolved once per instance — the warmup helper has already
        # asserted single-token encoding.
        _, tokenizer = self._model_and_tokenizer
        return (
            int(tokenizer.encode("yes", add_special_tokens=False)[0]),
            int(tokenizer.encode("no", add_special_tokens=False)[0]),
        )

    @cached_property
    def _non_passage_token_count(self) -> int:
        # Count the tokens consumed by the prompt scaffold (prefix +
        # instruction + query placeholder + "<Document>: " + suffix) so
        # `_truncate_passage_to_budget` can size the remaining budget
        # for the passage portion. `query` varies per call so the
        # budget is recomputed per `score()` — this property only
        # caches the QUERY-INDEPENDENT scaffold cost.
        _, tokenizer = self._model_and_tokenizer
        scaffold = (
            _RERANKER_PROMPT_PREFIX
            + f"<Instruct>: {_RERANKER_INSTRUCTION}\n"
            + "<Query>: \n"
            + "<Document>: "
            + _RERANKER_PROMPT_SUFFIX
        )
        return len(tokenizer.encode(scaffold, add_special_tokens=False))

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score `(query, passage)` pairs with the Qwen3-Reranker yes/no
        head; return per-passage `p(yes) / (p(yes) + p(no))`.

        Positional-friendly so `scorer=reranker.score` binds directly
        when passed to `rerank()` — no lambda wrapper required.

        Deterministic on CPU fp32; eval() mode, no sampling, no
        dropout. MPS fp16 is deterministic in practice for the relevant
        kernels but `torch.use_deterministic_algorithms` is NOT set
        (the flag is process-global and would spill into unrelated
        tests). Score magnitudes are only comparable within a single
        `(tier, model, dtype, device)` — cross-tier comparison is
        forbidden and the `reranker_version` token records the score
        space for downstream guards.
        """
        if not passages:
            return []
        import torch

        model, tokenizer = self._model_and_tokenizer
        yes_id, no_id = self._yes_no_token_ids
        # Reserve scaffold + query tokens; rest is the per-passage budget.
        # Keep an 8-token safety margin against tokenizer joining edge
        # cases (e.g., a passage that joins with a preceding token on
        # the prompt seam).
        query_tokens = len(tokenizer.encode(query, add_special_tokens=False))
        passage_budget = max(
            16, _RERANKER_MAX_LENGTH - self._non_passage_token_count - query_tokens - 8
        )
        scores: list[float] = []
        with torch.no_grad():
            for passage in passages:
                fitted_passage = _truncate_passage_to_budget(
                    passage=passage, tokenizer=tokenizer, budget=passage_budget
                )
                prompt = (
                    _RERANKER_PROMPT_PREFIX
                    + f"<Instruct>: {_RERANKER_INSTRUCTION}\n"
                    + f"<Query>: {query}\n"
                    + f"<Document>: {fitted_passage}"
                    + _RERANKER_PROMPT_SUFFIX
                )
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(self._device)
                out = model(**inputs)
                last_logits = out.logits[0, -1, :]
                yes_logit = float(last_logits[yes_id])
                no_logit = float(last_logits[no_id])
                m = max(yes_logit, no_logit)
                ey = math.exp(yes_logit - m)
                en = math.exp(no_logit - m)
                scores.append(ey / (ey + en))
        return scores


@dataclass(frozen=True)
class RerankHit:
    """One reranked hit, carrying both the reranker score and the prior
    RRF context.

    Score scales differ — `score` is the reranker yes-probability
    ∈ [0, 1]; `prev_rrf_score` is the RRF fusion score (typically
    ~0.01-0.05 for `rrf_k=60`). They are NOT directly comparable; the
    reranker score is the authoritative ranking signal post-rerank,
    `prev_rrf_score` is provenance.

    `reranker_version` records the score space (`{tier}:{model}:{tag}`)
    so a downstream worker persisting reranked rows can refuse to mix
    them across tiers — the row-stamp analogue of `embed_model_version`.
    Defaults to `None` when callers do not stamp it (e.g., unit tests
    using a synthetic scorer).
    """

    parent: ParentChunk
    score: float
    prev_rrf_score: float
    prev_rank: int
    reranker_version: str | None = None


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
    reranker_version: str | None = None,
) -> list[RerankHit]:
    """Reorder `hits` by `scorer(query, [h.parent.text for h in hits])` and
    return the top `top_k`.

    Tie-break for deterministic output: descending reranker score, then
    descending prior RRF score, then ascending `doc_id`. The reranker's
    yes-probability is dense-floating-point and ties are statistically
    rare, but they DO happen on identical passages (e.g., two filings
    that quote the same boilerplate); the fixed tie-break makes the
    output order reproducible across runs.

    Pass `reranker_version=reranker.reranker_version` to stamp the
    score-space token on every `RerankHit`. Required for any downstream
    consumer that persists scores — the token is the cross-tier guard
    against silent score-space mixing.
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
            reranker_version=reranker_version,
        )
        for orig_idx, (h, s) in indexed[:top_k]
    ]


__all__ = [
    "ALLOWED_TIERS",
    "RERANKER_VERSION_TAG",
    "Qwen3Reranker",
    "RerankHit",
    "ScorerFn",
    "_assert_yes_no_single_token",
    "_ensure_qwen3_reranker_warmup",
    "_truncate_passage_to_budget",
    "rerank",
    "reranker_version",
]
