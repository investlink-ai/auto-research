"""Embedding adapter for the RAG retrieval layer.

Backend selection is **explicit** — the caller passes one of:

- `backend="voyage"` — production cloud (`voyage-finance-2` per ADR D1).
- `backend="bge"` — cross-platform in-process `bge-small-en-v1.5`,
  used by Linux CI and airgapped dev.
- `backend="qwen3-mlx"` — Apple-Silicon-only in-process Qwen3-Embedding
  via `mlx_embeddings` (`Qwen3-Embedding-0.6B` for dev, `-4B` as a
  credible offline alternative to Voyage at backfill scope on
  dedicated Apple-Silicon hardware).

There is no env-var-driven implicit fallback: a missing
`VOYAGE_API_KEY` does NOT silently switch the adapter to BGE — it
raises when the Voyage client is first constructed, surfacing the
misconfiguration loudly. The Qwen3-MLX path raises a clear
`RuntimeError` on non-Apple-Silicon hosts pointing at the
cross-platform backends. Workers / CLI entry points read
`EMBEDDING_BACKEND` themselves and pass the choice in.

The backend is locked for the adapter's lifetime. There is no
mid-run switch on quota or any other Voyage error — a single corpus
must live in a single vector space; same-dim does not imply same
space (Voyage and Qwen3-0.6B both emit 1024-dim vectors but in
incompatible coordinate systems), so dense retrieval over a
mixed-backend corpus silently degrades. On
`voyageai.error.RateLimitError` the call propagates; operational
handling (retry-with-backoff, circuit breaking, quota alerting)
lives at the worker layer.
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

from auto_research.extract.chunking import CHUNKER_VERSION
from auto_research.extract.chunking_contextual import ContextualChildChunk
from auto_research.extract.materialization import (
    ActiveMaterialization,
    compute_materialization_version,
    read_active_materialization,
    versioned_table_name,
)
from auto_research.extract.prompts.contextual_chunk import (
    CONTEXTUAL_CHUNK_PROMPT_VERSION,
)
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

# Qwen3-Embedding MLX backend — Apple-Silicon-only, in-process. Two
# variants: 0.6B (1024-dim, ~600 MB, dev/test default) and 4B (2560-dim,
# ~8 GB, dedicated-hardware deployment alternative to Voyage at backfill
# scope). Same vector space ≠ same dim; treat each (backend, model) as
# its own corpus per the embedding-vector-space-consistency rule.
ALLOWED_MLX_QWEN3_MODELS: frozenset[str] = frozenset({
    "Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B",
})
DEFAULT_MLX_QWEN3_MODEL = "Qwen3-Embedding-0.6B"

_MODEL_DIM: dict[str, int] = {
    "voyage-finance-2": 1024,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "bge-small-en-v1.5": 384,
    "Qwen3-Embedding-0.6B": 1024,
    "Qwen3-Embedding-4B": 2560,
}

NARRATIVE_DOC_TYPES: frozenset[str] = frozenset({"10-K", "10-Q", "transcript"})
_BGE_MODEL_NAME = "bge-small-en-v1.5"
_BGE_HF_ID = f"BAAI/{_BGE_MODEL_NAME}"
_PER_CORPUS_STORE = "_corpus_narrative"
_WORKER = "embeddings"

EMBED_MODEL_VERSION_TAG: str = "v1"
"""Bump when the embed-model contract changes.

The pure-function contract for embedding is `(text, backend, model) →
vector`. Bump triggers:

- A vendor opaquely changes the model behind the same name (Voyage
  occasionally re-weights without renaming; HuggingFace
  `bge-small-en-v1.5` could be re-uploaded under the same id).
- Query / document side prefixing changes (Qwen3 instruction prefix
  edit, BGE `query:` prefix policy flip).
- Tokenizer truncation budget changes (`_QWEN3_MAX_LENGTH` adjustment).
- L2-normalization policy flip (currently all backends emit normalized
  vectors; turning that off would change cosine semantics).

Non-triggers:
- Choosing a different backend / model at call time — the resulting
  `embed_model_version()` token differs by `backend`/`model` already.
- Adding a new allowed model to `ALLOWED_VOYAGE_MODELS` /
  `ALLOWED_MLX_QWEN3_MODELS` (existing model contracts unchanged).

This tag is the embed-model analogue of `*_PROMPT_VERSION` and
`CHUNKER_VERSION` and is covered by the same `bump-prompt-version`
skill workflow. Per AGENTS.md INV-6, exactly one upstream version is
bumped per PR; downstream row materialization stamps the resulting
token transitively.
"""


def embed_model_version(
    backend: Literal["voyage", "bge", "qwen3-mlx"], model: str
) -> str:
    """Stable token identifying an embedding vector space.

    Returns `"{backend}:{model}:{EMBED_MODEL_VERSION_TAG}"`. Including the
    backend distinguishes future overlap (e.g., a Qwen3 weight served by
    both MLX and `transformers` would diverge on quantization-driven
    tiny float differences). Including the tag lets us invalidate
    downstream caches / row metadata transitively even when neither
    backend nor model id changed (vendor opaque re-upload).
    """
    return f"{backend}:{model}:{EMBED_MODEL_VERSION_TAG}"

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

# One-shot Qwen3-MLX warmup, same singleton pattern as BGE but keyed
# by model id so 0.6B and 4B can both be cached if both are exercised
# in the same process. Each cache value is the `(model, tokenizer)`
# pair `mlx_embeddings.load(...)` returns. Apple-Silicon-only; the
# platform check inside `_ensure_qwen3_warmup` raises on Linux /
# Intel-Mac with a clear remediation pointing at the cross-platform
# backends.
_QWEN3_MODELS: dict[str, Any] = {}

# Map allowlist names → HuggingFace repo ids for the MLX-converted
# weights. The mlx-community org publishes quantized variants only;
# `mxfp8` (microscaled 8-bit float) is the best-quality 8-bit
# quantization and matches the published native dims (1024 / 2560)
# observed end-to-end.
_QWEN3_MLX_HF_REPOS: dict[str, str] = {
    "Qwen3-Embedding-0.6B": "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
    "Qwen3-Embedding-4B": "mlx-community/Qwen3-Embedding-4B-mxfp8",
}

# Asymmetric-encoder instruction prefix applied on the query side only,
# per the Qwen3-Embedding model card. Document-side text is encoded
# unprefixed. The task description is part of the embedding signal —
# the model card explicitly recommends domain-tailoring it. This value
# matches the corpus the adapter actually serves (SEC filings + earnings
# transcripts + analyst materials); revisit when a Ragas / DeepEval
# baseline gives a measurable handle for tuning.
_QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a financial research query, retrieve relevant "
    "passages from SEC filings, earnings transcripts, and analyst "
    "materials\nQuery: "
)

# Tokenizer truncation budget for `mlx_embeddings.generate`. The upstream
# default is 512, which silently truncates long passages even though
# Qwen3-Embedding's native context is 32K. 2048 is generous enough for
# the contextual-chunking pattern (LLM-generated context prefix + child
# text) without paying the memory cost of pinning to full 32K on every
# batch.
_QWEN3_MAX_LENGTH = 2048

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

# Voyage USD price per 1M tokens, keyed by model id. Used only by
# `EmbeddingAdapter.reembed_*` dry-run cost estimation; nothing in the
# normal embed path reads this. Source: voyageai.com/pricing as of the
# project's account onboarding (constrained tier, finance-2 + 3.x family
# at $0.12/MTok). Bump or split when Voyage publishes a different rate;
# missing entries cause dry-run to report `unknown` rather than crash.
#
# BGE / Qwen3-MLX backends are in-process and report $0.0 — encoded as
# absence from this dict and special-cased in the dry-run formatter.
_VOYAGE_USD_PER_MTOK: dict[str, float] = {
    "voyage-finance-2": 0.12,
    "voyage-3": 0.06,
    "voyage-3-large": 0.18,
    "voyage-3.5": 0.06,
    "voyage-4": 0.12,
    "voyage-4-large": 0.18,
}

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


def _ensure_qwen3_warmup(model_id: str) -> Any:
    """Load a Qwen3-Embedding MLX model once and return `(model, tokenizer)`.

    Idempotent via the module-level `_QWEN3_MODELS` dict keyed by
    `model_id`, so multiple adapters in the same process — and the
    0.6B / 4B variants — each pay the load cost at most once.

    Raises `RuntimeError` up front on non-Apple-Silicon hosts with a
    remediation pointing at the cross-platform backends. Mirrors the
    `_ensure_bge_warmup` HF-cache-miss error: when MLX is reachable but
    the weights aren't yet cached, the first call downloads them; a
    network-blocked cache miss surfaces `RuntimeError` naming
    `make setup-mlx` (which pre-pulls the 0.6B weights, and the 4B
    weights when `QWEN3_FULL=1`) as the fix.
    """
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Qwen3-MLX embedding backend requires Apple Silicon "
            f"(Darwin/arm64); detected system={platform.system()!r} "
            f"machine={platform.machine()!r}. Construct the adapter "
            "with backend='voyage' or backend='bge' on this host."
        )

    cached = _QWEN3_MODELS.get(model_id)
    if cached is not None:
        return cached

    repo = _QWEN3_MLX_HF_REPOS.get(model_id)
    if repo is None:
        raise ValueError(
            f"No HF repo mapping for Qwen3-MLX model {model_id!r}; "
            f"known: {sorted(_QWEN3_MLX_HF_REPOS)}"
        )

    try:
        from mlx_embeddings import load as _mlx_load
    except ImportError as exc:
        raise RuntimeError(
            f"Qwen3-MLX model {model_id!r} requested but the "
            "`mlx-embeddings` extra is not installed. Install it with:\n"
            "    uv sync --extra mlx\n"
            "On non-Apple-Silicon hosts use backend='voyage' or "
            "backend='bge' instead."
        ) from exc

    # Network / cache failures during `_mlx_load` propagate with their
    # own message so operators don't get pointed at the wrong fix. The
    # `make setup-mlx` target populates the HF cache one-shot; the
    # underlying HF/MLX error already names the failing repo.
    loaded = _mlx_load(repo)

    _QWEN3_MODELS[model_id] = loaded
    return loaded


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
    # `chunker_version` / `contextual_prompt_version` / `embed_model_version`
    # stamp each row with the three pure-function contracts that produced
    # it. At backfill scope these enable point-in-time provenance queries
    # ("which rows came from chunker v3?") and back the materialization-
    # versioned tables follow-up; at present they are write-only audit
    # metadata. Same-name string values use `pa.string()` rather than
    # dictionary-encoding — cardinality is tiny but per-row cost stays
    # negligible at the corpus sizes LanceDB handles here.
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
        ("chunker_version", pa.string()),
        ("contextual_prompt_version", pa.string()),
        ("embed_model_version", pa.string()),
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


def resolve_backend_from_env() -> Literal["voyage", "bge", "qwen3-mlx"]:
    """Read `EMBEDDING_BACKEND` from the environment and validate.

    Workers / CLI entry points call this once at startup and pass the
    explicit choice into `EmbeddingAdapter(backend=...)`. The adapter
    itself never reads this env var — selection is the caller's
    responsibility, missing config is a loud error not a silent
    fallback.
    """
    raw = os.environ.get("EMBEDDING_BACKEND")
    if raw not in {"voyage", "bge", "qwen3-mlx"}:
        raise RuntimeError(
            "EMBEDDING_BACKEND env var must be set to 'voyage', 'bge', "
            f"or 'qwen3-mlx'; got {raw!r}. Set explicitly — there is "
            "no default fallback."
        )
    return raw  # type: ignore[return-value]


class EmbeddingAdapter:
    def __init__(
        self,
        *,
        backend: Literal["voyage", "bge", "qwen3-mlx"],
        rag_root: Path = Path("data/rag"),
        voyage_model: str | None = None,
        mlx_qwen3_model: str | None = None,
    ) -> None:
        """Construct an adapter bound to an explicitly-chosen backend.

        `backend` is required — no env-var inference, no default. Pass
        `"voyage"` for production (`voyage-finance-2` by default, or
        the model named in `voyage_model` / `$VOYAGE_MODEL`),
        `"bge"` for the cross-platform in-process `bge-small-en-v1.5`
        fallback, or `"qwen3-mlx"` for the Apple-Silicon-only
        Qwen3-Embedding MLX backend (`Qwen3-Embedding-0.6B` by default
        or the variant named in `mlx_qwen3_model`).

        Model kwargs are mutually exclusive with non-matching backends:
        `voyage_model` is only valid with `backend="voyage"` and
        `mlx_qwen3_model` only with `backend="qwen3-mlx"`. Passing the
        wrong pairing is rejected so the caller's intent stays
        unambiguous.
        """
        if backend == "voyage":
            if mlx_qwen3_model is not None:
                raise ValueError(
                    "mlx_qwen3_model is only valid when "
                    "backend='qwen3-mlx'; got backend='voyage' with "
                    f"mlx_qwen3_model={mlx_qwen3_model!r}"
                )
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
            self._backend: Literal["voyage", "bge", "qwen3-mlx"] = "voyage"
            self._model = resolved
        elif backend == "bge":
            if voyage_model is not None:
                raise ValueError(
                    "voyage_model is only valid when backend='voyage'; "
                    f"got backend='bge' with voyage_model={voyage_model!r}"
                )
            if mlx_qwen3_model is not None:
                raise ValueError(
                    "mlx_qwen3_model is only valid when "
                    "backend='qwen3-mlx'; got backend='bge' with "
                    f"mlx_qwen3_model={mlx_qwen3_model!r}"
                )
            self._backend = "bge"
            self._model = BGE_MODEL_ID
        elif backend == "qwen3-mlx":
            if voyage_model is not None:
                raise ValueError(
                    "voyage_model is only valid when backend='voyage'; "
                    f"got backend='qwen3-mlx' with "
                    f"voyage_model={voyage_model!r}"
                )
            mlx_resolved = mlx_qwen3_model or DEFAULT_MLX_QWEN3_MODEL
            if mlx_resolved not in ALLOWED_MLX_QWEN3_MODELS:
                raise ValueError(
                    f"mlx_qwen3_model={mlx_resolved!r} not in "
                    f"{sorted(ALLOWED_MLX_QWEN3_MODELS)}"
                )
            self._backend = "qwen3-mlx"
            self._model = mlx_resolved
        else:
            raise ValueError(
                f"backend must be 'voyage', 'bge', or 'qwen3-mlx'; "
                f"got {backend!r}"
            )
        self._rag_root = rag_root
        _log.info(
            "embedding_adapter_init backend=%s model=%s",
            self._backend,
            self._model,
        )

    @property
    def backend(self) -> Literal["voyage", "bge", "qwen3-mlx"]:
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    @cached_property
    def embed_model_version(self) -> str:
        """Frozen at construction so an adapter's identity is stable across
        the lifetime of the process — bumping the module-level
        `EMBED_MODEL_VERSION_TAG` after the adapter exists does NOT change
        what this adapter's `embed_model_version` reports. Pairs naturally
        with `materialization_version` (also cached) so reads-vs-active-
        pointer mismatch checks are well-defined per adapter rather than
        leaking the current module state across instances."""
        return embed_model_version(self._backend, self._model)

    @cached_property
    def materialization_version(self) -> str:
        """The 8-hex-char hash of this adapter's `(chunker, contextual-prompt,
        embed-model)` contract triple — the namespace this adapter's writes
        land in and the fallback read namespace when no active pointer is
        promoted. Always derived from the three module-level version tokens
        so changes propagate transitively (the orthogonal-cache-keys
        contract from #67).
        """
        return compute_materialization_version(
            CHUNKER_VERSION,
            CONTEXTUAL_CHUNK_PROMPT_VERSION,
            self.embed_model_version,
        )

    def _resolve_read_active(self) -> ActiveMaterialization | None:
        """Read the on-disk active pointer if present, validating that its
        `embed_model_version` matches this adapter's.

        Mismatch raises loudly per the embedding-vector-space-consistency
        rule: querying with adapter-encoder B against rows produced by
        encoder A silently degrades retrieval and is exactly the
        backfill-scale failure mode this issue exists to prevent. The
        cleanest operational answer is "construct an adapter that matches
        the active pointer's embed model" — the error surfaces that
        guidance immediately.
        """
        active = read_active_materialization(self._rag_root)
        if active is None:
            return None
        if active.embed_model_version != self.embed_model_version:
            raise RuntimeError(
                "active materialization mismatch: pointer is "
                f"embed_model_version={active.embed_model_version!r} but this "
                f"adapter produces {self.embed_model_version!r}. Querying "
                "across vector spaces silently degrades retrieval. "
                "Construct an adapter matched to the active pointer, "
                "or run `auto-research extract list-materializations` to "
                "see what's available."
            )
        return active

    def _read_table_name(self, base: str) -> str:
        """Resolve the table name for a READ operation.

        Uses the active pointer's version when promoted, else falls back to
        this adapter's own materialization version. The fallback lets fresh
        installs and tests embed-then-query in one session without a manual
        promote step; production deployments always have an active pointer
        (the migration script seeds one).
        """
        active = self._resolve_read_active()
        version = active.version if active is not None else self.materialization_version
        return versioned_table_name(base, version)

    def _write_table_name(self, base: str) -> str:
        """Resolve the table name for a WRITE operation.

        Writes always land in `{base}__{self.materialization_version}` —
        the build path is independent of the active pointer per the issue
        spec ("Build path ignores it (writes to its own version
        namespace)"). Building a new materialization never perturbs
        live-query reads against the currently-active one.
        """
        return versioned_table_name(base, self.materialization_version)

    @cached_property
    def _vector_dim(self) -> int:
        return _MODEL_DIM[self._model]

    @cached_property
    def _bge(self) -> Any:
        # Routes through the module-level singleton so every adapter in
        # the process shares one warm model; surfaces a clear remediation
        # error when the HF cache is empty and the network is blocked.
        return _ensure_bge_warmup()

    @cached_property
    def _qwen3(self) -> Any:
        # Same singleton routing as `_bge`, but the cache key is the
        # model id — 0.6B and 4B live in different namespaces, both
        # entered through `_ensure_qwen3_warmup`. Returns the
        # `(model, tokenizer)` pair from `mlx_embeddings.load`.
        return _ensure_qwen3_warmup(self._model)

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

        if self._backend == "qwen3-mlx":
            # Qwen3-Embedding is asymmetric: the query side wants an
            # instruction prefix per the model card; the document side
            # is unprefixed. Mirrors the Voyage `input_type` semantics
            # from the same call site below. `text_embeds` from the
            # MLX `generate` call is already L2-normalized.
            #
            # Resolve `self._qwen3` FIRST so the platform check inside
            # `_ensure_qwen3_warmup` raises with its friendly remediation
            # before the `mlx_embeddings` import runs — otherwise a
            # Linux host without the `[mlx]` extra installed would leak
            # a cryptic `ModuleNotFoundError` instead.
            model, tokenizer = self._qwen3
            from mlx_embeddings import generate as _mlx_generate

            if input_type == "query":
                prepared = [_QWEN3_QUERY_INSTRUCTION + t for t in texts]
            else:
                prepared = list(texts)
            out = _mlx_generate(
                model, tokenizer, prepared, max_length=_QWEN3_MAX_LENGTH
            )
            # `out.text_embeds` is an `mx.array` of shape (batch, dim);
            # numpy conversion via `np.asarray` triggers materialization
            # off the MLX device.
            return np.asarray(out.text_embeds, dtype=np.float32)

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
        embed_version = self.embed_model_version
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
                "chunker_version": CHUNKER_VERSION,
                "contextual_prompt_version": CONTEXTUAL_CHUNK_PROMPT_VERSION,
                "embed_model_version": embed_version,
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
            span.set_attribute(
                "embedding.materialization_version", self.materialization_version
            )
            try:
                texts = [c.embedding_text for c in chunks]
                vectors = self._encode(texts)
                rows = self._rows(chunks, vectors)

                self._rag_root.mkdir(parents=True, exist_ok=True)
                db = lancedb.connect(self._rag_root)
                schema = _schema(self._vector_dim)

                per_doc_tbl = db.create_table(
                    self._write_table_name(doc_id),
                    data=rows,
                    schema=schema,
                    mode="overwrite",
                )
                per_doc_tbl.create_fts_index("text", **_FTS_INDEX_KWARGS)

                narrative_rows = [r for r in rows if r["doc_type"] in NARRATIVE_DOC_TYPES]
                if narrative_rows:
                    corpus_name = self._write_table_name(_PER_CORPUS_STORE)
                    if corpus_name in db.table_names():
                        # LanceDB updates the FTS index incrementally on `.add()`.
                        db.open_table(corpus_name).add(narrative_rows)
                    else:
                        corpus_tbl = db.create_table(
                            corpus_name, data=narrative_rows, schema=schema
                        )
                        corpus_tbl.create_fts_index("text", **_FTS_INDEX_KWARGS)
                span.set_attribute("embedding.narrative_count", len(narrative_rows))
                span.set_attribute("extract.outcome", "success")
            except Exception as exc:
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

    def _existing_vector_dim(self, table: Any) -> int:
        """Read the fixed-size vector column dim from a LanceDB table schema.

        The vector column is declared as `pa.list_(pa.float32(), N)` at
        write-time (see `_schema`); arrow surfaces `N` as the field type's
        `list_size`. Used by the reembed path to refuse cross-dim swaps
        loudly instead of silently corrupting a mixed-vector-space corpus.
        """
        field_type = table.schema.field("vector").type
        return int(field_type.list_size)

    def _assert_dim_compatible(self, table: Any, label: str) -> None:
        existing = self._existing_vector_dim(table)
        if existing != self._vector_dim:
            raise RuntimeError(
                f"reembed dim mismatch for {label}: existing vectors are "
                f"{existing}-dim but current adapter "
                f"(backend={self._backend!r} model={self._model!r}) produces "
                f"{self._vector_dim}-dim. Same-dim cross-encoder swaps are the "
                "caller's responsibility, but a dim change means a new vector "
                "space — build a new corpus instead of re-embedding in place."
            )

    def _rows_from_existing(
        self, df: Any, vectors: NDArray[np.float32]
    ) -> list[dict[str, object]]:
        """Build row dicts from an existing LanceDB dataframe + new vectors.

        Metadata columns + `chunker_version` + `contextual_prompt_version`
        are preserved byte-for-byte from the source row — by definition this
        path is encoder-only, so those upstream-contract stamps must NOT be
        re-stamped to the current module-level values (that would lie about
        which chunker contract produced the row). Only `vector` and
        `embed_model_version` are updated.
        """
        embed_version = self.embed_model_version
        rows: list[dict[str, object]] = []
        for (_, row), vec in zip(df.iterrows(), vectors, strict=True):
            rows.append({
                "text": row["text"],
                "vector": vec.tolist(),
                "ticker": row["ticker"],
                "filing_date": row["filing_date"],
                "fiscal_period": row["fiscal_period"],
                "doc_type": row["doc_type"],
                "doc_id": row["doc_id"],
                "parent_id": row["parent_id"],
                "section_name": row["section_name"],
                "chunker_version": row["chunker_version"],
                "contextual_prompt_version": row["contextual_prompt_version"],
                "embed_model_version": embed_version,
            })
        return rows

    def _reembed_table(
        self,
        *,
        source_table: str,
        dest_table: str,
        span_doc_id: str | None,
    ) -> tuple[int, list[dict[str, object]]]:
        """Shared body for `reembed_doc` / `reembed_corpus`.

        Reads the `text` column from `source_table`, re-encodes with the
        adapter's current backend/model, writes to `dest_table` under
        the schema for this adapter's vector dim, and builds the FTS
        index with `_FTS_INDEX_KWARGS`. Source and dest may name the
        same table (legacy in-place semantics) or different tables (the
        materialization-versioned path, where source lives in the active
        namespace and dest lives in this adapter's own namespace).

        Returns `(rows_reencoded, new_rows)` — callers (specifically
        `reembed_doc`) reuse the encoded rows to propagate vectors into
        the per-corpus narrative store without re-encoding.

        Empty source short-circuits BEFORE the dim check: there is
        nothing to corrupt in a zero-row table, so refusing it on dim
        grounds would surprise operators who hit an empty placeholder
        during a re-embed sweep. Non-empty source still fails loudly on
        dim drift relative to the adapter's current model.

        `span_doc_id` is only set for per-doc reembeds — the per-corpus
        store contains rows from many docs, so the OTel `extract.doc_id`
        attribute is omitted for it (matching the behavior of `embed`
        which only sets it on the per-doc surface).
        """
        with _tracer.start_as_current_span("extract.reembed") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.backend", self._backend)
            span.set_attribute("embedding.model", self._model)
            span.set_attribute(
                "embedding.materialization_version", self.materialization_version
            )
            span.set_attribute("embedding.store", dest_table)
            span.set_attribute("embedding.source_store", source_table)
            if span_doc_id is not None:
                span.set_attribute("extract.doc_id", span_doc_id)
            try:
                db = lancedb.connect(self._rag_root)
                tbl = db.open_table(source_table)
                df = tbl.to_pandas()
                n = len(df)
                if n == 0:
                    span.set_attribute("embedding.rows_reencoded", 0)
                    span.set_attribute("embedding.narrative_count", 0)
                    span.set_attribute("extract.outcome", "success")
                    return (0, [])
                # Dim guard applies to the cross-namespace case too: if the
                # source's vector column is 1024-dim and the adapter would
                # produce 2560-dim, the destination namespace would contain
                # rows of one dim while another doc embedded in the same
                # namespace produces a different dim — corruption. Raise
                # before writing anything.
                self._assert_dim_compatible(tbl, source_table)
                texts = df["text"].tolist()
                vectors = self._encode(texts, input_type="document")
                new_rows = self._rows_from_existing(df, vectors)
                schema = _schema(self._vector_dim)
                new_tbl = db.create_table(
                    dest_table, data=new_rows, schema=schema, mode="overwrite"
                )
                new_tbl.create_fts_index("text", **_FTS_INDEX_KWARGS)
                narrative_count = sum(
                    1 for r in new_rows if r["doc_type"] in NARRATIVE_DOC_TYPES
                )
                span.set_attribute("embedding.rows_reencoded", n)
                span.set_attribute("embedding.narrative_count", narrative_count)
                span.set_attribute("extract.outcome", "success")
                return (n, new_rows)
            except Exception as exc:
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

    def _propagate_to_corpus(
        self, new_rows: list[dict[str, object]]
    ) -> int:
        """Push fresh vectors for the narrative-eligible rows of one doc into
        the per-corpus narrative store FOR THIS ADAPTER'S MATERIALIZATION
        VERSION by deleting the doc's existing corpus rows and re-
        inserting the new ones in a single LanceDB transaction-equivalent
        pair.

        Encoder-only: no `_encode` call here. Vectors are copied from
        `new_rows` (already computed by the caller's `_reembed_table`
        invocation).

        Target table is `_corpus_narrative__{self.materialization_version}`
        — the destination namespace, not the active one. The active-
        namespace corpus is left untouched so a partial reembed can't
        corrupt the live query surface.

        No-ops in three cases:
        - `new_rows` is empty (caller just reembedded an empty per-doc
          table).
        - None of `new_rows` are narrative-eligible (S-1 / S-3 / 8-K /
          DEF 14A doc types).
        - The destination corpus table does not exist yet (no narrative
          doc has ever been reembedded into this materialization). Per
          the original `embed()` contract, the corpus is lazily created
          when the first narrative chunk lands; we honor that — re-
          embedding a doc that is not narrative-eligible must not
          create an empty corpus, and the first narrative reembed of a
          materialization can't propagate yet (the build path is
          responsible for creating the corpus on the first per-doc
          embed; the reembed path appends).

        Implementation note: `merge_insert(on=...)` looked tempting but
        no LanceDB column is a unique chunk identifier — `parent_id` is
        shared across the multiple child chunks of one parent, and
        `(parent_id, text)` would require a composite join key that
        LanceDB's merge_insert handles awkwardly with duplicate-key
        source rows. Delete-by-doc-id + insert is the simplest
        primitive that matches the embed-time write semantics exactly.
        Single-quoted SQL literal is safe because `doc_id` is validated
        at chunking time (no embedded quotes in the doc_id grammar) and
        we further reject any `'` defensively.
        """
        narrative_rows = [
            r for r in new_rows if r["doc_type"] in NARRATIVE_DOC_TYPES
        ]
        if not narrative_rows:
            return 0
        db = lancedb.connect(self._rag_root)
        corpus_name = self._write_table_name(_PER_CORPUS_STORE)
        # All narrative_rows share the same doc_id by construction
        # (caller is _reembed_table for a single per-doc table).
        doc_id = str(narrative_rows[0]["doc_id"])
        if "'" in doc_id:
            raise ValueError(
                f"doc_id {doc_id!r} contains a single quote; refusing to "
                "build the corpus-delete SQL predicate. Sanitize upstream."
            )
        # Reuse the same fixed-size-list schema the original embed() path
        # writes; pa.Table.from_pylist needs the schema to lock in
        # `pa.list_(pa.float32(), dim)` instead of inferring a variable-
        # length list from the row dicts.
        schema = _schema(self._vector_dim)
        if corpus_name not in db.table_names():
            # Destination corpus doesn't exist yet — first narrative reembed
            # into this materialization. Create it, mirroring the embed-time
            # lazy-create pattern. Build the FTS index here too so any
            # subsequent BM25 query against the destination materialization's
            # corpus has the index it needs.
            corpus_tbl = db.create_table(
                corpus_name, data=narrative_rows, schema=schema
            )
            corpus_tbl.create_fts_index("text", **_FTS_INDEX_KWARGS)
        else:
            corpus_tbl = db.open_table(corpus_name)
            corpus_tbl.delete(f"doc_id = '{doc_id}'")
            new_data = pa.Table.from_pylist(narrative_rows, schema=schema)
            corpus_tbl.add(new_data)
        return len(narrative_rows)

    def _resolve_reembed_source_version(self) -> str:
        """Return the materialization version `reembed_*` reads source rows
        from, raising if the operation is impossible / a no-op.

        Source is the currently-active materialization: the operator wants
        to re-encode the rows that production is serving today. Destination
        is `self.materialization_version` (the build path's own
        namespace). The two MUST differ — if they match, the destination
        rows are exactly what the source rows already are, so there's
        nothing to re-encode.

        Unlike `_resolve_read_active`, this method does NOT compare
        `embed_model_version`s: reembed is precisely the operation that
        crosses the embed-model boundary, so a mismatch is the expected
        case, not an error.
        """
        active = read_active_materialization(self._rag_root)
        if active is None:
            raise RuntimeError(
                "no active materialization to re-embed from. Build the "
                "initial materialization with `embed()` (or migrate the "
                "legacy layout via "
                "`scripts/migrate_materialization_to_v0.py`) and promote "
                "it before re-embedding."
            )
        if active.version == self.materialization_version:
            raise RuntimeError(
                "active materialization version "
                f"{active.version!r} already matches this adapter's "
                f"materialization_version {self.materialization_version!r}. "
                "Re-embed produces no new namespace — the build path "
                "already writes here. Bump one of CHUNKER_VERSION, "
                "CONTEXTUAL_CHUNK_PROMPT_VERSION, or EMBED_MODEL_VERSION_TAG "
                "before re-embedding."
            )
        return active.version

    def _emit_reembed_precondition_error(
        self, exc: RuntimeError, *, span_doc_id: str | None
    ) -> None:
        """Emit a one-shot `extract.reembed` error span for failures raised
        BEFORE `_reembed_table`'s own span context begins.

        Without this, the "no active materialization" / "already at this
        materialization" guards in `_resolve_reembed_source_version`
        surface as Python exceptions but leave no trace in OTel — error
        dashboards bucketing reembed attempts would silently undercount
        precondition failures. Same span name as the normal path so
        consumers don't need a second filter.
        """
        with _tracer.start_as_current_span("extract.reembed") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.backend", self._backend)
            span.set_attribute("embedding.model", self._model)
            span.set_attribute(
                "embedding.materialization_version", self.materialization_version
            )
            if span_doc_id is not None:
                span.set_attribute("extract.doc_id", span_doc_id)
            span.set_attribute("extract.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))

    def reembed_doc(self, doc_id: str) -> int:
        """Re-encode rows of `<doc_id>` from the active materialization into
        this adapter's own namespace; return the row count.

        Encoder-only path: the `text` column (already the contextual prefix
        + chunk text concatenation from the original `embed()` call) is
        re-encoded, no Anthropic / contextual-chunking calls are made,
        upstream chunker output is not touched. This is the cost-saving
        primitive for embed-model swaps at backfill scope.

        Source table: `{doc_id}__{active.version}` (the currently-active
        materialization). Destination table:
        `{doc_id}__{self.materialization_version}` (this adapter's own
        namespace). The destination is created if missing; existing
        contents under the destination version are overwritten.

        Also propagates the new vectors into the per-corpus narrative
        store of the destination materialization (if a corpus already
        exists there from a prior per-doc reembed in this sweep) via
        vector-copy — no second encoder pass.

        Raises `RuntimeError` if no active materialization exists, if the
        active version already matches this adapter's (the reembed would
        be a no-op), or if the source vectors' dim does not match the
        current adapter's `_vector_dim` (cross-dim swaps imply a new
        vector space and must build a fresh corpus, not append).
        """
        try:
            source_version = self._resolve_reembed_source_version()
        except RuntimeError as exc:
            self._emit_reembed_precondition_error(exc, span_doc_id=doc_id)
            raise
        source_table = versioned_table_name(doc_id, source_version)
        dest_table = self._write_table_name(doc_id)
        n, new_rows = self._reembed_table(
            source_table=source_table,
            dest_table=dest_table,
            span_doc_id=doc_id,
        )
        self._propagate_to_corpus(new_rows)
        return n

    def reembed_corpus(self) -> int:
        """Re-encode the per-corpus narrative table from the active
        materialization into this adapter's own namespace; return count.

        Same encoder-only semantics as `reembed_doc`. The corpus narrative
        store aggregates rows from many docs; re-encoding it on its own
        is correct because every row's `text` column already contains the
        embedding-ready string written by the original `embed()` call.

        Returns 0 (no-op) when the source corpus table does not exist —
        an active materialization populated solely from non-narrative
        doc_types (S-1, S-3, 8-K, DEF 14A) has no `_corpus_narrative__*`
        table, and `reembed --all` over such a corpus must succeed
        silently rather than crash with `FileNotFoundError`.
        """
        try:
            source_version = self._resolve_reembed_source_version()
        except RuntimeError as exc:
            self._emit_reembed_precondition_error(exc, span_doc_id=None)
            raise
        source_table = versioned_table_name(_PER_CORPUS_STORE, source_version)
        dest_table = self._write_table_name(_PER_CORPUS_STORE)
        db = lancedb.connect(self._rag_root)
        if source_table not in db.table_names():
            return 0
        n, _ = self._reembed_table(
            source_table=source_table,
            dest_table=dest_table,
            span_doc_id=None,
        )
        return n

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
            # Stamp the adapter's materialization_version up front so the
            # attribute is present even when `_read_table_name` raises on
            # an active-pointer mismatch — dashboards bucketing errors by
            # version still have a key to group on.
            span.set_attribute(
                "embedding.materialization_version", self.materialization_version
            )
            if doc_id is not None:
                span.set_attribute("extract.doc_id", doc_id)
            try:
                db = lancedb.connect(self._rag_root)
                # doc_id is non-None for per_doc (guarded above); narrow for mypy.
                base = (
                    doc_id
                    if (store == "per_doc" and doc_id is not None)
                    else _PER_CORPUS_STORE
                )
                table_name = self._read_table_name(base)
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
            # Stamp the adapter's materialization_version up front so error
            # spans from `_read_table_name`'s mismatch guard still carry the
            # routing key dashboards filter by.
            span.set_attribute(
                "embedding.materialization_version", self.materialization_version
            )
            if doc_id is not None:
                span.set_attribute("extract.doc_id", doc_id)
            try:
                db = lancedb.connect(self._rag_root)
                # doc_id is non-None for per_doc (guarded above); narrow for mypy.
                base = (
                    doc_id
                    if (store == "per_doc" and doc_id is not None)
                    else _PER_CORPUS_STORE
                )
                table_name = self._read_table_name(base)
                tbl = db.open_table(table_name)
                # Voyage and Qwen3-MLX both consume input_type as the
                # asymmetric-encoder switch on the query side (Voyage
                # via its `input_type` kwarg, Qwen3 via the model-card
                # instruction prefix). BGE ignores input_type.
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
