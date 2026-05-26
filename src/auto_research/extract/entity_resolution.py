"""Entity resolution — fuzzy supplier/customer mention -> tradeable ticker.

Flow 3 of `docs/specs/2026-05-22-design.md` §8.3:

  1. Universe entries with aliases are embedded once with the adapter's
     backend (Voyage `voyage-finance-2` in production; BGE in tests).
     The index is a numpy matrix of L2-normalized vectors plus a sidecar
     list of (ticker, alias_text, primary_name, sector). LanceDB is not
     used — the index is ~250 rows total; an in-memory dot product
     handles top-k in microseconds, and the bookkeeping for a versioned
     table at this scale would dwarf the value.

  2. A mention text is embedded with the same adapter (`input_type="query"`
     on the asymmetric backends), cosine-scored against the index, and
     deduped by ticker (max score per ticker) to produce the top-k
     candidate tickers. Ties break by ticker symbol so candidate ordering
     is stable across universe re-sorts.

  3. The LLM disambiguator receives the mention text + the top-k candidate
     (ticker, primary_name, sector) tuples and returns one ticker or null
     ("unknown"). Returning null is a first-class result, not a failure —
     a false-confident pick corrupts downstream signal A1 data, so the
     prompt explicitly forbids picking a candidate without positive
     evidence from the mention text.

The resolver intentionally does NOT carry citation-grounding (INV-2). That
contract is enforced by the upstream mention-extraction worker
(`SupplierMention.citation`); entity resolution operates on the already-
extracted mention text and emits an `EntityResolution` decision. Because
`SupplierMention` is frozen, integrators write back the resolver fields
via `mention.model_copy(update={"resolved_ticker": ..., "resolver_confidence":
..., "resolver_reasoning": ...})` rather than direct attribute assignment.

The resolver hardens the LLM contract at the parser boundary: the
disambiguator response schema requires `confidence > 0` paired with any
non-null ticker, and `reasoning` must be non-empty. Several
LLM-misbehavior modes (off-list ticker, stringified `"null"`, truncated
response, case/whitespace drift) collapse to a structured `unknown` with
the cause captured in `reasoning` rather than silently propagating an
unverifiable mapping.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import anthropic
import numpy as np
from numpy.typing import NDArray
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.embeddings import EmbeddingAdapter
from auto_research.extract.prompts.entity_resolution import (
    ENTITY_RESOLUTION_PROMPT,
    ENTITY_RESOLUTION_PROMPT_VERSION,
)
from auto_research.telemetry import truncate_status_description as _truncate
from auto_research.universe import TickerEntry, load_universe

_WORKER = "entity_resolution"
_TASK = "disambiguate_mention"
_DEFAULT_TOP_K = 3
# Worker-wide USD cap. Matches the s_filings worker's default tier. The
# resolver issues short Haiku calls (~500 in + ~150 out tokens each), so
# 5.0 covers ~5000 mentions per process before the cap trips.
_DEFAULT_USD_CAP = 5.0
# Headroom over the prompt's "1-2 sentence reasoning" request. 1024 keeps
# us well clear of truncation on Haiku's typical ~120-token responses
# without paying for headroom we'll never use.
_MAX_TOKENS = 1024
# Cosine-via-dot-product requires L2-normalized vectors on both sides.
# All three backends (Voyage / BGE / Qwen3-MLX) emit normalized vectors
# today, but we re-check at construction so a future contract change
# (vendor opaque re-upload, normalization-policy flip) surfaces loudly
# instead of silently corrupting top-k ranking.
_NORM_TOLERANCE = 1e-3

_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid")
_tracer = trace.get_tracer(__name__)

# Module-level singleton client. Mirrors the `_CLIENT` pattern in
# `extract/workers/s_filings.py` — per-worker `@cost_cap` and
# `@circuit_breaker` state accumulates across calls inside one process
# rather than splintering across `EntityResolver` instances. Tests that
# pass `anthropic_client=...` get a fresh per-call client and bypass the
# singleton (test isolation).
_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    """Return the production singleton, or a fresh client for test injection."""
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(
            worker=_WORKER,
            usd_cap=_DEFAULT_USD_CAP,
            anthropic_client=anthropic_client,
        )
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER, usd_cap=_DEFAULT_USD_CAP)
    return _CLIENT


class CandidateTicker(BaseModel):
    """One top-k candidate surfaced by dense retrieval."""

    model_config = _FROZEN_STRICT

    ticker: str
    primary_name: str
    sector: str
    score: float = Field(ge=-1.0, le=1.0)


class EntityResolution(BaseModel):
    """Result of resolving one mention to a ticker (or `unknown`).

    `resolved_ticker is None` is a first-class outcome — the caller must
    treat None as "do not record a ticker mapping for this mention"
    rather than retrying with a degraded threshold. `considered` carries
    the candidate slate the disambiguator saw, so a reviewer can replay
    the decision without re-running embeddings.
    """

    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    mention_text: str
    resolved_ticker: str | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    considered: tuple[CandidateTicker, ...]
    prompt_version: str
    embed_model_version: str


class _DisambiguatorResponse(BaseModel):
    """Internal shape used to parse the LLM's JSON output.

    `reasoning` is required and non-empty: the issue's acceptance criterion
    ("disambiguator stores reasoning per resolution for audit") would be
    silently violated by an LLM that emitted an empty string. A non-null
    ticker must be paired with `confidence > 0` — a self-contradictory
    `confidence == 0` paired with a real ticker would multiply to zero in
    downstream weighted-mention code and silently drop the mapping. Both
    violations raise `ValidationError`, which the resolver catches and
    reports as `malformed_disambiguator`.
    """

    model_config = _FROZEN_STRICT

    ticker: str | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ticker_confidence_pair(self) -> _DisambiguatorResponse:
        if self.ticker is not None and (
            self.confidence is None or self.confidence <= 0.0
        ):
            raise ValueError(
                f"non-null ticker {self.ticker!r} requires confidence > 0; "
                f"got {self.confidence!r}"
            )
        return self


@dataclass(frozen=True)
class _IndexEntry:
    """One row in the universe alias index."""

    ticker: str
    alias_text: str
    primary_name: str
    sector: str


# Find the JSON object inside the response. Prefer the simple case (the
# whole response is `{...}`); fall back to slicing from first `{` to last
# `}` so a stray prelude or fence doesn't quarantine an otherwise-good
# payload. Regex-anchored fence stripping was too strict — partial fences
# like `"Here is the JSON:\n```{...}```"` slipped through.
def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return stripped  # let json.loads fail naturally
    return stripped[start : end + 1]


# Patterns the LLM occasionally uses for "null" instead of JSON null —
# normalize these to None so the off-list guard doesn't misdiagnose a
# JSON-stringification bug as a prompt-following bug.
_NULL_LIKE_TICKERS = frozenset({"null", "none", "n/a", "na", "unknown", ""})


def _normalize_picked_ticker(
    raw: str | None, candidates: frozenset[str]
) -> tuple[str | None, str | None]:
    """Canonicalize the LLM's picked ticker against the candidate set.

    Returns `(picked, off_list_reason)`. `picked` is the canonical ticker
    when matched, None otherwise. `off_list_reason` is None on match or
    on a clean null/null-like input, and a human-readable explanation
    when the LLM picked a ticker not in the candidate slate.
    """
    if raw is None:
        return None, None
    cleaned = raw.strip()
    if cleaned.lower() in _NULL_LIKE_TICKERS:
        return None, None
    # Case-insensitive lookup against the candidate slate; return the
    # canonical-cased ticker from the universe rather than the LLM's
    # casing so the recorded EntityResolution matches what Feast expects.
    by_upper = {c.upper(): c for c in candidates}
    canonical = by_upper.get(cleaned.upper())
    if canonical is not None:
        return canonical, None
    return None, f"disambiguator returned ticker={raw!r} which is not in the candidate list"


def _format_candidates(candidates: Sequence[CandidateTicker]) -> str:
    lines = [
        f"- {c.ticker}: {c.primary_name} ({c.sector})" for c in candidates
    ]
    return "\n".join(lines)


def _build_user_content(
    mention_text: str, candidates: Sequence[CandidateTicker]
) -> str:
    return (
        f"Mention: {mention_text}\n\n"
        f"Candidates:\n{_format_candidates(candidates)}"
    )


class EntityResolver:
    """Embeds universe aliases once, then resolves mentions on demand."""

    def __init__(
        self,
        *,
        adapter: EmbeddingAdapter,
        universe: Iterable[TickerEntry] | None = None,
        anthropic_client: anthropic.Anthropic | None = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {top_k}")
        loaded = (
            tuple(universe) if universe is not None else load_universe()
        )
        index_entries = _flatten_aliases(loaded)
        if not index_entries:
            raise ValueError(
                "universe has no aliases-bearing entries; entity resolution "
                "needs at least one TickerEntry with non-empty `aliases`."
            )
        self._adapter = adapter
        self._top_k = top_k
        self._index_entries: tuple[_IndexEntry, ...] = tuple(index_entries)
        # Build the index under its own OTel span so a Voyage 429-retry
        # storm during warmup shows up in the same trace surface as
        # query-time embeddings, not as an unparented startup log.
        with _tracer.start_as_current_span("extract.entity_resolution_init") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.backend", adapter.backend)
            span.set_attribute("embedding.model", adapter.model)
            span.set_attribute("entity.index_rows", len(self._index_entries))
            # `_encode` is the package-internal primitive both `embed` and
            # `query` delegate to; the in-memory index doesn't need the
            # LanceDB write path that `embed()` adds, so we call it
            # directly. Voyage rate-limit retry is inside `_encode`, so
            # we inherit that behavior here too.
            self._matrix: NDArray[np.float32] = self._adapter._encode(
                [e.alias_text for e in self._index_entries],
                input_type="document",
            )
        # Capture the embed-model version once at construction so a
        # mid-run module-level EMBED_MODEL_VERSION_TAG bump can't drift
        # the recorded version away from the contract the matrix vectors
        # were actually produced under.
        self._embed_version: str = adapter.embed_model_version
        # Re-check the cosine invariant at construction. If a future
        # backend stops normalizing, `self._matrix @ qvec` would silently
        # become magnitude-weighted top-k; this assert turns the silent
        # corruption into a loud `RuntimeError` at init time.
        norms = np.linalg.norm(self._matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            raise RuntimeError(
                f"Embedding backend {adapter.backend!r}/{adapter.model!r} "
                f"returned non-L2-normalized vectors (norm range "
                f"[{norms.min():.4f}, {norms.max():.4f}]); resolver relies "
                "on cosine == dot product."
            )
        self._client: ExtractionFn = _get_client(anthropic_client)

    @property
    def universe_size(self) -> int:
        """Distinct tickers present in the index (entries with aliases)."""
        return len({e.ticker for e in self._index_entries})

    def resolve(self, mention_text: str) -> EntityResolution:
        """Resolve one mention to a ticker (or `unknown`).

        Always returns an `EntityResolution`; failure modes (empty mention,
        truncated / malformed disambiguator output, off-list ticker)
        collapse to `resolved_ticker=None` with the cause captured in
        `reasoning`.
        """
        with _tracer.start_as_current_span("extract.entity_resolve") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.backend", self._adapter.backend)
            span.set_attribute("embedding.model", self._adapter.model)

            stripped = mention_text.strip()
            if not stripped:
                span.set_attribute("extract.outcome", "empty_mention")
                return self._build_unknown(
                    mention_text=mention_text,
                    reasoning="mention text was empty after stripping whitespace",
                    considered=(),
                )

            candidates = self._top_candidates(stripped)
            span.set_attribute("entity.candidate_count", len(candidates))

            response = self._client(
                task=_TASK,
                system_prompt=ENTITY_RESOLUTION_PROMPT,
                user_content=_build_user_content(stripped, candidates),
                max_tokens=_MAX_TOKENS,
            )

            # Detect truncation BEFORE attempting to parse — a partial JSON
            # is indistinguishable from a malformed one once we hit
            # JSONDecodeError, and conflating the two misdirects operators.
            if response.stop_reason == "max_tokens":
                span.set_attribute("extract.outcome", "truncated_disambiguator")
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        _truncate(
                            f"disambiguator response truncated at max_tokens={_MAX_TOKENS}"
                        ),
                    )
                )
                return self._build_unknown(
                    mention_text=mention_text,
                    reasoning=(
                        f"disambiguator response was truncated at "
                        f"max_tokens={_MAX_TOKENS} (stop_reason='max_tokens'); "
                        "the JSON object was cut mid-stream and is unparseable"
                    ),
                    considered=candidates,
                )

            # Drop non-text blocks (refusals, tool-use). An all-tool-use
            # response collapses to "" and falls through to the malformed
            # branch below with an explicit "no text" note in reasoning.
            text_blocks = [b.text for b in response.content if b.type == "text"]
            if not text_blocks:
                span.set_attribute("extract.outcome", "no_text_block")
                span.set_status(
                    Status(StatusCode.ERROR, "disambiguator returned no text block")
                )
                return self._build_unknown(
                    mention_text=mention_text,
                    reasoning=(
                        "disambiguator response contained no text block "
                        "(likely a refusal or tool-use-only response)"
                    ),
                    considered=candidates,
                )

            text = _strip_fence("".join(text_blocks).strip())
            parsed = self._parse_disambiguator(text)
            if parsed is None:
                span.set_attribute("extract.outcome", "malformed_disambiguator")
                span.set_status(
                    Status(StatusCode.ERROR, "disambiguator returned malformed JSON")
                )
                return self._build_unknown(
                    mention_text=mention_text,
                    reasoning=(
                        "disambiguator response could not be parsed as JSON "
                        f"matching the expected schema: {text!r}"
                    ),
                    considered=candidates,
                )

            candidate_tickers = frozenset(c.ticker for c in candidates)
            picked, off_list_reason = _normalize_picked_ticker(
                parsed.ticker, candidate_tickers
            )
            if off_list_reason is not None:
                # The model picked a ticker outside the candidate list
                # (and it wasn't a null-like sentinel either). Refuse to
                # propagate. Surfaces as `unknown` with the bug noted in
                # `reasoning` so the disambiguator's drift is auditable.
                span.set_attribute("extract.outcome", "off_list_ticker")
                span.set_status(
                    Status(StatusCode.ERROR, _truncate(off_list_reason))
                )
                return self._build_unknown(
                    mention_text=mention_text,
                    reasoning=(
                        f"{off_list_reason}; downgraded to unknown. "
                        f"Disambiguator reasoning: {parsed.reasoning}"
                    ),
                    considered=candidates,
                )

            # confidence must be None when ticker is None — enforce here so
            # bad pairings don't leak into the audit trail. (Pydantic's
            # post-validator already rejects ticker-with-zero-confidence;
            # this handles the symmetric null-ticker-with-leftover-
            # confidence case where the LLM contradicts itself.)
            confidence = parsed.confidence if picked is not None else None
            span.set_attribute(
                "extract.outcome",
                "resolved" if picked is not None else "unknown",
            )
            if picked is not None:
                span.set_attribute("entity.resolved_ticker", picked)
            return EntityResolution(
                mention_text=mention_text,
                resolved_ticker=picked,
                confidence=confidence,
                reasoning=parsed.reasoning,
                considered=candidates,
                prompt_version=ENTITY_RESOLUTION_PROMPT_VERSION,
                embed_model_version=self._embed_version,
            )

    def _build_unknown(
        self,
        *,
        mention_text: str,
        reasoning: str,
        considered: tuple[CandidateTicker, ...],
    ) -> EntityResolution:
        """Construct an `unknown` EntityResolution with the captured version tags."""
        return EntityResolution(
            mention_text=mention_text,
            resolved_ticker=None,
            confidence=None,
            reasoning=reasoning,
            considered=considered,
            prompt_version=ENTITY_RESOLUTION_PROMPT_VERSION,
            embed_model_version=self._embed_version,
        )

    def _top_candidates(self, mention_text: str) -> tuple[CandidateTicker, ...]:
        """Top-k unique tickers ranked by max cosine over their aliases.

        Tie-broken by ticker symbol (ascending) so the candidate slate is
        deterministic across universe re-sorts — alphabetizing the
        universe JSON, for instance, would otherwise quietly reorder the
        top-k on float ties because dict insertion order is the implicit
        tiebreaker.
        """
        qvec = self._adapter._encode([mention_text], input_type="query")[0]
        scores = self._matrix @ qvec
        best_by_ticker: dict[str, tuple[float, _IndexEntry]] = {}
        for entry, score in zip(self._index_entries, scores, strict=True):
            score_f = float(score)
            current = best_by_ticker.get(entry.ticker)
            if current is None or score_f > current[0]:
                best_by_ticker[entry.ticker] = (score_f, entry)
        ranked = sorted(
            best_by_ticker.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )[: self._top_k]
        return tuple(
            CandidateTicker(
                ticker=ticker,
                primary_name=entry.primary_name,
                sector=entry.sector,
                score=score,
            )
            for ticker, (score, entry) in ranked
        )

    @staticmethod
    def _parse_disambiguator(text: str) -> _DisambiguatorResponse | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        try:
            return _DisambiguatorResponse.model_validate(payload)
        except ValidationError:
            return None


def _flatten_aliases(entries: Iterable[TickerEntry]) -> list[_IndexEntry]:
    """Expand each (ticker, alias) pair into one index row.

    The first alias is treated as the canonical primary_name — used in the
    candidate list shown to the disambiguator. Tickers with no aliases are
    skipped (they can't be matched from narrative mentions anyway).

    Empty / whitespace-only alias strings raise `ValueError`: embedding an
    empty string yields a degenerate vector that magnets unrelated
    mentions into the top-k. A loud failure at construction makes the
    universe-edit typo visible immediately.
    """
    out: list[_IndexEntry] = []
    for entry in entries:
        if not entry.aliases:
            continue
        for alias in entry.aliases:
            if not alias.strip():
                raise ValueError(
                    f"ticker {entry.ticker!r} has an empty / whitespace-only "
                    f"alias {alias!r}; either remove it from the universe JSON "
                    "or replace it with a real surface form."
                )
        primary = entry.aliases[0]
        for alias in entry.aliases:
            out.append(
                _IndexEntry(
                    ticker=entry.ticker,
                    alias_text=alias,
                    primary_name=primary,
                    sector=entry.sector,
                )
            )
    return out


__all__ = [
    "CandidateTicker",
    "EntityResolution",
    "EntityResolver",
]
