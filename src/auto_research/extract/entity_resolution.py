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
     candidate tickers.

  3. The LLM disambiguator receives the mention text + the top-k candidate
     (ticker, primary_name, sector) tuples and returns one ticker or null
     ("unknown"). Returning null is a first-class result, not a failure —
     a false-confident pick corrupts downstream signal A1 data, so the
     prompt explicitly forbids picking a candidate without positive
     evidence from the mention text.

The resolver intentionally does NOT carry citation-grounding (INV-2). That
contract is enforced by the upstream mention-extraction worker
(`SupplierMention.citation`); entity resolution operates on the already-
extracted mention text and emits an `EntityResolution` decision that the
caller writes back to `SupplierMention.resolver_*` fields.

A defensive check guards against the LLM returning a ticker that wasn't in
the candidate list — that would be a prompt-following bug, and the
resolver downgrades it to `unknown` with the bug noted in `reasoning`
rather than silently propagating an unverifiable mapping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import anthropic
import numpy as np
from numpy.typing import NDArray
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
_DEFAULT_USD_CAP = 2.0
_MAX_TOKENS = 512

_FROZEN_STRICT = ConfigDict(frozen=True, extra="forbid")
_tracer = trace.get_tracer(__name__)


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
    reasoning: str
    considered: tuple[CandidateTicker, ...]
    prompt_version: str
    embed_model_version: str


class _DisambiguatorResponse(BaseModel):
    """Internal shape used to parse the LLM's JSON output."""

    model_config = _FROZEN_STRICT

    ticker: str | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str


@dataclass(frozen=True)
class _IndexEntry:
    """One row in the universe alias index."""

    ticker: str
    alias_text: str
    primary_name: str
    sector: str


# Markdown-fence strip mirrors the s_filings worker; the prompt forbids
# fences but defensive cleanup keeps a single stray ``` from quarantining
# an otherwise-good response.
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


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
        usd_cap: float = _DEFAULT_USD_CAP,
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
        # `_encode` is the package-internal primitive both `embed` and
        # `query` delegate to; we use it directly here because the entity
        # index is in-memory (numpy) rather than LanceDB-backed.
        self._matrix: NDArray[np.float32] = self._adapter._encode(
            [e.alias_text for e in self._index_entries], input_type="document"
        )
        self._client: ExtractionFn = make_extraction_client(
            worker=_WORKER, usd_cap=usd_cap, anthropic_client=anthropic_client
        )

    @property
    def universe_size(self) -> int:
        """Distinct tickers present in the index (entries with aliases)."""
        return len({e.ticker for e in self._index_entries})

    def resolve(self, mention_text: str) -> EntityResolution:
        """Resolve one mention to a ticker (or `unknown`).

        Always returns an `EntityResolution`; failure modes (empty mention,
        malformed disambiguator output, off-list ticker) collapse to
        `resolved_ticker=None` with the reason captured in `reasoning`.
        """
        embed_version = self._adapter.embed_model_version
        with _tracer.start_as_current_span("extract.entity_resolve") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("embedding.backend", self._adapter.backend)
            span.set_attribute("embedding.model", self._adapter.model)

            stripped = mention_text.strip()
            if not stripped:
                span.set_attribute("extract.outcome", "empty_mention")
                return EntityResolution(
                    mention_text=mention_text,
                    resolved_ticker=None,
                    confidence=None,
                    reasoning="mention text was empty after stripping whitespace",
                    considered=(),
                    prompt_version=ENTITY_RESOLUTION_PROMPT_VERSION,
                    embed_model_version=embed_version,
                )

            candidates = self._top_candidates(stripped)
            span.set_attribute("entity.candidate_count", len(candidates))

            response = self._client(
                task=_TASK,
                system_prompt=ENTITY_RESOLUTION_PROMPT,
                user_content=_build_user_content(stripped, candidates),
                max_tokens=_MAX_TOKENS,
            )
            text = _strip_fence(
                "".join(b.text for b in response.content if b.type == "text").strip()
            )
            parsed = self._parse_disambiguator(text)
            if parsed is None:
                span.set_attribute("extract.outcome", "malformed_disambiguator")
                span.set_status(
                    Status(StatusCode.ERROR, "disambiguator returned malformed JSON")
                )
                return EntityResolution(
                    mention_text=mention_text,
                    resolved_ticker=None,
                    confidence=None,
                    reasoning=(
                        "disambiguator response could not be parsed as JSON "
                        f"matching the expected schema: {text!r}"
                    ),
                    considered=candidates,
                    prompt_version=ENTITY_RESOLUTION_PROMPT_VERSION,
                    embed_model_version=embed_version,
                )

            candidate_tickers = {c.ticker for c in candidates}
            picked = parsed.ticker
            if picked is not None and picked not in candidate_tickers:
                # The model picked a ticker outside the candidate list. That
                # violates the prompt; refuse to propagate. Surfaces as
                # `unknown` with the bug noted in `reasoning` so the
                # disambiguator's drift is auditable.
                span.set_attribute("extract.outcome", "off_list_ticker")
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        _truncate(f"disambiguator picked off-list ticker {picked!r}"),
                    )
                )
                return EntityResolution(
                    mention_text=mention_text,
                    resolved_ticker=None,
                    confidence=None,
                    reasoning=(
                        f"disambiguator returned ticker={picked!r} which was "
                        "not in the candidate list; downgraded to unknown. "
                        f"Disambiguator reasoning: {parsed.reasoning}"
                    ),
                    considered=candidates,
                    prompt_version=ENTITY_RESOLUTION_PROMPT_VERSION,
                    embed_model_version=embed_version,
                )

            # confidence must be None when ticker is None — enforce here so
            # bad pairings don't leak into the audit trail.
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
                embed_model_version=embed_version,
            )

    def _top_candidates(self, mention_text: str) -> tuple[CandidateTicker, ...]:
        """Top-k unique tickers ranked by max cosine over their aliases."""
        qvec = self._adapter._encode([mention_text], input_type="query")[0]
        # All backends emit L2-normalized vectors, so dot product = cosine.
        scores = self._matrix @ qvec
        best_by_ticker: dict[str, tuple[float, _IndexEntry]] = {}
        for entry, score in zip(self._index_entries, scores, strict=True):
            score_f = float(score)
            current = best_by_ticker.get(entry.ticker)
            if current is None or score_f > current[0]:
                best_by_ticker[entry.ticker] = (score_f, entry)
        ranked = sorted(
            best_by_ticker.items(),
            key=lambda kv: kv[1][0],
            reverse=True,
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
    """
    out: list[_IndexEntry] = []
    for entry in entries:
        if not entry.aliases:
            continue
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
