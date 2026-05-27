"""10-K extraction worker.

Three paths, one entry point:

1. **Narrative single-shot.** `count_tokens(raw_doc) <
   SINGLE_SHOT_TOKEN_CUTOFF` and no chunkset supplied → one Anthropic
   call against the full raw doc via the shared
   `run_single_shot_extraction` driver.
2. **Narrative RAG.** `count_tokens(raw_doc) ≥ SINGLE_SHOT_TOKEN_CUTOFF`
   AND a `ChunkSet` is supplied → one Anthropic call PER narrative
   field, each scoped to the top reranked parents returned by the
   injected `retrieve_fn`. The five fields (guidance_tone,
   accrual_flags, supplier_mentions, customer_mentions,
   risk_factor_deltas) each get a field-specific query string and a
   distinct `doc_id` cache key.
3. **Item 8 financials.** Independent of the narrative path: when a
   chunkset is supplied AND has at least one parent with
   `table_html is not None`, the first table parent is fed to the
   `ten_k_financials` prompt + `TenKFinancials` schema. The result is
   merged onto the narrative output via `model_copy`.

`retrieve_fn` is injected so this module stays orthogonal to the RAG
stack — the backfill orchestrator owns wiring it to the real
`hybrid_retrieve + rerank` composition. Unit tests pass a deterministic
stub.

Production callers omit `cache_root` / `quarantine_root` (package
defaults) and rely on `_CLIENT`'s singleton state across docs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.chunking import (
    SINGLE_SHOT_TOKEN_CUTOFF,
    ChunkSet,
    ParentChunk,
    count_tokens,
)
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.ten_k_financials import (
    TEN_K_FINANCIALS_PROMPT,
    TEN_K_FINANCIALS_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT,
    TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.schemas import TenKFinancials, TenKOutput
from auto_research.extract.workers._common import run_single_shot_extraction

_WORKER = "ten_k"
# Narrative single-shot is dominated by the cross-doc supplier/customer
# fields per spec §7.3 — route through the highest-tier field key so
# the unified call gets Sonnet rather than Haiku.
_NARRATIVE_DEFAULT_TASK = "supplier_mentions"
_FINANCIALS_TASK = "financials"
_NARRATIVE_MAX_TOKENS = 8192
_FINANCIALS_MAX_TOKENS = 4096

_CLIENT: ExtractionFn | None = None

# Per-field RAG queries. Tuned at writing time; expected to evolve under
# eval (#20) — the prompt itself is frozen at v1, the query strings are
# orchestration code.
_NARRATIVE_RAG_QUERIES: dict[str, str] = {
    "guidance_tone": (
        "What is management's tone on forward growth, gross margin, and "
        "demand in the MD&A section?"
    ),
    "accrual_flags": (
        "What are the accrual-quality concerns: unbilled receivables, "
        "deferred revenue swings, capitalized R&D, restructuring resets?"
    ),
    "supplier_mentions": (
        "Which specific named suppliers (e.g., TSMC, Foxconn, Samsung, "
        "ASML) does the company rely on?"
    ),
    "customer_mentions": (
        "Which specific named customers — hyperscalers, large enterprises "
        "— are explicitly called out by name?"
    ),
    "risk_factor_deltas": (
        "What new, removed, or modified Item 1A risk factors does this "
        "filing disclose?"
    ),
}

RetrieveFn = Callable[[str], list[ParentChunk]]


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    """Return the production singleton, or a fresh client for test injection."""
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(
            worker=_WORKER, anthropic_client=anthropic_client
        )
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER)
    return _CLIENT


def _format_parents_as_context(parents: list[ParentChunk]) -> str:
    """Concatenate parent text with section-name headers.

    The LLM sees a single user-content block with the retrieved
    passages. Spans returned by the LLM will be resolved against this
    assembled string in the worker (not against the raw 10-K), so
    INV-2's `source_text[span] == source_quote` holds against the text
    the model actually saw.
    """
    return "\n\n".join(f"[{p.section_name}]\n{p.text}" for p in parents)


def _extract_item8_financials(
    *,
    parent_table_html: str,
    doc_id: str,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None,
) -> TenKFinancials | None:
    """Run the financials prompt against `parent_table_html`.

    Item 8's raw_doc is the table HTML itself, so the per-row
    `source_quote` resolution and cache key naturally key off the table
    contents alone. Different table HTML → different cache key, even
    if the surrounding 10-K is identical.
    """
    return run_single_shot_extraction(
        raw_doc=parent_table_html,
        doc_id=f"{doc_id}#item8",  # distinct cache key from narrative
        worker=_WORKER,
        task=_FINANCIALS_TASK,
        prompt=TEN_K_FINANCIALS_PROMPT,
        prompt_version=TEN_K_FINANCIALS_PROMPT_VERSION,
        output_model=TenKFinancials,
        max_tokens=_FINANCIALS_MAX_TOKENS,
        cache_root=cache_root,
        quarantine_root=quarantine_root,
        anthropic_client=anthropic_client,
        client_factory=_get_client,
    )


def _extract_ten_k_rag(
    *,
    raw_doc: str,
    doc_id: str,
    chunkset: ChunkSet,
    retrieve_fn: RetrieveFn,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None,
) -> TenKOutput | None:
    """Per-field RAG narrative extraction.

    For each TenKOutput narrative field, retrieve the top parents via
    `retrieve_fn`, format them as user content, and run the narrative
    prompt. The prompt asks for the full narrative output; the worker
    takes only the relevant field from each call and merges into one
    TenKOutput. Identity fields (cik, accession_number,
    fiscal_period_end, language_novelty_score) come from the first
    successful call.

    Per-field LLM calls trade higher cost for per-field reranker
    selectivity, which is the spec §8.1 pattern ("for each Pydantic
    schema field, retrieve top-k child chunks → resolve to parents →
    extraction call from retrieved context"). The five-call cost is
    accepted because long 10-Ks are ~20% of the corpus and the per-field
    selectivity gain on accrual_flags + risk_factor_deltas is large.
    """
    partials: dict[str, Any] = {}
    for field, query in _NARRATIVE_RAG_QUERIES.items():
        parents = retrieve_fn(query)
        user_content = _format_parents_as_context(parents)
        per_field = run_single_shot_extraction(
            raw_doc=user_content,
            doc_id=f"{doc_id}#{field}",  # distinct cache key per field
            worker=_WORKER,
            task=field,
            prompt=TEN_K_NARRATIVE_PROMPT,
            prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
            output_model=TenKOutput,
            max_tokens=_NARRATIVE_MAX_TOKENS,
            cache_root=cache_root,
            quarantine_root=quarantine_root,
            anthropic_client=anthropic_client,
            client_factory=_get_client,
        )
        if per_field is None:
            # One field's quarantine drops the whole 10-K — narrative
            # output without a key field is misleading rather than
            # incomplete. Reviewer reads the per-field quarantine record.
            return None
        partials[field] = getattr(per_field, field)
        partials.setdefault("cik", per_field.cik)
        partials.setdefault("accession_number", per_field.accession_number)
        partials.setdefault("fiscal_period_end", per_field.fiscal_period_end)
        partials.setdefault(
            "language_novelty_score", per_field.language_novelty_score
        )

    return TenKOutput(**partials)


def extract_ten_k(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
    chunkset: ChunkSet | None = None,
    retrieve_fn: RetrieveFn | None = None,
) -> TenKOutput | None:
    """Extract a TenKOutput from a raw 10-K filing.

    Single-shot when `count_tokens(raw_doc) < SINGLE_SHOT_TOKEN_CUTOFF`
    and no chunkset is supplied; RAG-per-narrative-field otherwise.
    Item 8 financials read from `ParentChunk.table_html` when a
    chunkset is supplied with a table parent — independent of the
    narrative path's branch choice.

    Returns `None` when ANY of the LLM calls fails parse /
    span-resolution / grounding; the caller MUST treat `None` as
    "do not persist." Each failure path writes its own
    QuarantineRecord.
    """
    cache_root_resolved = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    quarantine_root_resolved = (
        quarantine_root
        if quarantine_root is not None
        else DEFAULT_QUARANTINE_ROOT
    )

    # 1. Narrative path: single-shot if short OR no chunkset, RAG otherwise.
    narrative_is_rag = (
        chunkset is not None
        and count_tokens(raw_doc) >= SINGLE_SHOT_TOKEN_CUTOFF
    )
    if narrative_is_rag:
        if retrieve_fn is None:
            raise ValueError(
                "RAG branch requires an explicit retrieve_fn; the backfill "
                "orchestrator owns wiring it to hybrid_retrieve + rerank."
            )
        assert chunkset is not None  # narrow for mypy
        narrative = _extract_ten_k_rag(
            raw_doc=raw_doc,
            doc_id=doc_id,
            chunkset=chunkset,
            retrieve_fn=retrieve_fn,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
        )
    else:
        narrative = run_single_shot_extraction(
            raw_doc=raw_doc,
            doc_id=doc_id,
            worker=_WORKER,
            task=_NARRATIVE_DEFAULT_TASK,
            prompt=TEN_K_NARRATIVE_PROMPT,
            prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
            output_model=TenKOutput,
            max_tokens=_NARRATIVE_MAX_TOKENS,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
            client_factory=_get_client,
        )
    if narrative is None:
        return None

    # 2. Item 8 financials: independent of narrative path. Runs only when
    # the caller supplied a chunkset with a table parent.
    if chunkset is None:
        return narrative
    table_parents = [p for p in chunkset.parents if p.table_html is not None]
    if not table_parents:
        return narrative
    table_html = table_parents[0].table_html
    assert table_html is not None  # narrow for mypy
    financials = _extract_item8_financials(
        parent_table_html=table_html,
        doc_id=doc_id,
        cache_root=cache_root_resolved,
        quarantine_root=quarantine_root_resolved,
        anthropic_client=anthropic_client,
    )
    return narrative.model_copy(update={"financials": financials})


__all__ = ["RetrieveFn", "extract_ten_k"]
