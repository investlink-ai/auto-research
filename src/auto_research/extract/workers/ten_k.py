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
defaults) and rely on `_common._CLIENTS["ten_k"]`'s singleton state
across docs.
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
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.ten_k_financials import (
    TEN_K_FINANCIALS_PROMPT,
    TEN_K_FINANCIALS_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT,
    TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative_field import (
    TEN_K_NARRATIVE_FIELD_CONFIGS,
    TEN_K_NARRATIVE_FIELD_PROMPT,
    TEN_K_NARRATIVE_FIELD_PROMPT_VERSION,
)
from auto_research.extract.schemas import TenKFinancials, TenKOutput
from auto_research.extract.workers._common import (
    check_identity_agreement,
    commit_staged_cache_writes,
    run_single_shot_extraction,
)

_WORKER = "ten_k"
# Narrative single-shot is dominated by the cross-doc supplier/customer
# fields per spec §7.3 — route through the highest-tier field key so
# the unified call gets Sonnet rather than Haiku.
_NARRATIVE_DEFAULT_TASK = "supplier_mentions"
_FINANCIALS_TASK = "financials"
_NARRATIVE_MAX_TOKENS = 8192
_FINANCIALS_MAX_TOKENS = 4096

RetrieveFn = Callable[[str], list[ParentChunk]]


def _format_parents_as_context(parents: list[ParentChunk]) -> str:
    """Concatenate parent text with section-name headers.

    The LLM sees a single user-content block with the retrieved
    passages. Spans returned by the LLM will be resolved against this
    assembled string in the worker (not against the raw 10-K), so
    INV-2's `source_text[span] == source_quote` holds against the text
    the model actually saw.
    """
    return "\n\n".join(f"[{p.section_name}]\n{p.text}" for p in parents)


def _render_table_html_to_text(html: str) -> str:
    """Render the outer table HTML to plain text with cells separated by
    spaces — so the LLM's source_quote (which is prompted to be label +
    value, e.g., 'Total revenue $1,234') can resolve against the
    rendered text via whitespace-flexible regex. Raw HTML would require
    the quote to bridge `</td><td>` literally, which it never does.
    """
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _extract_item8_financials(
    *,
    parent_table_html: str,
    doc_id: str,
    cache_index: int,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None,
) -> TenKFinancials | None:
    """Run the financials prompt against a single Item 8 table.

    `parent_table_html` is rendered to plain text before the LLM sees
    it — the prompt asks for cell-text quotes (e.g., 'Total revenue
    $1,234') that the whitespace-flex regex cannot resolve against raw
    HTML across `</td><td>` boundaries. `cache_index` disambiguates per
    table within one filing (income-statement / balance-sheet / cash-
    flow are separate parents) so each table has its own cache key and
    quarantine record.
    """
    rendered = _render_table_html_to_text(parent_table_html)
    return run_single_shot_extraction(
        raw_doc=rendered,
        doc_id=f"{doc_id}#item8.{cache_index}",
        worker=_WORKER,
        task=_FINANCIALS_TASK,
        prompt=TEN_K_FINANCIALS_PROMPT,
        prompt_version=TEN_K_FINANCIALS_PROMPT_VERSION,
        output_model=TenKFinancials,
        max_tokens=_FINANCIALS_MAX_TOKENS,
        cache_root=cache_root,
        quarantine_root=quarantine_root,
        anthropic_client=anthropic_client,
    )


def _merge_financials(parts: list[TenKFinancials]) -> TenKFinancials:
    """Merge per-table TenKFinancials by first-non-None per field.

    Each Item 8 line item exists in exactly one primary statement
    (revenue → income statement; total_assets → balance sheet; etc.),
    so 'first non-None wins' has no real ambiguity for the common
    income/balance/cash-flow case. `parts` MUST be ordered as the
    chunker emits them (document order); first non-None then favors
    primary statements over later notes-table sub-aggregations that
    might reuse a label.
    """
    field_names = list(TenKFinancials.model_fields.keys())
    merged: dict[str, object] = {}
    for field in field_names:
        for part in parts:
            value = getattr(part, field)
            if value is not None:
                merged[field] = value
                break
        else:
            merged[field] = None
    return TenKFinancials.model_validate(merged)


def _extract_ten_k_rag(
    *,
    doc_id: str,
    chunkset: ChunkSet,
    retrieve_fn: RetrieveFn,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None,
) -> TenKOutput | None:
    """Per-field RAG narrative extraction.

    For each narrative field, retrieve the top parents via
    `retrieve_fn`, format them as user content, and run a
    field-scoped prompt against a partial schema that carries exactly
    that field plus the identity columns. The worker assembles the
    five partials into a full `TenKOutput` at the end.

    Per-field calls stage their cache writes; the worker commits all
    staged writes only AFTER the full 5-field loop succeeds AND the
    cross-partial identity check passes. A mid-loop quarantine returns
    `None` without persisting ANY of the earlier fields' results — so
    re-runs see a consistent cache state rather than half-cached
    partial output.

    Each per-field call uses the model tier the routing table actually
    declares (3 of 5 are Haiku) — the unified pre-split call routed
    everything to Sonnet via `_NARRATIVE_DEFAULT_TASK = supplier_mentions`,
    paying for the wrong tier on guidance_tone / accrual_flags /
    risk_factor_deltas. The partial schemas eliminate the dual-output
    waste from emitting the full TenKOutput shape on every call and
    discarding all but one field downstream — the unified-call path
    paid ~5x output cost for this waste.

    `chunkset` is currently unused inside the loop — `retrieve_fn`
    abstracts the chunkset-to-parent pipeline behind the
    field-keyed query. Kept on the signature because the caller
    (`extract_ten_k`) already has it and a future identity-from-cover-
    page-chunk pass will need it directly.
    """
    del chunkset  # currently consumed by `retrieve_fn`; keep on signature.

    pending_writes: list[tuple[str, dict[str, Any]]] = []

    def _stage(cache_key: str, payload: dict[str, Any]) -> None:
        pending_writes.append((cache_key, payload))

    narrative_partials: dict[str, Any] = {}
    identity_seen: dict[str, list[Any]] = {
        "cik": [],
        "accession_number": [],
        "fiscal_period_end": [],
    }
    for config in TEN_K_NARRATIVE_FIELD_CONFIGS:
        parents = retrieve_fn(config.retrieval_query)
        user_content = _format_parents_as_context(parents)
        field_prompt = TEN_K_NARRATIVE_FIELD_PROMPT.format(
            field_name=config.field_name,
            field_description=config.description,
        )
        per_field = run_single_shot_extraction(
            raw_doc=user_content,
            doc_id=f"{doc_id}#{config.field_name}",
            worker=_WORKER,
            task=config.field_name,
            prompt=field_prompt,
            prompt_version=TEN_K_NARRATIVE_FIELD_PROMPT_VERSION,
            output_model=config.schema,
            max_tokens=_NARRATIVE_MAX_TOKENS,
            cache_root=cache_root,
            quarantine_root=quarantine_root,
            anthropic_client=anthropic_client,
            cache_write_handler=_stage,
        )
        if per_field is None:
            # One field's quarantine drops the whole 10-K — narrative
            # output without a key field is misleading rather than
            # incomplete. Reviewer reads the per-field quarantine
            # record. No staged writes are committed, so re-runs see
            # no stale per-field cache entries from this attempt.
            return None
        narrative_partials[config.field_name] = getattr(
            per_field, config.field_name
        )
        for identity_field in identity_seen:
            identity_seen[identity_field].append(
                getattr(per_field, identity_field)
            )

    agreed_identity = check_identity_agreement(
        identity_values=identity_seen,
        quarantine_root=quarantine_root,
        worker=_WORKER,
        prompt_version=TEN_K_NARRATIVE_FIELD_PROMPT_VERSION,
        doc_id=doc_id,
    )
    if agreed_identity is None:
        return None

    commit_staged_cache_writes(
        cache_root=cache_root, worker=_WORKER, pending=pending_writes
    )
    return TenKOutput(**agreed_identity, **narrative_partials)


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
    raw_doc_tokens = count_tokens(raw_doc)
    if raw_doc_tokens >= SINGLE_SHOT_TOKEN_CUTOFF and chunkset is None:
        # Silently falling through to single-shot here would send a 100K+
        # token doc as one user_content block — billed as fresh input
        # every call and likely to exceed the model's input window.
        # Long docs MUST go through RAG; raise loudly so the caller wires
        # the chunker rather than burning the API budget.
        raise ValueError(
            f"raw_doc has {raw_doc_tokens} tokens "
            f"(>= {SINGLE_SHOT_TOKEN_CUTOFF} cutoff) but no chunkset was "
            "supplied; long 10-Ks require the RAG path. Wire the chunker "
            "upstream and pass `chunkset=parse_filing(...)`."
        )
    narrative_is_rag = (
        chunkset is not None and raw_doc_tokens >= SINGLE_SHOT_TOKEN_CUTOFF
    )
    if narrative_is_rag:
        if retrieve_fn is None:
            raise ValueError(
                "RAG branch requires an explicit retrieve_fn; the backfill "
                "orchestrator owns wiring it to hybrid_retrieve + rerank."
            )
        assert chunkset is not None  # narrow for mypy
        narrative = _extract_ten_k_rag(
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
        )
    if narrative is None:
        return None

    # 2. Item 8 financials: independent of narrative path. Iterates EVERY
    # table parent in chunkset (document) order — a real 10-K Item 8
    # emits income-statement, balance-sheet, and cash-flow as separate
    # `<table>` parents, each carrying its own line items. Per-table
    # failures quarantine that table only (its own cache key + record);
    # surviving tables still contribute. None of `parts` is acceptable
    # (all tables quarantined or no tables at all) — leaves
    # `financials=None` on the merged output, indistinguishable today
    # from "no Item 8 supplied" but a follow-up may add a "no_data"
    # discriminator.
    if chunkset is None:
        return narrative
    table_parents = [p for p in chunkset.parents if p.table_html is not None]
    if not table_parents:
        return narrative
    financials_parts: list[TenKFinancials] = []
    for i, parent in enumerate(table_parents):
        assert parent.table_html is not None  # narrow for mypy
        part = _extract_item8_financials(
            parent_table_html=parent.table_html,
            doc_id=doc_id,
            cache_index=i,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
        )
        if part is not None:
            financials_parts.append(part)
    financials = _merge_financials(financials_parts) if financials_parts else None
    return narrative.model_copy(update={"financials": financials})


__all__ = ["RetrieveFn", "extract_ten_k"]
