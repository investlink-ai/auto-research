"""10-K extraction worker.

Two paths, one entry point:

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
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT,
    TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative_field import (
    TEN_K_NARRATIVE_FIELD_CONFIGS,
    TEN_K_NARRATIVE_FIELD_PROMPT,
    TEN_K_NARRATIVE_FIELD_PROMPT_VERSION,
)
from auto_research.extract.schemas import TenKOutput
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
_NARRATIVE_MAX_TOKENS = 8192

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
    # TODO: remove rag_defaults once critical_accounting_estimate_changes
    # has a per-field config in TEN_K_NARRATIVE_FIELD_CONFIGS — the loop
    # will then populate it directly and the default becomes a dead
    # branch that narrative_partials always overrides.
    rag_defaults: dict[str, Any] = {
        "critical_accounting_estimate_changes": [],
    }
    return TenKOutput(**agreed_identity, **rag_defaults, **narrative_partials)


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

    # Narrative path: single-shot if short OR no chunkset, RAG otherwise.
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
    return narrative


__all__ = ["RetrieveFn", "extract_ten_k"]
