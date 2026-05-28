"""Earnings-transcript extraction worker.

Binary split — two calls per transcript, one per tier (spec §7.3):

1. **Prepared-remarks tone (Haiku).** `TRANSCRIPT_PREPARED_REMARKS_PROMPT`
   against `TranscriptPreparedRemarksPartial` — templated tone
   classification on the prepared-remarks portion.
2. **Q&A + forward statements (Sonnet).** `TRANSCRIPT_QA_PROMPT`
   against `TranscriptQAPartial` — cross-utterance reasoning on Q&A
   evasiveness plus forward-statement extraction. The two fields
   share one Sonnet call rather than splitting into a third because
   forward-statements reasoning is the same Sonnet-tier work the
   evasiveness judgment already pays for.

The split eliminates the unified-call waste where every transcript paid
Sonnet pricing on the templated prepared-remarks half.

Single-shot only. A 1-3 hour earnings call transcribes to ~25-60K
Anthropic tokens — well under `SINGLE_SHOT_TOKEN_CUTOFF`, so neither
half needs RAG.

The transcript identity fields (ticker, event_datetime) are extracted
on BOTH calls and cross-checked; disagreement quarantines, matching
the 10-K RAG worker's identity-disagreement discipline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.transcript_split import (
    TRANSCRIPT_PREPARED_REMARKS_PROMPT,
    TRANSCRIPT_PREPARED_REMARKS_PROMPT_VERSION,
    TRANSCRIPT_QA_PROMPT,
    TRANSCRIPT_QA_PROMPT_VERSION,
)
from auto_research.extract.schemas import (
    TranscriptOutput,
    TranscriptPreparedRemarksPartial,
    TranscriptQAPartial,
)
from auto_research.extract.workers._common import (
    _write_quarantine,
    run_single_shot_extraction,
)

_WORKER = "transcript"
_PREPARED_REMARKS_TASK = "prepared_remarks_tone"
_QA_TASK = "q_and_a_evasiveness"
_MAX_TOKENS = 8192


def extract_transcript(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> TranscriptOutput | None:
    """Extract a TranscriptOutput from a raw earnings-call transcript.

    Two calls per transcript: prepared-remarks tone (Haiku) and
    Q&A evasiveness + forward statements (Sonnet). Per-call cache
    writes are staged; both must succeed AND their identity fields
    (ticker, event_datetime) must agree before the worker commits
    either to cache. A mid-loop quarantine returns `None` without
    persisting either call's result.

    Returns `None` when ANY of the LLM calls fails parse /
    span-resolution / grounding; the caller MUST treat `None` as
    "do not persist."
    """
    cache_root_resolved = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    quarantine_root_resolved = (
        quarantine_root
        if quarantine_root is not None
        else DEFAULT_QUARANTINE_ROOT
    )

    pending_writes: list[tuple[str, dict[str, Any]]] = []

    def _stage(cache_key: str, payload: dict[str, Any]) -> None:
        pending_writes.append((cache_key, payload))

    prepared = run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=f"{doc_id}#prepared_remarks",
        worker=_WORKER,
        task=_PREPARED_REMARKS_TASK,
        prompt=TRANSCRIPT_PREPARED_REMARKS_PROMPT,
        prompt_version=TRANSCRIPT_PREPARED_REMARKS_PROMPT_VERSION,
        output_model=TranscriptPreparedRemarksPartial,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root_resolved,
        quarantine_root=quarantine_root_resolved,
        anthropic_client=anthropic_client,
        cache_write_handler=_stage,
    )
    if prepared is None:
        return None

    qa = run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=f"{doc_id}#qa",
        worker=_WORKER,
        task=_QA_TASK,
        prompt=TRANSCRIPT_QA_PROMPT,
        prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
        output_model=TranscriptQAPartial,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root_resolved,
        quarantine_root=quarantine_root_resolved,
        anthropic_client=anthropic_client,
        cache_write_handler=_stage,
    )
    if qa is None:
        return None

    # Identity-field consistency check (same discipline as the 10-K RAG
    # path). Two calls; if they disagree on `ticker` or
    # `event_datetime`, at least one hallucinated, and silently keeping
    # one would corrupt downstream attribution. Quarantine without
    # committing staged cache writes.
    if prepared.ticker != qa.ticker:
        _write_quarantine(
            quarantine_root=quarantine_root_resolved,
            worker=_WORKER,
            prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
            doc_id=f"{doc_id}#identity-disagreement",
            parsed={
                "field": "ticker",
                "prepared_remarks_value": prepared.ticker,
                "qa_value": qa.ticker,
            },
            error=(
                "transcript per-call identity disagreement on `ticker`: "
                f"prepared={prepared.ticker!r}, qa={qa.ticker!r}"
            ),
        )
        return None
    if prepared.event_datetime != qa.event_datetime:
        _write_quarantine(
            quarantine_root=quarantine_root_resolved,
            worker=_WORKER,
            prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
            doc_id=f"{doc_id}#identity-disagreement",
            parsed={
                "field": "event_datetime",
                "prepared_remarks_value": (
                    prepared.event_datetime.isoformat()
                    if prepared.event_datetime is not None
                    else None
                ),
                "qa_value": (
                    qa.event_datetime.isoformat()
                    if qa.event_datetime is not None
                    else None
                ),
            },
            error=(
                "transcript per-call identity disagreement on "
                "`event_datetime`: "
                f"prepared={prepared.event_datetime!r}, qa={qa.event_datetime!r}"
            ),
        )
        return None

    for cache_key, payload in pending_writes:
        content_cache.write(cache_root_resolved, _WORKER, cache_key, payload)

    return TranscriptOutput(
        ticker=prepared.ticker,
        event_datetime=prepared.event_datetime,
        prepared_remarks_tone=prepared.prepared_remarks_tone,
        q_and_a_evasiveness=qa.q_and_a_evasiveness,
        forward_statements=qa.forward_statements,
    )


__all__ = ["extract_transcript"]
