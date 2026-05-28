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

The two calls' identity fields (ticker, event_datetime) are extracted
on BOTH and cross-checked via `_common.check_identity_agreement`;
disagreement quarantines with the same discipline the 10-K RAG worker
applies across its 5 per-field calls. Adding a third tier (e.g.,
`earnings_estimates → Opus`) is mechanical: append a new
`_TranscriptCallConfig` and extend the final assembly block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import anthropic
from pydantic import BaseModel

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
    check_identity_agreement,
    commit_staged_cache_writes,
    run_single_shot_extraction,
)

_WORKER = "transcript"
_MAX_TOKENS = 8192


class _TranscriptCallConfig(NamedTuple):
    """Per-call config for the transcript binary-split loop.

    Adding a third tier (e.g., earnings-estimates @ Opus) is mechanical:
    append a fourth tuple. The final-assembly block that wires
    partials back into `TranscriptOutput` still has to know which
    field comes from which call, but the staging / identity / commit
    plumbing stays in the loop.
    """

    sub_doc_id_suffix: str
    task: str
    prompt: str
    prompt_version: str
    schema: type[BaseModel]


_TRANSCRIPT_CALLS: tuple[_TranscriptCallConfig, ...] = (
    _TranscriptCallConfig(
        sub_doc_id_suffix="prepared_remarks",
        task="prepared_remarks_tone",
        prompt=TRANSCRIPT_PREPARED_REMARKS_PROMPT,
        prompt_version=TRANSCRIPT_PREPARED_REMARKS_PROMPT_VERSION,
        schema=TranscriptPreparedRemarksPartial,
    ),
    _TranscriptCallConfig(
        sub_doc_id_suffix="qa",
        task="q_and_a_evasiveness",
        prompt=TRANSCRIPT_QA_PROMPT,
        prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
        schema=TranscriptQAPartial,
    ),
)


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

    partials_by_suffix: dict[str, BaseModel] = {}
    identity_seen: dict[str, list[Any]] = {
        "ticker": [],
        "event_datetime": [],
    }
    for call in _TRANSCRIPT_CALLS:
        result = run_single_shot_extraction(
            raw_doc=raw_doc,
            doc_id=f"{doc_id}#{call.sub_doc_id_suffix}",
            worker=_WORKER,
            task=call.task,
            prompt=call.prompt,
            prompt_version=call.prompt_version,
            output_model=call.schema,
            max_tokens=_MAX_TOKENS,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
            cache_write_handler=_stage,
        )
        if result is None:
            return None
        partials_by_suffix[call.sub_doc_id_suffix] = result
        for identity_field in identity_seen:
            identity_seen[identity_field].append(
                getattr(result, identity_field)
            )

    if (
        check_identity_agreement(
            identity_values=identity_seen,
            quarantine_root=quarantine_root_resolved,
            worker=_WORKER,
            prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
            doc_id=doc_id,
        )
        is None
    ):
        return None

    # Final assembly: the binary split's field-to-call mapping is the
    # one piece that genuinely doesn't fit in the loop (the loop emits
    # partials with overlapping identity fields but disjoint payload
    # fields). A future 3-way split adds one new call to
    # `_TRANSCRIPT_CALLS` above and one new line here — no other
    # changes to the loop / staging / identity-check plumbing.
    prepared = partials_by_suffix["prepared_remarks"]
    qa = partials_by_suffix["qa"]
    assert isinstance(prepared, TranscriptPreparedRemarksPartial)
    assert isinstance(qa, TranscriptQAPartial)

    commit_staged_cache_writes(
        cache_root=cache_root_resolved,
        worker=_WORKER,
        pending=pending_writes,
    )
    return TranscriptOutput(
        ticker=prepared.ticker,
        event_datetime=prepared.event_datetime,
        prepared_remarks_tone=prepared.prepared_remarks_tone,
        q_and_a_evasiveness=qa.q_and_a_evasiveness,
        forward_statements=qa.forward_statements,
    )


__all__ = ["extract_transcript"]
