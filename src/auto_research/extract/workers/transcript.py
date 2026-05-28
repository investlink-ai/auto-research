"""Earnings-transcript extraction worker.

Single-shot only. A 1-3 hour earnings call transcribes to ~25-60K
Anthropic tokens — well under `SINGLE_SHOT_TOKEN_CUTOFF`, so no RAG
path is wired here.

Composes `_common.run_single_shot_extraction` with the transcript
prompt and `TranscriptOutput` schema. Routing-table key
`prepared_remarks_tone` is the default task; `q_and_a_evasiveness`
(Sonnet) is routed for nuance fields when a future worker per-field
breakout lands, but this single-shot call uses one model for the whole
output. The current routing for the unified call is the Sonnet tier
because the Q&A evasiveness judgment is the bottleneck — Haiku
underperforms on subjective Q&A evasion per spec §7.3.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.transcript import (
    TRANSCRIPT_PROMPT,
    TRANSCRIPT_PROMPT_VERSION,
)
from auto_research.extract.schemas import TranscriptOutput
from auto_research.extract.workers._common import run_single_shot_extraction

_WORKER = "transcript"
# Single-shot extraction produces the entire TranscriptOutput in one
# call; route by the highest-tier field on the output to honor the spec
# §7.3 "Q&A evasiveness ⇒ Sonnet" rule for the unified call.
_TASK = "q_and_a_evasiveness"
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

    Returns `None` when the output failed any parse / span-resolution /
    grounding check; the caller MUST treat `None` as "do not persist."
    """
    return run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        task=_TASK,
        prompt=TRANSCRIPT_PROMPT,
        prompt_version=TRANSCRIPT_PROMPT_VERSION,
        output_model=TranscriptOutput,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root
        if cache_root is not None
        else content_cache.DEFAULT_CACHE_ROOT,
        quarantine_root=quarantine_root
        if quarantine_root is not None
        else DEFAULT_QUARANTINE_ROOT,
        anthropic_client=anthropic_client,
    )


__all__ = ["extract_transcript"]
