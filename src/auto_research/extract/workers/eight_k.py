"""8-K extraction worker.

Single-shot only. 8-K filings are short (typically 2-10 pages, well
under `SINGLE_SHOT_TOKEN_CUTOFF`) so the RAG path is not wired here —
the contextual-chunking + retrieval stack would add latency and cost
without retrieval benefit at this doc size.

Composes `_common.run_single_shot_extraction` with the 8-K prompt and
`EightKOutput` schema. Production callers omit `cache_root` /
`quarantine_root` and get the package defaults; the worker-keyed
singleton in `_common._CLIENTS` preserves `@cost_cap` +
`@circuit_breaker` state across docs in a single backfill run.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.eight_k import (
    EIGHT_K_PROMPT,
    EIGHT_K_PROMPT_VERSION,
)
from auto_research.extract.schemas import EightKOutput
from auto_research.extract.workers._common import run_single_shot_extraction

_WORKER = "eight_k"
_TASK = "event_classification"  # matches EightKOutput.event_classification
_MAX_TOKENS = 4096


def extract_eight_k(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> EightKOutput | None:
    """Extract an EightKOutput from a raw 8-K text.

    Returns `None` when the output failed any parse / span-resolution
    / grounding check; the caller MUST treat `None` as "do not persist."
    """
    return run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        task=_TASK,
        prompt=EIGHT_K_PROMPT,
        prompt_version=EIGHT_K_PROMPT_VERSION,
        output_model=EightKOutput,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root
        if cache_root is not None
        else content_cache.DEFAULT_CACHE_ROOT,
        quarantine_root=quarantine_root
        if quarantine_root is not None
        else DEFAULT_QUARANTINE_ROOT,
        anthropic_client=anthropic_client,
    )


__all__ = ["extract_eight_k"]
