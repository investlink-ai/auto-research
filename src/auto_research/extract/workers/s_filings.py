"""S-1 / S-3 extraction worker.

Composes `_common.run_single_shot_extraction` with the dilution prompt
and `SFilingOutput` schema. The original `_strip_fence` / `_resolve_spans`
/ `_write_quarantine` helpers now live in `_common.py` and are shared
by the 10-K / transcript / 8-K workers, so the four workers route INV-2
the same way.

Production callers omit `cache_root` / `quarantine_root` and get the
package defaults. `anthropic_client` is injected by tests; production
callers omit it and share the module-level singleton (`_CLIENT`) so
the per-worker `@cost_cap` and `@circuit_breaker` state accumulates
across calls.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.schemas import SFilingOutput
from auto_research.extract.workers._common import run_single_shot_extraction

_WORKER = "s_filings"
_TASK = "dilution_event"  # matches SFilingOutput.dilution_event field name
_MAX_TOKENS = 4096

# Module-level lazy client so per-worker cost_cap + circuit_breaker state
# accumulates across calls. Each call site that passes its own
# `anthropic_client` (test injection) gets a fresh per-call client and
# bypasses the singleton — fine because per-test state isolation is what
# tests want.
_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    """Return the production singleton, or a fresh client for test injection.

    The singleton path is the one whose `@cost_cap` counter and
    `@circuit_breaker` state must persist across calls; the injection
    path is exercised only by tests that don't care about those.
    """
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(
            worker=_WORKER, anthropic_client=anthropic_client
        )
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER)
    return _CLIENT


def extract_s_filing(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> SFilingOutput | None:
    """Extract an SFilingOutput from a raw S-1/S-3 text.

    Returns `None` when the output failed any parse / span-resolution /
    grounding check; the caller MUST treat `None` as "do not persist."
    The raw model output is always captured in a QuarantineRecord on
    the failure path.
    """
    return run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        task=_TASK,
        prompt=S_FILINGS_DILUTION_PROMPT,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        output_model=SFilingOutput,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root
        if cache_root is not None
        else content_cache.DEFAULT_CACHE_ROOT,
        quarantine_root=quarantine_root
        if quarantine_root is not None
        else DEFAULT_QUARANTINE_ROOT,
        anthropic_client=anthropic_client,
        client_factory=_get_client,
    )


__all__ = ["extract_s_filing"]
