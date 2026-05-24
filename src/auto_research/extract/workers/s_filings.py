"""S-1 / S-3 extraction worker — first end-to-end validator (Issue #11).

Composes the W1 extraction primitives:

    prompt + client + cache + guardrails -> SFilingOutput | None

Flow:

1. Build the cache key from the full completion config (raw_doc bytes,
   prompt version, schema version, routed model, decoding params).
2. Look up `data/cache/extract/s_filings/<sha>.json`. Hit -> deserialize
   into `SFilingOutput`, return.
3. Miss -> invoke the Anthropic client (with reliability + caching from
   `make_extraction_client`), parse the JSON content block into
   `SFilingOutput`, validate via `validate_or_quarantine`.
4. On validation success: write to cache, return the output. On failure:
   the guardrail already wrote a QuarantineRecord; return None.

`cache_root` and `quarantine_root` are injected so tests can pass
`tmp_path` and stay hermetic. Production callers omit them and get the
package defaults. `anthropic_client` is injected the same way
`make_extraction_client` accepts it — production callers omit it; tests
pass a `MagicMock`.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.client import make_extraction_client
from auto_research.extract.guardrails import (
    DEFAULT_QUARANTINE_ROOT,
    validate_or_quarantine,
)
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.schemas import SFilingOutput

_WORKER = "s_filings"
_TASK = "dilution_event"  # matches SFilingOutput.dilution_event field name
_MAX_TOKENS = 4096
_DECODING_PARAMS: dict[str, object] = {"max_tokens": _MAX_TOKENS}


def extract_s_filing(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> SFilingOutput | None:
    """Extract an SFilingOutput from a raw S-1/S-3 text.

    Returns `None` when the output failed citation grounding; the caller
    MUST treat None as "do not persist."
    """
    effective_cache_root = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    effective_quarantine_root = (
        quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT
    )
    model_id = route_model(_WORKER, _TASK)

    key = content_cache.cache_key(
        raw_doc=raw_doc.encode(),
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        schema_version=SFilingOutput.SCHEMA_VERSION,
        model_id=model_id,
        decoding_params=_DECODING_PARAMS,
    )

    cached = content_cache.read(effective_cache_root, _WORKER, key)
    if cached is not None:
        return SFilingOutput.model_validate(cached)

    client = make_extraction_client(
        worker=_WORKER,
        anthropic_client=anthropic_client,
    )
    response = client(
        task=_TASK,
        system_prompt=S_FILINGS_DILUTION_PROMPT.format(source_text=raw_doc),
        user_content=raw_doc,
        max_tokens=_MAX_TOKENS,
    )

    # Anthropic responses are a list of content blocks; the worker expects
    # one TextBlock containing JSON. Any other shape is model misbehavior —
    # let `model_validate` raise so quarantine catches it.
    text = "".join(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    output = SFilingOutput.model_validate(parsed)

    validated = validate_or_quarantine(
        output,
        source_text=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        quarantine_root=effective_quarantine_root,
    )
    if validated is None:
        return None

    content_cache.write(
        effective_cache_root, _WORKER, key, validated.model_dump(mode="json")
    )
    return validated


__all__ = ["extract_s_filing"]
