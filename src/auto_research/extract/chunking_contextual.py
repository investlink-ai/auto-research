"""Contextual chunking — Anthropic's contextual-retrieval pattern.

For each `ChildChunk` produced by `extract.chunking`, generate a one-line
context (≤100 tokens) that situates the chunk within its source filing,
and pair it with the original chunk in a `ContextualChildChunk`. Downstream
embedding (Issue #15) prepends `context` to `child.text` before sending
the result through Voyage / BGE — Anthropic reports ~50% retrieval lift
from this transformation.

Three things make this module behave correctly under the cost / determinism
constraints of the rest of `extract/`:

1. **Anthropic prompt cache.** Instructions + parent text live in the
   cacheable system block via `cached_system_block`. All children of the
   same parent share the cached prefix, so the per-child marginal cost is
   the chunk-text tokens plus the response.
2. **Content-hash cache.** Generated contexts persist at
   `data/cache/extract/contextual_chunk/<sha>.json`. The key includes
   `CONTEXTUAL_CHUNK_PROMPT_VERSION` (ADR D6, INV-6), so bumping the prompt
   forces fresh generation and never silently reuses stale text.
3. **≤100-token cap.** A context that drifts over the cap is dropped
   (the chunk passes through with `context=""`) rather than quarantined.
   Contextual chunking is a retrieval-quality feature, not a citation-
   grounding invariant; a soft fall-through is the correct failure mode.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.chunking import (
    ChildChunk,
    ChunkSet,
    ParentChunk,
    count_tokens,
)
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.prompts.contextual_chunk import (
    CONTEXTUAL_CHUNK_PROMPT,
    CONTEXTUAL_CHUNK_PROMPT_VERSION,
)
from auto_research.telemetry import truncate_status_description as _truncate

_WORKER = "extract"
_TASK = "contextual_chunk"
_SCHEMA_VERSION = "v1"  # the stored payload shape: {"context": str}
_MAX_TOKENS = 160  # Anthropic budget; post-validate at 100 below.
_MAX_CONTEXT_TOKENS = 100
_DECODING_PARAMS: dict[str, object] = {"max_tokens": _MAX_TOKENS}

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

# Module-level singleton so per-worker cost-cap + circuit-breaker state
# accumulates across calls in production. Tests injecting their own
# `anthropic_client` bypass the singleton.
_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(worker=_WORKER, anthropic_client=anthropic_client)
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER)
    return _CLIENT


@dataclass(frozen=True)
class ContextualChildChunk:
    """A `ChildChunk` paired with its generated one-line context.

    `child` is the original chunk, unmodified — preserved for audit so the
    raw retrieval unit (and its `char_span`) is always recoverable. `context`
    is the generated sentence (empty string if generation produced a
    >100-token line or whitespace). `embedding_text` is what downstream
    LanceDB writes embed: the context prepended to the chunk text with a
    blank-line separator, matching Anthropic's published pattern.
    """

    child: ChildChunk
    context: str

    @property
    def embedding_text(self) -> str:
        if not self.context:
            return self.child.text
        return f"{self.context}\n\n{self.child.text}"


def _system_block_text(parent: ParentChunk) -> str:
    """Combine instructions + parent text into one cacheable system block."""
    return (
        f"{CONTEXTUAL_CHUNK_PROMPT}\n\n"
        f"<parent_passage>\n{parent.text}\n</parent_passage>"
    )


def _cache_payload_key(
    *,
    child: ChildChunk,
    parent: ParentChunk,
    model_id: str,
) -> str:
    """Build the content-hash cache key per ADR D6.

    Covers child text + parent text + document metadata + contextual prompt
    version + payload schema version + routed model + decoding params. Any
    change forces a fresh generation, so a `bump-prompt-version` edit
    cannot silently reuse stale cache.
    """
    metadata = {
        "ticker": child.metadata.ticker,
        "filing_date": child.metadata.filing_date.isoformat(),
        "fiscal_period": child.metadata.fiscal_period,
        "doc_type": child.metadata.doc_type,
        "doc_id": child.metadata.doc_id,
    }
    raw = json.dumps(
        {
            "child_text": child.text,
            "parent_text": parent.text,
            "metadata": metadata,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return content_cache.cache_key(
        raw_doc=raw,
        prompt_version=CONTEXTUAL_CHUNK_PROMPT_VERSION,
        schema_version=_SCHEMA_VERSION,
        model_id=model_id,
        decoding_params=_DECODING_PARAMS,
    )


def _extract_text(response: Any) -> str:
    """Pull the text block off an Anthropic Message. Defensive on shape."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def contextualize_chunks(
    *,
    chunkset: ChunkSet,
    cache_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> tuple[ContextualChildChunk, ...]:
    """Generate a one-line context per `ChildChunk` and return paired results.

    Pure batch over `chunkset.children`. Cache hits short-circuit the SDK;
    misses route through the wrapped extraction client (cost-cap, circuit-
    breaker, retry, ephemeral prompt cache on the system block).
    """
    effective_cache_root = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    model_id = route_model(_WORKER, _TASK)

    parents_by_id: dict[str, ParentChunk] = {
        f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}": p
        for p in chunkset.parents
    }

    client: ExtractionFn | None = None
    out: list[ContextualChildChunk] = []

    for child in chunkset.children:
        parent = parents_by_id.get(child.parent_id)
        if parent is None:
            raise ValueError(
                f"child references unknown parent_id={child.parent_id!r}; "
                f"chunkset.parents missing the corresponding ParentChunk"
            )

        key = _cache_payload_key(child=child, parent=parent, model_id=model_id)

        with _tracer.start_as_current_span("extract.contextual_chunk") as span:
            span.set_attribute("extract.worker", _WORKER)
            span.set_attribute("extract.doc_id", child.metadata.doc_id)
            span.set_attribute("extract.parent_id", child.parent_id)

            cached = content_cache.read(effective_cache_root, _TASK, key)
            if cached is not None:
                span.set_attribute("extract.outcome", "cache_hit")
                out.append(
                    ContextualChildChunk(child=child, context=cached["context"])
                )
                continue

            if client is None:
                client = _get_client(anthropic_client)

            try:
                response = client(
                    task=_TASK,
                    system_prompt=_system_block_text(parent),
                    user_content=child.text,
                    max_tokens=_MAX_TOKENS,
                )
            except Exception as exc:
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

            raw_context = _extract_text(response)
            if not raw_context:
                _log.warning(
                    "contextual_chunk: empty response for doc_id=%s parent_id=%s",
                    child.metadata.doc_id,
                    child.parent_id,
                )
                context = ""
            elif count_tokens(raw_context) > _MAX_CONTEXT_TOKENS:
                _log.warning(
                    "contextual_chunk: dropped %d-token context for doc_id=%s "
                    "parent_id=%s (cap=%d)",
                    count_tokens(raw_context),
                    child.metadata.doc_id,
                    child.parent_id,
                    _MAX_CONTEXT_TOKENS,
                )
                context = ""
            else:
                context = raw_context

            content_cache.write(effective_cache_root, _TASK, key, {"context": context})
            span.set_attribute("extract.outcome", "persisted")
            span.set_attribute("extract.context_dropped", context == "")
            out.append(ContextualChildChunk(child=child, context=context))

    return tuple(out)


__all__ = [
    "CONTEXTUAL_CHUNK_PROMPT_VERSION",
    "ContextualChildChunk",
    "contextualize_chunks",
]
