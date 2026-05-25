"""Contextual chunking — Anthropic's contextual-retrieval pattern.

For each `ChildChunk` produced by `extract.chunking`, generate a one-line
context (≤120 Anthropic tokens) that situates the chunk within its source
filing, and pair it with the original chunk in a `ContextualChildChunk`.
Downstream embedding prepends `context` to `child.text` before sending the
result through Voyage / BGE — Anthropic reports ~50% retrieval lift from
this transformation.

Four things make this module behave correctly under the cost / determinism
constraints of the rest of `extract/`:

1. **Anthropic prompt cache.** Instructions + document metadata (ticker,
   fiscal_period, doc_type, section_name) + parent text live in the
   cacheable system block via `cached_system_block`. All children of the
   same parent share the cached prefix, so the per-child marginal cost is
   the chunk-text tokens plus the response.
2. **Metadata injection.** The prompt asks the model to name the
   ticker / fiscal period / doc type / section; those values must therefore
   be visible to the model. They live in the cached system block alongside
   the parent text, so the model never has to hallucinate them or copy
   few-shot example values.
3. **Content-hash cache.** Successfully-generated contexts persist at
   `data/cache/extract/contextual_chunking/<sha>.json`. The key includes
   `CONTEXTUAL_CHUNK_PROMPT_VERSION` (ADR D6, INV-6), so bumping the prompt
   forces fresh generation and never silently reuses stale text. Empty
   contexts (drops) are NOT cached — a transient model glitch can recover
   on the next batch instead of being baked into the cache forever.
4. **Output-token cap via Anthropic's own count.** `response.usage.
   output_tokens` is Anthropic's tokenizer; cl100k_base (used elsewhere
   for chunk-size budgeting) can disagree by 10-15% on tickers and jargon.
   Validating with the SDK's own count is the only honest way to honor
   the "≤100 tokens" AC the prompt promises the model.

Failure modes:
- Anthropic returns >100-token line, or `stop_reason="max_tokens"`
  (truncated fragment), or empty content → emit `ContextualChildChunk(
  context="")` and DO NOT cache. Span outcome = "dropped".
- Per-child SDK exception (rate limit, circuit open, cost cap) → log,
  emit `ContextualChildChunk(context="")` for that child, DO NOT cache,
  continue the batch. Partial progress survives; a re-run picks up the
  failed children since their cache slot is empty. Span outcome = "error".

OTel `extract.outcome` enum values emitted by this module:
`{cache_hit, persisted, dropped, error}`. `quarantined` is intentionally
not emitted — contextual chunking is a retrieval-quality feature, not a
citation-grounding invariant.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import anthropic
from anthropic.types import Message
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.chunking import (
    ChildChunk,
    ChunkSet,
    ParentChunk,
)
from auto_research.extract.chunking._packing import _parent_id
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.prompts.contextual_chunk import (
    CONTEXTUAL_CHUNK_PROMPT,
    CONTEXTUAL_CHUNK_PROMPT_VERSION,
)
from auto_research.telemetry import truncate_status_description as _truncate

_WORKER = "contextual_chunking"  # module identity (matches cache dir name)
_TASK = "contextual_chunk"  # routing-table key + telemetry tag
_SCHEMA_VERSION = "v1"  # the stored payload shape: {"context": str}
# Anthropic max_tokens budget. Sits above the 100-token cap so the model
# has room to terminate cleanly; truncated outputs (stop_reason="max_tokens")
# are rejected explicitly below.
_MAX_TOKENS = 160
# Hard ceiling on the generated context, in Anthropic output tokens (NOT
# cl100k). The 20-token slack above the prompt's "under 100 tokens"
# directive admits an honest model overshoot without dropping a usable
# context for a 5-10% tokenizer disagreement.
_MAX_CONTEXT_TOKENS = 120
_DECODING_PARAMS: dict[str, object] = {"max_tokens": _MAX_TOKENS}
_WHITESPACE_RE = re.compile(r"\s+")

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

# Module-level singleton so per-worker cost-cap + circuit-breaker state
# accumulates across calls in production. Tests injecting their own
# `anthropic_client` bypass the singleton. Lock guards the lazy init
# against the LangGraph-threadpool fan-out path.
_CLIENT: ExtractionFn | None = None
_CLIENT_LOCK = threading.Lock()


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(worker=_WORKER, anthropic_client=anthropic_client)
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = make_extraction_client(worker=_WORKER)
        return _CLIENT


@dataclass(frozen=True)
class ContextualChildChunk:
    """A `ChildChunk` paired with its generated one-line context.

    `child` is the original chunk, unmodified — preserved for audit so the
    raw retrieval unit (and its `char_span`) is always recoverable. `context`
    is the generated sentence (empty string if generation was dropped or
    raised). `embedding_text` is what downstream LanceDB writes embed: the
    context prepended to the chunk text with a blank-line separator,
    matching Anthropic's published pattern.
    """

    child: ChildChunk
    context: str

    @property
    def embedding_text(self) -> str:
        if not self.context:
            return self.child.text
        return f"{self.context}\n\n{self.child.text}"


def _system_block_text(parent: ParentChunk) -> str:
    """Combine instructions + per-parent metadata + parent text into one
    cacheable system block.

    Metadata is rendered as a small XML-style header the model can read off
    directly — without this, the prompt asks the model to name fields
    (ticker, fiscal period, doc type, section) that it can't see, and the
    model either hallucinates or copies few-shot example values.
    """
    md = parent.metadata
    return (
        f"{CONTEXTUAL_CHUNK_PROMPT}\n\n"
        f"<doc_metadata>\n"
        f"  ticker: {md.ticker}\n"
        f"  fiscal_period: {md.fiscal_period}\n"
        f"  doc_type: {md.doc_type}\n"
        f"  section: {parent.section_name}\n"
        f"</doc_metadata>\n"
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

    `filing_date` is rendered via `strftime` (not `isoformat`) so a
    `datetime` accidentally passed where a `date` is expected does not
    silently produce a divergent key for the same logical day.
    """
    metadata = {
        "ticker": child.metadata.ticker,
        "filing_date": child.metadata.filing_date.strftime("%Y-%m-%d"),
        "fiscal_period": child.metadata.fiscal_period,
        "doc_type": child.metadata.doc_type,
        "doc_id": child.metadata.doc_id,
        "section_name": child.section_name,
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


def _extract_text(response: Message) -> str:
    """Pull the text blocks off an Anthropic Message and normalize whitespace.

    Multi-block responses are joined with a single space and all whitespace
    runs (including embedded newlines) collapse to one space. This enforces
    the "one short sentence" contract documented on `ContextualChildChunk`
    — without it, a multi-block or multi-line model response leaks
    irregular newline runs into `embedding_text`.
    """
    raw = " ".join(b.text for b in response.content if b.type == "text")
    return _WHITESPACE_RE.sub(" ", raw).strip()


def contextualize_chunks(
    *,
    chunkset: ChunkSet,
    cache_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> tuple[ContextualChildChunk, ...]:
    """Generate a one-line context per `ChildChunk` and return paired results.

    Pure batch over `chunkset.children`. Cache hits short-circuit the SDK;
    misses route through the wrapped extraction client (cost-cap, circuit-
    breaker, retry, ephemeral prompt cache on the system block). Per-child
    SDK failures are caught and the failed children pass through with
    `context=""` so partial progress survives — see the module docstring
    "Failure modes" section.
    """
    effective_cache_root = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    model_id = route_model(_WORKER, _TASK)

    parents_by_id: dict[str, ParentChunk] = {
        _parent_id(p): p for p in chunkset.parents
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

            cached = content_cache.read(effective_cache_root, _WORKER, key)
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
                # Per-child failure (rate-limit retries exhausted, circuit
                # open, cost cap) — log, emit empty context, DO NOT cache,
                # continue the batch. Cache is left untouched so a re-run
                # picks up this child fresh once the underlying failure
                # clears.
                _log.warning(
                    "contextual_chunk: SDK exception for doc_id=%s parent_id=%s: %s",
                    child.metadata.doc_id,
                    child.parent_id,
                    exc,
                )
                span.set_attribute("extract.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                out.append(ContextualChildChunk(child=child, context=""))
                continue

            context = _validate_response(response, child=child, span=span)

            if not context:
                # Drop path: empty / over-cap / truncated. Don't cache —
                # next batch retries; otherwise one transient glitch
                # becomes a permanent retrieval-quality regression.
                out.append(ContextualChildChunk(child=child, context=""))
                continue

            content_cache.write(
                effective_cache_root, _WORKER, key, {"context": context}
            )
            span.set_attribute("extract.outcome", "persisted")
            out.append(ContextualChildChunk(child=child, context=context))

    return tuple(out)


def _validate_response(
    response: Message,
    *,
    child: ChildChunk,
    span: trace.Span,
) -> str:
    """Return a clean context string, or "" if the response should be dropped.

    Three drop conditions, each logged at WARNING with enough context to
    diagnose in dashboards:

    - Empty / whitespace-only text (model refusal, content-policy strip,
      or pure tool-use response).
    - `stop_reason == "max_tokens"` — the response is a truncated fragment
      that would otherwise be cached as a sentence-without-period.
    - Anthropic output_tokens > `_MAX_CONTEXT_TOKENS` — the prompt told the
      model to stay under 100 tokens; values that drift well past the cap
      indicate the model misread the budget and the result is too long
      to prepend usefully.
    """
    raw = _extract_text(response)
    if not raw:
        _log.warning(
            "contextual_chunk: empty response for doc_id=%s parent_id=%s",
            child.metadata.doc_id,
            child.parent_id,
        )
        span.set_attribute("extract.outcome", "dropped")
        span.set_attribute("extract.drop_reason", "empty")
        return ""

    if response.stop_reason == "max_tokens":
        _log.warning(
            "contextual_chunk: truncated (stop_reason=max_tokens) for "
            "doc_id=%s parent_id=%s",
            child.metadata.doc_id,
            child.parent_id,
        )
        span.set_attribute("extract.outcome", "dropped")
        span.set_attribute("extract.drop_reason", "truncated")
        return ""

    if response.usage.output_tokens > _MAX_CONTEXT_TOKENS:
        _log.warning(
            "contextual_chunk: dropped %d-token context for doc_id=%s "
            "parent_id=%s (cap=%d)",
            response.usage.output_tokens,
            child.metadata.doc_id,
            child.parent_id,
            _MAX_CONTEXT_TOKENS,
        )
        span.set_attribute("extract.outcome", "dropped")
        span.set_attribute("extract.drop_reason", "over_cap")
        return ""

    return raw


__all__ = [
    "ContextualChildChunk",
    "contextualize_chunks",
]
