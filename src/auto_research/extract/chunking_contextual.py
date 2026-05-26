"""Contextual chunking — Anthropic's contextual-retrieval pattern.

For each `ChildChunk` produced by `extract.chunking`, generate a one-line
context (≤100 Anthropic tokens) that situates the chunk within its source
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
   the chunk-text tokens plus the response. The parent text is
   XML-escaped before interpolation so a filer cannot inject metadata or
   spoof the prompt structure through SEC-filing prose.
2. **Metadata injection.** The prompt asks the model to name the
   ticker / fiscal period / doc type / section; those values must therefore
   be visible to the model. They live in the cached system block BEFORE
   the parent passage, so the model never has to hallucinate them or copy
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
   Validating with the SDK's own count is the only way to honor the
   "≤100 tokens" AC the prompt promises the model.

Failure modes:
- Anthropic returns >100 output_tokens, OR `stop_reason` not in
  {end_turn, stop_sequence} (max_tokens / refusal / pause_turn / tool_use
  — any of which produce text we cannot safely embed), OR empty content
  → emit `ContextualChildChunk(context="")` and DO NOT cache. Span
  outcome = "dropped", status = ERROR (so error-rate dashboards see it).
- Per-child `anthropic.APIError` (retries exhausted, transient network /
  500 / 429) → log, emit `ContextualChildChunk(context="")`, DO NOT
  cache, continue the batch. `CostCapExceeded` and `CircuitOpen`
  propagate so the batch aborts on terminal signals.

OTel `extract.outcome` enum values emitted by this module:
`{cache_hit, persisted, dropped, error}` (see `docs/ARCHITECTURE.md`).
`quarantined` is intentionally not emitted — contextual chunking is a
retrieval-quality feature, not a citation-grounding invariant.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import anthropic
from anthropic.types import Message
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.chunking import (
    CHUNKER_VERSION,
    ChildChunk,
    ChunkSet,
    ParentChunk,
    _parent_id,
)
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
# can terminate cleanly under cap; truncated outputs (stop_reason="max_tokens")
# are rejected explicitly below — the budget exists to make terminated
# outputs cheaper, not to admit longer ones.
_MAX_TOKENS = 160
# Hard ceiling on the generated context, in Anthropic output tokens. The
# prompt tells the model "under 100 tokens" and this enforces it; matches
# the AC. Output may still be over by tokenizer slack (~5-10 tokens), in
# which case the drop path applies.
_MAX_CONTEXT_TOKENS = 100
_DECODING_PARAMS: dict[str, object] = {"max_tokens": _MAX_TOKENS}
_WHITESPACE_RE = re.compile(r"\s+")
# Only these terminal stop reasons signal a complete, usable response.
# `max_tokens` is a truncated fragment; `refusal` is a model-side refusal
# sentence; `pause_turn`/`tool_use` produce text we don't want to embed.
_VALID_STOP_REASONS: frozenset[str] = frozenset({"end_turn", "stop_sequence"})

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

# Module-level singleton so per-worker cost-cap + circuit-breaker state
# accumulates across calls in production. Tests injecting their own
# `anthropic_client` bypass the singleton. Lock guards the lazy init
# against the LangGraph-threadpool fan-out path; fast-path check outside
# the lock so warm-path callers don't serialize through it.
_CLIENT: ExtractionFn | None = None
_CLIENT_LOCK = threading.Lock()


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(worker=_WORKER, anthropic_client=anthropic_client)
    if _CLIENT is not None:
        return _CLIENT
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

    Metadata is rendered as a small XML-style header the model reads off
    directly — without this, the prompt asks the model to name fields
    (ticker, fiscal period, doc type, section) that it can't see, and the
    model either hallucinates or copies few-shot example values.

    Parent text is XML-escaped before interpolation so an adversarial
    filing whose prose contains literal `</parent_passage>` or
    `</doc_metadata>` cannot close the structural framing early and
    spoof the metadata header. `section_name` is constrained by
    `_section_name_from_title` (regex-anchored "Item N") so it never
    needs escaping; metadata fields come from the ingest pipeline, not
    the filer.
    """
    md = parent.metadata
    safe_parent_text = _xml_escape(parent.text)
    return (
        f"{CONTEXTUAL_CHUNK_PROMPT}\n\n"
        f"<doc_metadata>\n"
        f"  ticker: {md.ticker}\n"
        f"  fiscal_period: {md.fiscal_period}\n"
        f"  doc_type: {md.doc_type}\n"
        f"  section: {parent.section_name}\n"
        f"</doc_metadata>\n"
        f"<parent_passage>\n{safe_parent_text}\n</parent_passage>"
    )


def _cache_payload_key(
    *,
    child: ChildChunk,
    parent: ParentChunk,
    model_id: str,
) -> str:
    """Build the content-hash cache key per ADR D6.

    Covers child text + parent text + document metadata + chunker
    version + contextual prompt version + payload schema version +
    routed model + decoding params. Any change forces a fresh
    generation, so a `bump-prompt-version`-style edit (whether the
    prompt, the chunker, or the embed model) cannot silently reuse
    stale cache.

    `CHUNKER_VERSION` lives in the inner payload (not the
    `prompt_version` slot of `cache_key`) so a chunker bump
    invalidates the contextual cache transitively even when the
    contextual prompt version stays put — and vice versa. This is the
    orthogonality the issue's design proposal calls out (#67).

    `EMBED_MODEL_VERSION` is deliberately NOT in this key: the embed
    model is a downstream consumer of `ContextualChildChunk` output,
    not an input. Bumping it must re-embed, not re-call the LLM for
    contextual text.

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
            "chunker_version": CHUNKER_VERSION,
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


def _validate_response(
    response: Message,
    *,
    child: ChildChunk,
    span: trace.Span,
) -> str:
    """Return a clean context string, or "" if the response should be dropped.

    Three drop conditions, each logged at WARNING and surfaced on the OTel
    span with both `extract.outcome="dropped"` and an ERROR status (so
    span-status dashboards count drops):

    - Empty / whitespace-only text (model refusal-with-no-text, content-
      policy strip, or pure tool-use response).
    - `stop_reason` not in {end_turn, stop_sequence}: catches max_tokens
      (truncated fragment), refusal (model refusal sentence — would
      otherwise be cached as the embedding context forever), and
      pause_turn / tool_use (response not intended as a final answer).
    - Anthropic `output_tokens` > `_MAX_CONTEXT_TOKENS`: the model
      overshot the 100-token directive; result is too long to prepend.
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
        span.set_status(Status(StatusCode.ERROR, "dropped:empty"))
        return ""

    if response.stop_reason not in _VALID_STOP_REASONS:
        reason = response.stop_reason or "none"
        _log.warning(
            "contextual_chunk: bad stop_reason=%s for doc_id=%s parent_id=%s",
            reason,
            child.metadata.doc_id,
            child.parent_id,
        )
        span.set_attribute("extract.outcome", "dropped")
        span.set_attribute("extract.drop_reason", reason)
        span.set_status(Status(StatusCode.ERROR, f"dropped:{reason}"))
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
        span.set_status(Status(StatusCode.ERROR, "dropped:over_cap"))
        return ""

    return raw


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
    `anthropic.APIError` failures (retries exhausted) emit
    `ContextualChildChunk(context="")` and continue the batch so partial
    progress survives. `CostCapExceeded` and `CircuitOpen` propagate so
    the batch aborts on terminal signals.

    Client construction is eager (before the loop) so a missing
    `ANTHROPIC_API_KEY` or other setup failure surfaces before any work
    starts — never as a half-done batch.
    """
    effective_cache_root = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    model_id = route_model(_WORKER, _TASK)

    parents_by_id: dict[str, ParentChunk] = {
        _parent_id(p): p for p in chunkset.parents
    }

    # Eager init — fail fast on missing API key / SDK construction errors,
    # rather than half-way through the batch.
    client = _get_client(anthropic_client)
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

            try:
                response = client(
                    task=_TASK,
                    system_prompt=_system_block_text(parent),
                    user_content=child.text,
                    max_tokens=_MAX_TOKENS,
                )
            except anthropic.APIError as exc:
                # Retries exhausted on a transient API failure. Log, mark
                # the span as error, emit empty context, DO NOT cache (so a
                # re-run retries once the API recovers), continue the batch.
                # CostCapExceeded / CircuitOpen are NOT APIError subclasses
                # — they propagate and abort the batch, which is correct
                # since they signal terminal conditions.
                _log.warning(
                    "contextual_chunk: APIError for doc_id=%s parent_id=%s: %s",
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
                # Drop path — span attrs already set by _validate_response.
                # Don't cache; next batch retries. The model's USD spend is
                # accumulated regardless (the SDK call succeeded), so a
                # persistently-dropping prompt will consume budget across
                # runs — accept this as the cost of not poisoning the cache.
                out.append(ContextualChildChunk(child=child, context=""))
                continue

            content_cache.write(
                effective_cache_root, _WORKER, key, {"context": context}
            )
            span.set_attribute("extract.outcome", "persisted")
            out.append(ContextualChildChunk(child=child, context=context))

    return tuple(out)


__all__ = [
    "ContextualChildChunk",
    "contextualize_chunks",
]
