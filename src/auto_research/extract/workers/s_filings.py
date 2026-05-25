"""S-1 / S-3 extraction worker — first end-to-end validator (Issue #11).

Composes the W1 extraction primitives:

    prompt + client + cache + guardrails -> SFilingOutput | None

Flow:

1. Build the cache key from the full completion config (raw_doc bytes,
   prompt version, schema version, routed model, decoding params).
2. Look up `data/cache/extract/s_filings/<sha>.json`. Hit -> deserialize
   into `SFilingOutput`, return.
3. Miss -> invoke the Anthropic client (with reliability + caching from
   `make_extraction_client`), parse the JSON content block into a dict,
   resolve `source_span` against the raw doc by whitespace-flexible
   regex match (LLMs are unreliable at character counting; the worker is
   the source of span truth), construct `SFilingOutput`, validate via
   `validate_or_quarantine`.
4. On parse / span-resolution / validation failure: write a
   `QuarantineRecord` snapshot of the *unmutated* parsed dict and return
   None. Any failure mode that leaves the model's output unauditable is
   a bug — every quarantine path captures what the model actually said.

`cache_root` and `quarantine_root` are injected so tests can pass
`tmp_path` and stay hermetic. Production callers omit them and get the
package defaults. `anthropic_client` is injected the same way
`make_extraction_client` accepts it — production callers omit it and
share the module-level singleton (preserves cost-cap and circuit-breaker
state across calls per `client.py` discipline); tests pass a `MagicMock`
and get a fresh per-call client.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError

from auto_research._io import atomic_write_text
from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import (
    DEFAULT_QUARANTINE_ROOT,
    QuarantineRecord,
    validate_or_quarantine,
)
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.schemas import SFilingOutput
from auto_research.telemetry import truncate_status_description as _truncate

_WORKER = "s_filings"
_TASK = "dilution_event"  # matches SFilingOutput.dilution_event field name
_MAX_TOKENS = 4096
_DECODING_PARAMS: dict[str, object] = {"max_tokens": _MAX_TOKENS}
_tracer = trace.get_tracer(__name__)

# Module-level lazy client so per-worker cost_cap + circuit_breaker state
# accumulates across calls. Each call site that passes its own
# `anthropic_client` (test injection) gets a fresh per-call client and
# bypasses the singleton — fine because per-test state isolation is what
# tests want.
_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    """Return the production singleton, or a fresh client for test injection.

    The singleton path is the one whose @cost_cap counter and
    @circuit_breaker state must persist across calls; the injection path
    is exercised only by tests that don't care about those.
    """
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(
            worker=_WORKER, anthropic_client=anthropic_client
        )
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER)
    return _CLIENT


# Markdown-fence strip: handles both `\`\`\`json\n{...}\n\`\`\`` and
# the no-newline single-line form `\`\`\`json{...}\`\`\``. Captures the JSON
# body in group 1. Defensive only — the prompt forbids fences.
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _quote_to_flex_regex(quote: str) -> str:
    r"""Convert `quote` to a regex pattern that treats any run of whitespace
    in `quote` as `\s+` — matches whitespace-equivalent occurrences in
    raw text without losing positional fidelity.

    Required because the prompt asks the model to quote verbatim, but
    LLMs collapse runs of whitespace ("We may offer\nshares" -> "We may
    offer shares") more reliably than they preserve them. We match the
    quote shape-flexibly against the raw doc and use the *raw match*'s
    offsets as `source_span` — INV-2 's `source_text[span] == source_quote`
    no longer holds literally, but the stronger property holds: the span
    points at the same semantic region the model quoted.
    """
    parts = re.split(r"\s+", quote.strip())
    if not parts or parts == [""]:
        return r"(?!x)x"  # never-matching pattern; treated as "not found"
    return r"\s+".join(re.escape(p) for p in parts)


def _resolve_spans(
    parsed: dict[str, Any], raw: str
) -> tuple[dict[str, Any], list[str]]:
    """Return (resolved_copy, problem_quotes).

    Walks a deep copy of `parsed` and assigns `source_span` to every
    Citation-shaped dict by whitespace-flexible regex match against `raw`.
    A quote is a "problem" (route to quarantine) if it is empty, not found
    in `raw`, or appears more than once — ambiguous matches mean we
    cannot honestly assign one span over another.

    Crucially: `parsed` is NOT mutated. The QuarantineRecord upstream
    snapshots the *original* parsed dict so reviewers see exactly what
    the model returned, not what the worker rewrote it to.
    """
    resolved = copy.deepcopy(parsed)
    problems: list[str] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            if "source_quote" in node:
                quote = node["source_quote"]
                if not isinstance(quote, str) or not quote.strip():
                    problems.append(repr(quote))
                else:
                    pattern = _quote_to_flex_regex(quote)
                    matches = list(re.finditer(pattern, raw))
                    if len(matches) == 0:
                        problems.append(quote)
                    elif len(matches) > 1:
                        problems.append(
                            f"AMBIGUOUS ({len(matches)} matches): {quote}"
                        )
                    else:
                        start, end = matches[0].span()
                        node["source_span"] = [start, end]
                        # Snap source_quote to the actual raw substring so
                        # validate_citation_grounding's
                        # `source_text[span] == quote` invariant holds
                        # literally (not just shape-equivalently).
                        node["source_quote"] = raw[start:end]
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(resolved)
    return resolved, problems


def _write_quarantine(
    *,
    quarantine_root: Path,
    doc_id: str,
    parsed: object,
    error: str,
) -> None:
    record = QuarantineRecord(
        doc_id=doc_id,
        worker=_WORKER,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        output=parsed if isinstance(parsed, dict) else {"raw": parsed},
        error=error,
    )
    target = quarantine_root / _WORKER / f"{doc_id}.json"
    atomic_write_text(target, record.model_dump_json(indent=2))


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
    grounding check; the caller MUST treat None as "do not persist." The
    raw model output is always captured in a QuarantineRecord on the
    failure path.
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

    with _tracer.start_as_current_span("extract.s_filings") as span:
        span.set_attribute("extract.worker", _WORKER)
        span.set_attribute("extract.doc_id", doc_id)

        cached = content_cache.read(effective_cache_root, _WORKER, key)
        if cached is not None:
            span.set_attribute("extract.outcome", "cache_hit")
            return SFilingOutput.model_validate(cached)

        client = _get_client(anthropic_client)
        try:
            response = client(
                task=_TASK,
                system_prompt=S_FILINGS_DILUTION_PROMPT,
                user_content=raw_doc,
                max_tokens=_MAX_TOKENS,
            )
        except Exception as exc:
            # SDK / rate-limit / circuit-breaker / cost-cap raised. The
            # documented `extract.outcome` enum is {cache_hit, persisted,
            # quarantined}; emit a fourth value here so dashboards
            # filtering on outcome aren't missing the infra-failure rate.
            span.set_attribute("extract.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
            raise

        # Extract the text content. An empty content list or a non-text-block
        # response (e.g. refusal, max_tokens before any text) leaves `text` as
        # the empty string — treat as a parse failure so the model's response
        # shape is auditable rather than crashing the batch.
        text = _strip_fence(
            "".join(b.text for b in response.content if b.type == "text").strip()
        )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            _write_quarantine(
                quarantine_root=effective_quarantine_root,
                doc_id=doc_id,
                parsed={"raw_text": text},
                error=f"json decode failed: {exc}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, _truncate(f"json decode failed: {exc}"))
            )
            return None

        # Snapshot BEFORE _resolve_spans builds its mutated copy, so a quarantine
        # write on the next branch persists the model's actual output.
        parsed_snapshot = copy.deepcopy(parsed)

        resolved, problem_quotes = _resolve_spans(parsed, raw_doc)
        if problem_quotes:
            _write_quarantine(
                quarantine_root=effective_quarantine_root,
                doc_id=doc_id,
                parsed=parsed_snapshot,
                error=f"source_quote(s) unresolvable in raw_doc: {problem_quotes!r}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, "source_quote(s) unresolvable")
            )
            return None

        try:
            output = SFilingOutput.model_validate(resolved)
        except ValidationError as exc:
            _write_quarantine(
                quarantine_root=effective_quarantine_root,
                doc_id=doc_id,
                parsed=parsed_snapshot,
                error=f"schema validation failed: {exc}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, _truncate(f"schema validation failed: {exc}"))
            )
            return None

        # Defense in depth — even though spans were just computed by the worker
        # against `raw_doc`, run the validator: catches future worker bugs
        # (e.g., a refactor that changes _resolve_spans without re-running it)
        # before the bad output reaches the cache.
        validated = validate_or_quarantine(
            output,
            source_text=raw_doc,
            doc_id=doc_id,
            worker=_WORKER,
            prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
            quarantine_root=effective_quarantine_root,
        )
        if validated is None:
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, "post-validation guardrail failed")
            )
            return None

        content_cache.write(
            effective_cache_root, _WORKER, key, validated.model_dump(mode="json")
        )
        span.set_attribute("extract.outcome", "persisted")
        return validated


__all__ = ["extract_s_filing"]
