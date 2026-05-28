"""Worker-agnostic scaffolding shared by the extraction workers.

Each worker is a `(raw_doc, prompt, output_model) -> Output | None` pipeline.
The pieces below are identical across workers and live here so they have
one implementation; only the worker-specific bits (prompt, schema, decoding
params, routing-table key) stay in each worker module.

This module owns the INV-2 boundary for the workers that compose it:
`_resolve_spans` assigns Citation `source_span` from `source_quote` against
the raw doc, and `_write_quarantine` captures the unmutated model output
on every failure path. Changing these is a sensitive-path edit by extension
of AGENTS.md §3 — apply the same Tier 2 discipline.

`run_single_shot_extraction` composes the helpers with the cache, the
Anthropic client, and the post-validation guardrail in a single function
that workers can call with their prompt + output model. The
`client_factory` parameter is the per-worker singleton getter (e.g.,
`s_filings._get_client`); when omitted, a fresh client is built per call,
which is fine for hermetic tests that inject `anthropic_client` directly
but defeats production cost-cap / circuit-breaker state accumulation.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ValidationError

from auto_research._io import atomic_write_text
from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import (
    QuarantineRecord,
    validate_or_quarantine,
)
from auto_research.telemetry import truncate_status_description as _truncate

_tracer = trace.get_tracer(__name__)


# Markdown-fence strip: handles both ```json\n{...}\n``` and the
# no-newline single-line form ```json{...}```. Captures the JSON body
# in group 1. Defensive only — prompts forbid fences.
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _quote_to_flex_regex(quote: str) -> str:
    r"""Convert `quote` to a regex pattern that treats any run of whitespace
    in `quote` as `\s+` — matches whitespace-equivalent occurrences in raw
    text without losing positional fidelity.

    LLMs collapse runs of whitespace ("We may offer\nshares" -> "We may
    offer shares") more reliably than they preserve them. Matching the
    quote shape-flexibly against the raw doc and using the *raw match*'s
    offsets as `source_span` preserves the stronger property: the span
    points at the same semantic region the model quoted, and we then
    snap `source_quote` to the raw substring so post-validation's
    `source_text[span] == source_quote` holds literally.
    """
    parts = re.split(r"\s+", quote.strip())
    if not parts or parts == [""]:
        return r"(?!x)x"  # never-matching pattern; treated as "not found"
    return r"\s+".join(re.escape(p) for p in parts)


def _resolve_spans(
    parsed: dict[str, Any], raw: str
) -> tuple[dict[str, Any], list[str]]:
    """Return (resolved_copy, problem_quotes). `parsed` is NOT mutated.

    Walks a deep copy of `parsed` and assigns `source_span` to every
    Citation-shaped dict by whitespace-flexible regex match against
    `raw`. When N Citation-shaped dicts share the same `source_quote`
    text and the raw doc has exactly N occurrences, they are paired
    in document order (first citation → first occurrence). This is the
    common case for entity mentions: a real 10-K names TSMC across
    Risk Factors, MD&A, and Properties, and the model emits one
    SupplierMention per textual occurrence.

    A quote is a "problem" (route to quarantine) when it is empty, not
    found in `raw`, has fewer occurrences than citations sharing it
    (the model fabricated extras), or has more occurrences than
    citations sharing it (we cannot honestly pick which occurrence
    each citation refers to).

    The original `parsed` is preserved so the upstream quarantine path
    can snapshot exactly what the model returned, not what the worker
    rewrote it to.
    """
    resolved = copy.deepcopy(parsed)
    problems: list[str] = []

    # First pass: collect every Citation-shaped dict, grouped by quote
    # text, in document order (json.loads preserves dict insertion
    # order and list ordering, so the traversal order matches the
    # order the LLM emitted the citations).
    quote_to_nodes: dict[str, list[dict[str, Any]]] = {}

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            if "source_quote" in node:
                quote = node["source_quote"]
                if not isinstance(quote, str) or not quote.strip():
                    problems.append(repr(quote))
                else:
                    quote_to_nodes.setdefault(quote, []).append(node)
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(resolved)

    # Second pass: for each unique quote, locate every occurrence in
    # raw and pair with the collected nodes by document order.
    for quote, nodes in quote_to_nodes.items():
        pattern = _quote_to_flex_regex(quote)
        matches = list(re.finditer(pattern, raw))
        if len(matches) == 0:
            problems.append(quote)
            continue
        if len(matches) < len(nodes):
            problems.append(
                f"INSUFFICIENT MATCHES ({len(matches)} matches for "
                f"{len(nodes)} citations): {quote}"
            )
            continue
        if len(matches) > len(nodes):
            problems.append(
                f"AMBIGUOUS ({len(matches)} matches for "
                f"{len(nodes)} citations): {quote}"
            )
            continue
        for node, match in zip(nodes, matches, strict=True):
            start, end = match.span()
            node["source_span"] = [start, end]
            # Snap source_quote to the actual raw substring so the
            # guardrail's `source_text[span] == quote` invariant holds
            # literally.
            node["source_quote"] = raw[start:end]

    return resolved, problems


def _write_quarantine(
    *,
    quarantine_root: Path,
    worker: str,
    prompt_version: str,
    doc_id: str,
    parsed: object,
    error: str,
) -> None:
    """Persist a QuarantineRecord snapshot.

    Captures `parsed` verbatim (wrapping non-dicts in `{"raw": ...}`) so
    reviewers see exactly what the model returned. The on-disk shape is
    the canonical `QuarantineRecord` JSON; downstream review tooling
    (`scripts/triage_quarantine.py`, when it lands) reads from here.
    """
    record = QuarantineRecord(
        doc_id=doc_id,
        worker=worker,
        prompt_version=prompt_version,
        output=parsed if isinstance(parsed, dict) else {"raw": parsed},
        error=error,
    )
    target = quarantine_root / worker / f"{doc_id}.json"
    atomic_write_text(target, record.model_dump_json(indent=2))


class ClientFactory(Protocol):
    """Per-worker singleton-or-fresh client getter.

    Each worker module owns a module-level `_CLIENT: ExtractionFn | None`
    and a `_get_client(anthropic_client)` function so the factory closure
    (with its `@cost_cap` and `@circuit_breaker` state) survives across
    calls. `run_single_shot_extraction` calls this factory once per
    invocation rather than building a new client itself — that's what
    keeps the production cost-cap counter accumulating across docs.
    """

    def __call__(
        self, anthropic_client: anthropic.Anthropic | None
    ) -> ExtractionFn: ...


def run_single_shot_extraction[OutputT: BaseModel](
    *,
    raw_doc: str,
    doc_id: str,
    worker: str,
    task: str,
    prompt: str,
    prompt_version: str,
    output_model: type[OutputT],
    max_tokens: int,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None = None,
    client_factory: ClientFactory | None = None,
    cache_write_handler: Callable[[str, dict[str, Any]], None] | None = None,
) -> OutputT | None:
    """One-shot LLM extraction with the shared scaffolding.

    Each worker passes its `(prompt, prompt_version, output_model, task,
    max_tokens)` and gets back a validated output (or `None` with a
    `QuarantineRecord` on the failure paths). `client_factory` is the
    per-worker singleton getter (`_get_client`); when omitted, a fresh
    client is built each call via `make_extraction_client` — acceptable
    only for tests that inject `anthropic_client` directly.

    `cache_write_handler`: when provided, the function calls
    `handler(cache_key, payload)` on a successful extraction INSTEAD of
    writing to the on-disk cache. This is the stage-and-commit hook
    used by multi-call composers (e.g., 10-K RAG) so a later-call
    failure doesn't leave earlier calls' per-field results half-
    persisted. Default `None` writes inline (the single-call workers'
    expected behavior). Cache HITS still return the cached output
    immediately and skip the handler — there is nothing to stage.

    Emits `extract.<worker>` OTel span with `extract.outcome` ∈
    `{cache_hit, persisted, staged, quarantined, error}`. Span status
    is set to ERROR on every failure path so alerts wired on OTel
    status surface guardrail failures (INV-2).
    """
    model_id = route_model(worker, task)
    schema_version: str = output_model.SCHEMA_VERSION  # type: ignore[attr-defined]
    decoding_params: dict[str, object] = {"max_tokens": max_tokens}
    key = content_cache.cache_key(
        raw_doc=raw_doc.encode(),
        prompt_version=prompt_version,
        schema_version=schema_version,
        model_id=model_id,
        decoding_params=decoding_params,
    )

    with _tracer.start_as_current_span(f"extract.{worker}") as span:
        span.set_attribute("extract.worker", worker)
        span.set_attribute("extract.doc_id", doc_id)

        cached = content_cache.read(cache_root, worker, key)
        if cached is not None:
            span.set_attribute("extract.outcome", "cache_hit")
            return output_model.model_validate(cached)

        if client_factory is not None:
            client = client_factory(anthropic_client)
        else:
            client = make_extraction_client(
                worker=worker, anthropic_client=anthropic_client
            )

        try:
            response = client(
                task=task,
                system_prompt=prompt,
                user_content=raw_doc,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            span.set_attribute("extract.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
            raise

        text = _strip_fence(
            "".join(b.text for b in response.content if b.type == "text").strip()
        )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=doc_id,
                parsed={"raw_text": text},
                error=f"json decode failed: {exc}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, _truncate(f"json decode failed: {exc}"))
            )
            return None

        parsed_snapshot = copy.deepcopy(parsed)
        resolved, problem_quotes = _resolve_spans(parsed, raw_doc)
        if problem_quotes:
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=doc_id,
                parsed=parsed_snapshot,
                error=f"source_quote(s) unresolvable in raw_doc: {problem_quotes!r}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(Status(StatusCode.ERROR, "source_quote(s) unresolvable"))
            return None

        try:
            output = output_model.model_validate(resolved)
        except ValidationError as exc:
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=doc_id,
                parsed=parsed_snapshot,
                error=f"schema validation failed: {exc}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, _truncate(f"schema validation failed: {exc}"))
            )
            return None

        validated = validate_or_quarantine(
            output,
            source_text=raw_doc,
            doc_id=doc_id,
            worker=worker,
            prompt_version=prompt_version,
            quarantine_root=quarantine_root,
            # Pass the pre-_resolve_spans snapshot so a downstream
            # CitationMismatch's QuarantineRecord shows what the model
            # actually returned rather than the worker's snapped quotes.
            original_output=parsed_snapshot,
        )
        if validated is None:
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, "post-validation guardrail failed")
            )
            return None

        payload = validated.model_dump(mode="json")
        if cache_write_handler is not None:
            cache_write_handler(key, payload)
            span.set_attribute("extract.outcome", "staged")
        else:
            content_cache.write(cache_root, worker, key, payload)
            span.set_attribute("extract.outcome", "persisted")
        return validated


__all__ = [
    "ClientFactory",
    "_quote_to_flex_regex",
    "_resolve_spans",
    "_strip_fence",
    "_write_quarantine",
    "run_single_shot_extraction",
]
