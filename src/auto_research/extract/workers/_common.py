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
that workers can call with their prompt + output model. The per-worker
extraction-client singleton lives here in `_CLIENTS` (keyed on worker
name) so the workers don't each own a module-level `_CLIENT` + getter —
that removed a footgun where a future worker forgetting to pass
`client_factory=_get_client` would silently build a fresh client per
call and defeat `@cost_cap` / `@circuit_breaker` state accumulation.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ValidationError

from auto_research._io import atomic_write_text
from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.client import (
    ExtractionFn,
    make_extraction_client,
)
from auto_research.extract.guardrails import (
    QuarantineRecord,
    validate_or_quarantine,
)
from auto_research.extract.openai_compat_client import (
    get_or_build_local_client,
)
from auto_research.telemetry import truncate_status_description as _truncate

_tracer = trace.get_tracer(__name__)


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


# Module-level per-worker extraction-client singleton table. Keyed on
# worker name so a single import-time data structure replaces the
# duplicated `_CLIENT: ExtractionFn | None = None` + `_get_client(...)`
# pattern that used to live in each worker module. Production calls
# reuse one client per worker so `@cost_cap` / `@circuit_breaker` state
# accumulates across docs; tests reset the dict per-test via a conftest
# fixture and bypass it entirely when injecting `anthropic_client`.
#
# The Anthropic singleton table lives here; the parallel OpenAI-compat
# singleton table is owned by `openai_compat_client._LOCAL_CLIENTS` so
# `(worker, local_model_id)` pairs key independently — a future routing
# flip from `local/qwen3.5:9b` to `local/qwen3.5:27b` builds a fresh
# client rather than reusing a stale entry.
_CLIENTS: dict[str, ExtractionFn] = {}


def _get_or_build_client(
    worker: str,
    task: str,
    anthropic_client: anthropic.Anthropic | None,
) -> ExtractionFn:
    """Return the right extraction client for `(worker, task)`.

    Dispatches on the routed model id: `local/...` ⇒ the OpenAI-compat
    HTTP wrapper (Ollama / vLLM / MLX-server); anything else ⇒ the
    Anthropic SDK wrapper. The dispatch is what makes the rest of the
    extraction pipeline provider-agnostic — workers call this helper
    and get back an `ExtractionFn`; whether the bytes leave the machine
    is decided by the routing table, not by per-worker code.

    `anthropic_client` is the test-injection escape hatch for the
    Anthropic path: when provided, each call builds a fresh wrapper
    around the duck-typed stub so per-test state stays isolated. When
    omitted (the production path), the worker-name-keyed singleton in
    `_CLIENTS` is created on first use and reused thereafter so
    `@cost_cap` and `@circuit_breaker` state accumulates across docs
    within a process. The local path's test-injection happens at the
    `make_openai_compat_extraction_client` level and is exercised by
    `tests/unit/test_extract_openai_compat_client.py`; production
    callers route through `get_or_build_local_client` here.

    Today no `_ROUTING` row resolves to a `local/*` model id; the
    dispatch infrastructure lands first per the cost-model doc §10.5
    Phase 1, and route flips ship per-worker as the eval suite
    validates the substitution.
    """
    model_id = route_model(worker, task)
    if model_id.startswith("local/"):
        # Test-injection on the local path is at the factory level
        # (`openai_client=` kwarg on `make_openai_compat_extraction_client`),
        # not threaded through this helper — keeps the Anthropic-only
        # `anthropic_client` parameter from leaking into the local
        # wrapper's signature. The `anthropic_client` arg is silently
        # ignored on the local branch; a worker that needs to inject a
        # local fake constructs the wrapper directly.
        return get_or_build_local_client(worker, model_id)

    if anthropic_client is not None:
        return make_extraction_client(
            worker=worker, anthropic_client=anthropic_client
        )
    if worker not in _CLIENTS:
        _CLIENTS[worker] = make_extraction_client(worker=worker)
    return _CLIENTS[worker]


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
    cache_write_handler: Callable[[str, dict[str, Any]], None] | None = None,
) -> OutputT | None:
    """One-shot LLM extraction with the shared scaffolding.

    Each worker passes its `(prompt, prompt_version, output_model, task,
    max_tokens)` and gets back a validated output (or `None` with a
    `QuarantineRecord` on the failure paths). The extraction client is
    pulled from the worker-keyed singleton in `_CLIENTS` (production)
    or built fresh per call when the caller injects `anthropic_client`
    (tests). Workers no longer pass their own `client_factory` — the
    consolidation here is what prevents a future worker from
    accidentally re-creating the singleton on every call and silently
    defeating cost-cap state accumulation.

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

        client = _get_or_build_client(worker, task, anthropic_client)

        try:
            parsed, usage = client(
                task=task,
                system_prompt=prompt,
                user_content=raw_doc,
                output_schema=output_model,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            span.set_attribute("extract.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
            raise

        # Local helper bundles the three-step quarantine ceremony: write
        # the QuarantineRecord, mark the OTel span outcome+status, and
        # return None. Keeps the four failure branches below visually
        # parallel and ensures any future change to the ceremony (an
        # added span attribute, a different status code) happens once.
        def _quarantine(
            *,
            parsed_payload: object,
            error: str,
            status_msg: str,
        ) -> None:
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=doc_id,
                parsed=parsed_payload,
                error=error,
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(Status(StatusCode.ERROR, _truncate(status_msg)))

        # The wrapper surfaces the structured payload directly: Anthropic
        # `tool_use.input` (forced via `tool_choice=record_extraction`)
        # and OpenAI-compat `response_format=json_schema` content both
        # parse to a `dict` on success, or `None` when the provider
        # returned no usable structured payload (refusal, mid-emission
        # truncation, content-filter strip). Either way we quarantine
        # rather than silently coerce a partial answer; the original
        # provider response shape never escapes the wrapper. The
        # diagnostic payload preserves the structural fingerprint the
        # pre-tuple code captured — `response_block_types` for
        # Anthropic (e.g., `["text"]` vs `["thinking", "text"]`,
        # distinguishing refusal from thinking-budget exhaustion);
        # `stop_reason` for the OpenAI-compat side (`content_filter`
        # vs `length` etc.). Operators triaging a `5% of S-1 docs
        # quarantine with the same error` debugging session need this.
        if parsed is None:
            diagnostic: dict[str, Any] = {
                "raw_response": "no structured payload",
            }
            block_types = usage.get("response_block_types")
            if block_types is not None:
                diagnostic["response_block_types"] = block_types
            stop_reason = usage.get("stop_reason")
            if stop_reason is not None:
                diagnostic["stop_reason"] = stop_reason
            _quarantine(
                parsed_payload=diagnostic,
                error="provider returned no structured payload (no tool_use / json_schema content)",
                status_msg="no structured payload",
            )
            return None
        if not isinstance(parsed, dict):
            # The `output_schema` branch of the Protocol returns
            # `dict | None`; a `str` here would signal a wrapper bug
            # returning the wrong arm of the union rather than a
            # model failure — surface it as quarantine with a precise
            # error so the bug is auditable.
            _quarantine(
                parsed_payload={"raw_response": repr(parsed)},
                error=f"wrapper returned non-dict structured payload: {type(parsed).__name__}",
                status_msg="non-dict structured payload",
            )
            return None

        # `_resolve_spans` deep-copies its input internally and only
        # mutates the copy; the original `parsed` is preserved for the
        # quarantine snapshot below without an extra copy here.
        resolved, problem_quotes = _resolve_spans(parsed, raw_doc)
        if problem_quotes:
            _quarantine(
                parsed_payload=parsed,
                error=f"source_quote(s) unresolvable in raw_doc: {problem_quotes!r}",
                status_msg="source_quote(s) unresolvable",
            )
            return None

        try:
            output = output_model.model_validate(resolved)
        except ValidationError as exc:
            _quarantine(
                parsed_payload=parsed,
                error=f"schema validation failed: {exc}",
                status_msg=f"schema validation failed: {exc}",
            )
            return None

        validated = validate_or_quarantine(
            output,
            source_text=raw_doc,
            doc_id=doc_id,
            worker=worker,
            prompt_version=prompt_version,
            quarantine_root=quarantine_root,
            # Pass the original tool_use.input so a downstream
            # CitationMismatch's QuarantineRecord shows what the model
            # actually returned rather than the worker's snapped quotes.
            # `_resolve_spans` only mutated its own deep copy, so
            # `parsed` is identical to the SDK's view of the response.
            original_output=parsed,
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


# --- Multi-call worker primitives -------------------------------------------
#
# Workers that issue more than one LLM call per document (10-K RAG per-field
# loop, transcript binary split, and any future N-way split) share two
# bits of orchestration on top of `run_single_shot_extraction`:
#
# 1. Stage per-call cache writes via `cache_write_handler=` and commit them
#    only after every per-call success AND every cross-call invariant
#    (e.g., identity-field agreement) has passed. Half-cached state on a
#    mid-loop failure would let re-runs hit stale entries from a prior
#    attempt — the staging-and-commit primitive below makes "all or
#    nothing" the default.
#
# 2. Check that identity columns the model emits on every call (cik,
#    accession_number, fiscal_period_end on 10-K; ticker, event_datetime
#    on transcripts) AGREE across calls. A model that hallucinates a
#    different value on one call would silently corrupt downstream
#    attribution if the first value were kept; the helper quarantines
#    instead and returns None so the caller drops the staged writes.


def commit_staged_cache_writes(
    *,
    cache_root: Path,
    worker: str,
    pending: list[tuple[str, dict[str, Any]]],
) -> None:
    """Commit a batch of staged cache writes after all cross-call checks pass.

    Called from a multi-call worker AFTER every per-call extraction
    returned a non-None result AND `check_identity_agreement` (or any
    other cross-call invariant) confirmed the calls agree. The worker
    accumulates entries by passing `cache_write_handler=pending.append`
    (or equivalent) to each `run_single_shot_extraction` call; only the
    final commit hits disk. On any failure path, the caller returns
    None without calling this, and re-runs see no half-cached state
    from this attempt.
    """
    for cache_key, payload in pending:
        content_cache.write(cache_root, worker, cache_key, payload)


def check_identity_agreement(
    *,
    identity_values: dict[str, list[Any]],
    quarantine_root: Path,
    worker: str,
    prompt_version: str,
    doc_id: str,
) -> dict[str, Any] | None:
    """Verify each identity field has exactly one unique value across calls.

    `identity_values` maps a field name to the list of values that field
    took across the N per-call extractions. On full agreement (every
    field has a single unique value), returns a dict mapping each
    field to the agreed value so the caller can construct the merged
    output without re-deriving identity. On disagreement on ANY field,
    writes a `{doc_id}#identity-disagreement` quarantine record naming
    the divergent field + per-call values and returns None — the
    caller MUST treat None as "do not commit staged cache writes;
    return None to your caller."

    Same discipline applies regardless of how many calls a worker
    issued: 5 for the 10-K RAG path, 2 for the transcript binary
    split, N for any future multi-call worker.
    """
    agreed: dict[str, Any] = {}
    for field_name, values in identity_values.items():
        unique_strs = {str(v) for v in values}
        if len(unique_strs) > 1:
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=f"{doc_id}#identity-disagreement",
                parsed={
                    "field": field_name,
                    "values_per_call": [str(v) for v in values],
                },
                error=(
                    "multi-call extraction disagrees on identity field "
                    f"`{field_name}`: {sorted(unique_strs)!r}"
                ),
            )
            return None
        agreed[field_name] = values[0]
    return agreed


__all__ = [
    "_get_or_build_client",
    "_quote_to_flex_regex",
    "_resolve_spans",
    "_write_quarantine",
    "check_identity_agreement",
    "commit_staged_cache_writes",
    "run_single_shot_extraction",
]
