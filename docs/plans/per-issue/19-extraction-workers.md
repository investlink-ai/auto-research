# Issue #19 — 10-K, transcript, 8-K worker bodies + prompts

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline)
> or superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.
> This plan is disposable — delete at PR merge per AI_WORKFLOW.md §1.5.

**Goal.** Implement three extraction workers (10-K, transcript, 8-K) and their
prompts, composing the W1+W2 primitives. 10-K supports a hybrid policy:
single-shot for narrative under `SINGLE_SHOT_TOKEN_CUTOFF`, contextual-RAG path
above; Item 8 financials always read from `ParentChunk.table_html` via a typed
Pydantic schema. Refactor `extract/workers/s_filings.py` to share the JSON /
span / quarantine scaffolding with the new workers.

**Architecture.** A new `extract/workers/_common.py` carries the
worker-agnostic scaffolding (`_strip_fence`, `_resolve_spans`,
`_write_quarantine`, an OTel-instrumented `run_single_shot_extraction` driver).
Each worker module composes the common driver with its prompt + output
model + decoding params. The 10-K worker adds two paths the others don't
need: (a) a RAG path that uses `chunking.parse_filing` + `hybrid_retrieve` +
`rerank` to gather top-5 parents per narrative field before the LLM call,
and (b) a structured Item 8 reader that maps `ParentChunk.table_html` into
a new `TenKFinancials` schema via a templated Anthropic call. `s_filings`
is refactored to use `_common.py` in the same PR so the four workers share
one scaffolding implementation.

**Tech stack.** Python 3.12, Pydantic v2 (frozen + `extra="forbid"`),
Anthropic SDK with prompt-caching, OpenTelemetry, pytest + MagicMock for
hermetic worker tests.

---

## Files touched

**Create.**
- `src/auto_research/extract/workers/_common.py` — shared scaffolding.
- `src/auto_research/extract/workers/ten_k.py` — 10-K worker.
- `src/auto_research/extract/workers/transcript.py` — transcript worker.
- `src/auto_research/extract/workers/eight_k.py` — 8-K worker.
- `src/auto_research/extract/prompts/ten_k_narrative.py` — 10-K narrative prompt.
- `src/auto_research/extract/prompts/ten_k_financials.py` — 10-K Item 8 financials prompt.
- `src/auto_research/extract/prompts/transcript.py` — transcript prompt.
- `src/auto_research/extract/prompts/eight_k.py` — 8-K prompt.
- `tests/unit/test_extract_worker_common.py` — `_common.py` unit tests.
- `tests/unit/test_extract_worker_ten_k.py` — 10-K worker tests (single-shot + RAG branch coverage + Item 8).
- `tests/unit/test_extract_worker_transcript.py` — transcript worker tests.
- `tests/unit/test_extract_worker_eight_k.py` — 8-K worker tests.

**Modify.**
- `src/auto_research/extract/schemas.py` — add `FinancialLineItem`,
  `TenKFinancials`, and `financials: TenKFinancials | None = None` on
  `TenKOutput`. Additive — no `SCHEMA_VERSION` bump (per schemas.py docstring
  "Adding a field to any output model is non-breaking").
- `src/auto_research/extract/workers/s_filings.py` — replace its private
  helpers with imports from `_common.py`. Tier-2 refactor: existing tests
  must continue to pass unchanged.
- `src/auto_research/_models.py` — add a routing row for `("ten_k", "financials")`
  → Haiku (templated table → JSON; pattern recognition per spec §7.3).
- `tests/unit/test_models.py` — add a one-line assertion for the new row.
- `tests/unit/test_extract_prompts.py` — extend existing assertions to cover
  the four new `*_PROMPT_VERSION` constants + `source_quote` clauses.
- `tests/unit/test_extract_worker_s_filings.py` — no change needed if the
  refactor preserves the public surface (`extract_s_filing`); verify after
  the refactor that all existing tests still pass.

**No-touch.**
- `extract/guardrails.py`, `extract/schemas.py` Citation/Claim base types,
  `extract/chunking/**`, `extract/rag_retrieval.py`, `extract/rerank.py`,
  `extract/chunking_contextual.py`. These are reused, not modified.

## Sensitive-path classification

This PR touches `extract/schemas.py` (Tier 2 — `[SENSITIVE]` per AGENTS.md §3)
and `extract/workers/s_filings.py` (Tier 2 by extension — citation-grounding
contract). Per AI_WORKFLOW.md §2: failing test first, full pytest run for
the touched modules, PR body cites test names.

The four new prompts are Tier 1 per AGENTS.md §6 (ordinary code; `make quick`
+ targeted tests). The new workers are Tier 2 in spirit (they own
citation-grounding routing for their outputs) — apply the same test-first
discipline as `s_filings`.

## Pre-existing constants the plan depends on

| Symbol | Where | Why this plan needs it |
|---|---|---|
| `SINGLE_SHOT_TOKEN_CUTOFF` | `extract.chunking._tokens` (= 100_000) | 10-K hybrid branch threshold (AC bullet 3). |
| `count_tokens` | `extract.chunking._tokens` | Decide which 10-K branch to take. |
| `parse_filing(html, metadata)` | `extract.chunking._entrypoint` | RAG-path chunking. |
| `EmbeddingAdapter` | `extract.embeddings` | RAG-path embed/query. |
| `hybrid_retrieve(...)` | `extract.rag_retrieval` | RAG-path retrieval. |
| `Qwen3Reranker`, `rerank(...)`, `RerankHit` | `extract.rerank` | top-20 → top-5. |
| `make_extraction_client`, `ExtractionFn` | `extract.client` | LLM call wrapper. |
| `route_model("ten_k", task)` etc. | `auto_research._models` | Tier routing. |
| `validate_or_quarantine` | `extract.guardrails` | Post-validation. |
| `content_cache.read/write/cache_key` | `extract.cache` | Idempotent cache. |
| `_truncate` | `auto_research.telemetry` | OTel status messages. |

---

## Phase A — Shared scaffolding (`_common.py`)

Goal: extract the four helpers from `s_filings.py` into a worker-agnostic
module, with the s_filings test suite as the regression net.

### Task A1: Add `_common.py` with `_strip_fence` and `_quote_to_flex_regex`

**Files:**
- Create: `src/auto_research/extract/workers/_common.py`
- Test: `tests/unit/test_extract_worker_common.py`

- [ ] **Step 1: Write the failing test** (`tests/unit/test_extract_worker_common.py`)

```python
"""Unit tests for the shared extraction-worker scaffolding."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.workers._common import (
    _quote_to_flex_regex,
    _resolve_spans,
    _strip_fence,
    _write_quarantine,
)


def test_strip_fence_removes_json_fence_with_newlines() -> None:
    assert _strip_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_strip_fence_removes_json_fence_no_newlines() -> None:
    assert _strip_fence("```{\"a\": 1}```") == '{"a": 1}'


def test_strip_fence_passthrough_when_no_fence() -> None:
    body = '{"a": 1}'
    assert _strip_fence(body) == body


def test_quote_to_flex_regex_collapses_whitespace() -> None:
    import re
    pattern = _quote_to_flex_regex("foo  bar")
    assert re.search(pattern, "foo\nbar") is not None


def test_quote_to_flex_regex_empty_quote_never_matches() -> None:
    import re
    pattern = _quote_to_flex_regex("")
    assert re.search(pattern, "any text") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_extract_worker_common.py -v`
Expected: ImportError (module not yet created).

- [ ] **Step 3: Create `_common.py` with the two helpers**

```python
"""Worker-agnostic scaffolding shared by the extraction workers.

Each worker is a `(raw_doc, prompt, output_model) -> Output | None` pipeline.
The four pieces below are identical across workers and live here so they
have one implementation; only the worker-specific bits (prompt, schema,
decoding params, routing-table key) stay in each worker module.

Note this module owns the INV-2 boundary for new workers: `_resolve_spans`
assigns Citation `source_span` from `source_quote` against the raw doc,
and `_write_quarantine` captures the unmutated model output on every
failure path. Changing these is a sensitive-path edit by extension
(AGENTS.md §3 reaches `extract/workers/s_filings.py`; this scaffold
factors out the same code so the same rules apply).
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from auto_research._io import atomic_write_text
from auto_research.extract.guardrails import QuarantineRecord

# Markdown-fence strip: handles both ```json\n{...}\n``` and the
# no-newline single-line form ```json{...}```. Captures the JSON body in
# group 1. Defensive only — prompts forbid fences.
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _quote_to_flex_regex(quote: str) -> str:
    r"""Convert `quote` to a regex pattern that treats any run of whitespace
    as `\s+` — matches whitespace-equivalent occurrences in raw text without
    losing positional fidelity. See `s_filings.py` original for the why.
    """
    parts = re.split(r"\s+", quote.strip())
    if not parts or parts == [""]:
        return r"(?!x)x"  # never-matching pattern; treated as "not found"
    return r"\s+".join(re.escape(p) for p in parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_extract_worker_common.py::test_strip_fence_removes_json_fence_with_newlines tests/unit/test_extract_worker_common.py::test_strip_fence_removes_json_fence_no_newlines tests/unit/test_extract_worker_common.py::test_strip_fence_passthrough_when_no_fence tests/unit/test_extract_worker_common.py::test_quote_to_flex_regex_collapses_whitespace tests/unit/test_extract_worker_common.py::test_quote_to_flex_regex_empty_quote_never_matches -v`
Expected: 5 passed.

### Task A2: Add `_resolve_spans` and `_write_quarantine`

**Files:**
- Modify: `src/auto_research/extract/workers/_common.py`
- Modify: `tests/unit/test_extract_worker_common.py`

- [ ] **Step 1: Append to the test file**

```python
def test_resolve_spans_finds_unique_quote() -> None:
    parsed = {
        "claim": {
            "citation": {"source_quote": "hello\nworld"},
            "confidence": 0.5,
        }
    }
    raw = "hello world is here"
    resolved, problems = _resolve_spans(parsed, raw)
    assert problems == []
    citation = resolved["claim"]["citation"]
    start, end = citation["source_span"]
    assert raw[start:end] == "hello world"
    assert citation["source_quote"] == "hello world"


def test_resolve_spans_flags_not_found() -> None:
    parsed = {"citation": {"source_quote": "missing-quote"}}
    raw = "different text entirely"
    _, problems = _resolve_spans(parsed, raw)
    assert problems == ["missing-quote"]


def test_resolve_spans_flags_ambiguous_with_count() -> None:
    parsed = {"citation": {"source_quote": "same"}}
    raw = "same same same"
    _, problems = _resolve_spans(parsed, raw)
    assert len(problems) == 1
    assert "AMBIGUOUS" in problems[0]
    assert "3 matches" in problems[0]


def test_resolve_spans_does_not_mutate_input() -> None:
    parsed = {"citation": {"source_quote": "hello"}}
    raw = "hello"
    snapshot = copy.deepcopy(parsed)
    _resolve_spans(parsed, raw)
    assert parsed == snapshot


def test_write_quarantine_writes_record(tmp_path: Path) -> None:
    _write_quarantine(
        quarantine_root=tmp_path / "q",
        worker="test_worker",
        prompt_version="v1",
        doc_id="doc-1",
        parsed={"raw": "thing"},
        error="bad json",
    )
    target = tmp_path / "q" / "test_worker" / "doc-1.json"
    assert target.exists()
    record = json.loads(target.read_text())
    assert record["worker"] == "test_worker"
    assert record["doc_id"] == "doc-1"
    assert record["error"] == "bad json"
    assert record["output"] == {"raw": "thing"}
```

Also add `import copy` and `from pathlib import Path` at the top if not
already present.

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/unit/test_extract_worker_common.py::test_resolve_spans_finds_unique_quote -v`
Expected: ImportError (`_resolve_spans` / `_write_quarantine` not defined).

- [ ] **Step 3: Append to `_common.py`**

```python
def _resolve_spans(
    parsed: dict[str, Any], raw: str
) -> tuple[dict[str, Any], list[str]]:
    """Return (resolved_copy, problem_quotes). `parsed` is NOT mutated.

    Walks a deep copy of `parsed` and assigns `source_span` to every
    Citation-shaped dict by whitespace-flexible regex match against `raw`.
    A quote is a "problem" (route to quarantine) if it is empty, not
    found in `raw`, or appears more than once.
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
                        # Snap the quote to the actual raw substring so
                        # post-validation's `source_text[span] == quote`
                        # holds literally.
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
    worker: str,
    prompt_version: str,
    doc_id: str,
    parsed: object,
    error: str,
) -> None:
    record = QuarantineRecord(
        doc_id=doc_id,
        worker=worker,
        prompt_version=prompt_version,
        output=parsed if isinstance(parsed, dict) else {"raw": parsed},
        error=error,
    )
    target = quarantine_root / worker / f"{doc_id}.json"
    atomic_write_text(target, record.model_dump_json(indent=2))


__all__ = [
    "_quote_to_flex_regex",
    "_resolve_spans",
    "_strip_fence",
    "_write_quarantine",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_extract_worker_common.py -v`
Expected: 10 passed.

### Task A3: Add `run_single_shot_extraction` driver

**Files:**
- Modify: `src/auto_research/extract/workers/_common.py`
- Modify: `tests/unit/test_extract_worker_common.py`

This is the orchestrator that workers call. It owns: cache key → cache read
→ LLM call → fence strip → JSON parse → span resolve → schema validate →
guardrail revalidate → cache write. The four worker-specific things it
takes as parameters are: prompt, prompt_version, output model class, task
key (for `route_model`).

- [ ] **Step 1: Write the test**

Append to `tests/unit/test_extract_worker_common.py`:

```python
from auto_research.extract.workers._common import run_single_shot_extraction
from auto_research.extract.schemas import EightKOutput
from auto_research.extract.enums import EventClassification


def _make_response(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text, citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=10, output_tokens=10,
            cache_creation=None, cache_creation_input_tokens=None,
            cache_read_input_tokens=None, inference_geo=None,
            server_tool_use=None, service_tier="standard",
        ),
    )


def _fake_client(text: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(text)
    return cast(anthropic.Anthropic, fake)


def test_run_single_shot_extraction_happy_path(tmp_path: Path) -> None:
    raw = "Material agreement signed with the Department of Defense."
    payload = {
        "cik": "0000000001",
        "accession_number": "0000000001-25-000001",
        "event_classification": "contract",
        "milestone_mentions": [],
        "dilution_language_flags": [],
    }
    client = _fake_client(json.dumps(payload))
    out = run_single_shot_extraction(
        raw_doc=raw,
        doc_id="doc-1",
        worker="eight_k",
        task="event_classification",
        prompt="…",
        prompt_version="v1",
        output_model=EightKOutput,
        max_tokens=512,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    assert out is not None
    assert out.event_classification == EventClassification.CONTRACT


def test_run_single_shot_extraction_quarantines_bad_json(tmp_path: Path) -> None:
    client = _fake_client("not json")
    out = run_single_shot_extraction(
        raw_doc="x",
        doc_id="doc-bad",
        worker="eight_k",
        task="event_classification",
        prompt="…",
        prompt_version="v1",
        output_model=EightKOutput,
        max_tokens=64,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "quar" / "eight_k" / "doc-bad.json").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_extract_worker_common.py::test_run_single_shot_extraction_happy_path -v`
Expected: ImportError.

- [ ] **Step 3: Append `run_single_shot_extraction` to `_common.py`**

```python
import copy as _copy
import json as _json
from typing import Protocol, TypeVar

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ValidationError

from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import validate_or_quarantine
from auto_research.telemetry import truncate_status_description as _truncate

_tracer = trace.get_tracer(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)


def run_single_shot_extraction(
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
    client_factory: "ClientFactory | None" = None,
) -> OutputT | None:
    """One-shot LLM extraction with the shared scaffolding.

    Each worker passes its `(prompt, prompt_version, output_model, task,
    max_tokens)` and gets back a validated output (or `None` with a
    QuarantineRecord on the failure paths). `client_factory` is the
    per-worker singleton getter (`_get_client`); a default factory builds
    a fresh client each call (acceptable when `anthropic_client` is
    injected by a test).
    """
    model_id = route_model(worker, task)
    schema_version = getattr(output_model, "SCHEMA_VERSION")
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
            parsed = _json.loads(text)
        except _json.JSONDecodeError as exc:
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

        parsed_snapshot = _copy.deepcopy(parsed)
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
        )
        if validated is None:
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, "post-validation guardrail failed")
            )
            return None

        content_cache.write(
            cache_root, worker, key, validated.model_dump(mode="json")
        )
        span.set_attribute("extract.outcome", "persisted")
        return validated


class ClientFactory(Protocol):
    def __call__(self, anthropic_client: anthropic.Anthropic | None) -> ExtractionFn: ...
```

Add `run_single_shot_extraction` and `ClientFactory` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_extract_worker_common.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/extract/workers/_common.py \
        tests/unit/test_extract_worker_common.py
git commit -m "feat(extract): shared scaffolding for extraction workers"
```

### Task A4: Refactor `s_filings.py` to use `_common.py`

**Files:**
- Modify: `src/auto_research/extract/workers/s_filings.py`

Touching a Tier-2 file. The regression contract is the existing test suite —
all `tests/unit/test_extract_worker_s_filings.py` tests must continue to pass
without modification.

- [ ] **Step 1: Re-run the existing s_filings test suite to capture the green baseline**

Run: `uv run pytest tests/unit/test_extract_worker_s_filings.py -v`
Expected: 17 passed (or whatever the current count is — record it).

- [ ] **Step 2: Replace the body of `s_filings.py`**

```python
"""S-1 / S-3 extraction worker.

Composes `_common.run_single_shot_extraction` with the dilution prompt and
`SFilingOutput` schema. The original `_strip_fence` / `_resolve_spans` /
`_write_quarantine` helpers now live in `_common.py` and are shared by
the 10-K / transcript / 8-K workers.
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
_TASK = "dilution_event"
_MAX_TOKENS = 4096

# Module-level lazy client so per-worker cost_cap + circuit_breaker state
# accumulates across calls. Each call site that passes its own
# `anthropic_client` (test injection) gets a fresh per-call client.
_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
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
    """Extract an SFilingOutput from a raw S-1/S-3 text."""
    return run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        task=_TASK,
        prompt=S_FILINGS_DILUTION_PROMPT,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        output_model=SFilingOutput,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT,
        quarantine_root=quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT,
        anthropic_client=anthropic_client,
        client_factory=_get_client,
    )


__all__ = ["extract_s_filing"]
```

- [ ] **Step 3: Re-run the s_filings test suite — every test must still pass**

Run: `uv run pytest tests/unit/test_extract_worker_s_filings.py -v`
Expected: same pass count as Step 1.

Critical assertion: `test_production_client_is_singleton`,
`test_injected_client_bypasses_singleton`, the OTel span tests, and the
quarantine-snapshot-preservation test must all still pass. If any fails,
the refactor changed observable behavior — revisit.

- [ ] **Step 4: Commit**

```bash
git add src/auto_research/extract/workers/s_filings.py
git commit -m "refactor(extract): s_filings worker uses _common.py scaffolding"
```

---

## Phase B — Schemas

### Task B1: Add `FinancialLineItem` and `TenKFinancials`, extend `TenKOutput`

**Files:**
- Modify: `src/auto_research/extract/schemas.py`
- Modify: `tests/unit/test_extract_schemas.py`

- [ ] **Step 1: Add a failing test in `tests/unit/test_extract_schemas.py`**

```python
def test_ten_k_financials_minimal_construction() -> None:
    from auto_research.extract.schemas import (
        Citation, FinancialLineItem, TenKFinancials,
    )
    line = FinancialLineItem(
        value_usd=1_000_000.0,
        citation=Citation(source_span=(0, 10), source_quote="0123456789"),
        confidence="high",
    )
    fin = TenKFinancials(revenue=line, gross_profit=None, operating_income=None,
                        net_income=None, total_assets=None, total_liabilities=None,
                        stockholders_equity=None, cash_from_operations=None,
                        cash_from_investing=None, cash_from_financing=None)
    assert fin.revenue == line
    assert fin.gross_profit is None


def test_ten_k_financials_confidence_is_categorical() -> None:
    """Per the LLM-confidence-is-categorical user-feedback memory: new
    confidence fields are Literal['high','medium','low'], not floats."""
    from auto_research.extract.schemas import FinancialLineItem, Citation
    with pytest.raises(ValueError):
        FinancialLineItem(
            value_usd=1.0,
            citation=Citation(source_span=(0, 5), source_quote="abcde"),
            confidence=0.9,  # type: ignore[arg-type]
        )


def test_ten_k_output_financials_field_defaults_none() -> None:
    """The new financials field is additive — existing TenKOutput
    construction without it must still validate (default None)."""
    from auto_research.extract.schemas import TenKOutput
    from datetime import date
    out = TenKOutput(
        cik="0000000001",
        accession_number="0000000001-25-000001",
        fiscal_period_end=date(2025, 12, 31),
        guidance_tone=_make_claim_for_test(),
        accrual_flags=[],
        supplier_mentions=[],
        customer_mentions=[],
        language_novelty_score=0.0,
        risk_factor_deltas=[],
    )
    assert out.financials is None
```

(Add a `_make_claim_for_test()` helper if not already in the file —
copy the pattern from existing tests.)

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_financials_minimal_construction -v`
Expected: ImportError or AttributeError.

- [ ] **Step 3: Edit `src/auto_research/extract/schemas.py`**

Add after `ForwardStatement`:

```python
class FinancialLineItem(BaseModel):
    """One line item from a 10-K Item 8 financial statement.

    `value_usd` is the dollar value reported on the filing (negatives
    allowed for losses / cash outflows). `confidence` is a categorical
    label — per `LLM confidence is categorical` user-feedback policy,
    float confidence is uncalibrated noise on table extraction.
    """

    model_config = _FROZEN_STRICT

    value_usd: float
    citation: Citation
    confidence: Literal["high", "medium", "low"]


class TenKFinancials(BaseModel):
    """10-K Item 8 financial statements extracted from `ParentChunk.table_html`.

    Each field is a `FinancialLineItem | None`; `None` means the line
    item wasn't reported in this filing (some firms don't break out
    cash flow categories the same way). Adding line items here is
    non-breaking; renaming or removing is a breaking change that
    requires a Feast migration.
    """

    model_config = _FROZEN_STRICT

    revenue: FinancialLineItem | None
    gross_profit: FinancialLineItem | None
    operating_income: FinancialLineItem | None
    net_income: FinancialLineItem | None
    total_assets: FinancialLineItem | None
    total_liabilities: FinancialLineItem | None
    stockholders_equity: FinancialLineItem | None
    cash_from_operations: FinancialLineItem | None
    cash_from_investing: FinancialLineItem | None
    cash_from_financing: FinancialLineItem | None
```

Add `from typing import Literal` to the imports (if not already present).

Extend `TenKOutput`:

```python
class TenKOutput(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    guidance_tone: Claim
    accrual_flags: list[Claim]
    supplier_mentions: list[SupplierMention]
    customer_mentions: list[CustomerMention]
    language_novelty_score: float
    risk_factor_deltas: list[RiskFactorDelta]
    financials: TenKFinancials | None = None  # Item 8; None if extraction skipped or only narrative path ran
```

Extend `__all__` to include `"FinancialLineItem", "TenKFinancials"`.

- [ ] **Step 4: Run schema tests**

Run: `uv run pytest tests/unit/test_extract_schemas.py -v`
Expected: all pass (including new ones).

- [ ] **Step 5: Run citation-grounding tests to confirm the walker handles the new types**

Run: `uv run pytest tests/unit/test_extract_guardrails.py -v`
Expected: all pass — the walker is generic over BaseModel and will
discover `Citation`s inside `FinancialLineItem` without changes.

- [ ] **Step 6: Commit**

```bash
git add src/auto_research/extract/schemas.py tests/unit/test_extract_schemas.py
git commit -m "feat(extract): TenKFinancials schema for Item 8 line items"
```

---

## Phase C — Prompts

Each prompt module is ~1 file, ~50 lines, mirroring `s_filings_dilution.py`.

### Task C1: 8-K prompt

**Files:**
- Create: `src/auto_research/extract/prompts/eight_k.py`
- Modify: `tests/unit/test_extract_prompts.py`

- [ ] **Step 1: Add failing test**

```python
def test_eight_k_prompt_exports() -> None:
    from auto_research.extract.prompts.eight_k import (
        EIGHT_K_PROMPT, EIGHT_K_PROMPT_VERSION,
    )
    import re
    assert re.fullmatch(r"v\d+", EIGHT_K_PROMPT_VERSION)
    assert "source_quote" in EIGHT_K_PROMPT
    assert "{source_text}" not in EIGHT_K_PROMPT
    # 8-K enums — at least one classification name must appear so
    # the model knows the closed-set values it should choose.
    assert "milestone" in EIGHT_K_PROMPT
    assert "partnership" in EIGHT_K_PROMPT
```

- [ ] **Step 2: Create `eight_k.py`**

```python
"""8-K event-extraction prompt.

Single-shot, ≤100K tokens. Instructions only — the filing text goes in
the user-content turn (so the cached prefix is the prompt + scheme
instructions, not the doc).
"""
from __future__ import annotations

EIGHT_K_PROMPT_VERSION = "v1"

EIGHT_K_PROMPT = """\
You are extracting structured event signals from an SEC 8-K current
report. The filing text will be supplied in the next user message.

Return a single JSON object matching the EightKOutput schema. Every claim
MUST include:
- source_quote: a verbatim substring of the filing text supporting the
  claim. Preserve original whitespace; do NOT collapse runs of whitespace
  or rewrite punctuation. The substring will be located in the filing by
  whitespace-flexible match; if no occurrence is found, OR if more than
  one occurrence is found, the claim will be rejected and the output
  quarantined. Choose quotes long and specific enough to be unique.

DO NOT include `source_span` — character offsets are computed in code.

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- event_classification: EXACTLY one of: "milestone", "partnership",
  "contract", "guidance_change", "leadership_change", "dilution", "other".
  Do not use "other" for events you are merely uncertain about —
  uncertainty belongs in the `confidence` of the supporting claims, not
  the classification. Use "other" only when the event genuinely falls
  outside the listed categories.
- milestone_mentions: list of Claims describing each FDA approval, clinical
  read-out, product launch, regulatory clearance, or technology milestone
  the filing announces (one Claim per distinct milestone).
- dilution_language_flags: list of Claims describing any language signalling
  potential dilution: shelf takedowns, at-the-market offerings, equity
  raises, convertible-debt issuance, warrant exercises.

A Claim is `{"citation": {"source_quote": "..."}, "confidence": 0.0-1.0}`.
No other fields are allowed inside a Claim or Citation.

If a field has no support in the filing, return an empty list — do not
fabricate citations.

Return ONLY the JSON object. Do not wrap it in markdown code fences. Do
not prepend or append any commentary.
"""

__all__ = ["EIGHT_K_PROMPT", "EIGHT_K_PROMPT_VERSION"]
```

- [ ] **Step 3: Run new test**

Run: `uv run pytest tests/unit/test_extract_prompts.py::test_eight_k_prompt_exports -v`
Expected: pass.

### Task C2: Transcript prompt

**Files:**
- Create: `src/auto_research/extract/prompts/transcript.py`
- Modify: `tests/unit/test_extract_prompts.py`

- [ ] **Step 1: Add failing test**

```python
def test_transcript_prompt_exports() -> None:
    from auto_research.extract.prompts.transcript import (
        TRANSCRIPT_PROMPT, TRANSCRIPT_PROMPT_VERSION,
    )
    import re
    assert re.fullmatch(r"v\d+", TRANSCRIPT_PROMPT_VERSION)
    assert "source_quote" in TRANSCRIPT_PROMPT
    assert "{source_text}" not in TRANSCRIPT_PROMPT
    # Transcript prompt must call out the prepared-remarks vs Q&A split
    # so the model produces both fields.
    assert "prepared_remarks_tone" in TRANSCRIPT_PROMPT
    assert "q_and_a_evasiveness" in TRANSCRIPT_PROMPT
    assert "forward_statements" in TRANSCRIPT_PROMPT
```

- [ ] **Step 2: Create `transcript.py`**

```python
"""Earnings-transcript extraction prompt.

Single-shot. Transcripts are short (1-3 hours of speech ≈ 25-60K
tokens). Instructions only; transcript text goes in user content.
"""
from __future__ import annotations

TRANSCRIPT_PROMPT_VERSION = "v1"

TRANSCRIPT_PROMPT = """\
You are extracting language signals from an earnings call transcript. The
transcript text will be supplied in the next user message.

Return a single JSON object matching the TranscriptOutput schema. Every
claim MUST include:
- source_quote: a verbatim substring of the transcript supporting the
  claim. Preserve original whitespace; do NOT collapse runs of whitespace.
  The substring will be located by whitespace-flexible match; if no
  occurrence is found, OR if more than one occurrence is found, the
  claim is rejected and the output quarantined. Choose quotes long and
  specific enough to be unique in the transcript.

DO NOT include `source_span` — character offsets are computed in code.

Fields to populate:
- ticker: the issuing company's stock ticker (uppercase).
- event_datetime: the earnings call's start time in ISO-8601 format
  (e.g., "2026-01-30T17:00:00-05:00"). If the transcript does not give
  a precise time, use 17:00 in the company's headquarters timezone.
- prepared_remarks_tone: a single Claim describing the tone of the
  prepared remarks (e.g., "cautious bullish on FY26 demand"). Confidence
  in [0, 1].
- q_and_a_evasiveness: a single Claim describing how evasive management
  was in the Q&A — i.e., whether they answered analyst questions
  directly or deflected. Confidence in [0, 1].
- forward_statements: list of ForwardStatement objects, each describing
  a forward-looking claim management made (e.g., "expect FY26 revenue
  growth above 30%"), with:
  - statement_text: the paraphrased forward statement.
  - citation: {source_quote: "..."}.
  - mentioned_entities: list of tickers or company names referenced.
  - horizon: phrase describing the time horizon (e.g., "next quarter",
    "FY26", "long-term", "by end of 2026").

A Claim is `{"citation": {"source_quote": "..."}, "confidence": 0.0-1.0}`.
A ForwardStatement is `{"statement_text": "...", "citation":
{"source_quote": "..."}, "mentioned_entities": [...], "horizon": "..."}`.

If a field has no support in the transcript, return an empty list. Do
not fabricate.

Return ONLY the JSON object. No markdown code fences. No commentary.
"""

__all__ = ["TRANSCRIPT_PROMPT", "TRANSCRIPT_PROMPT_VERSION"]
```

- [ ] **Step 3: Run new test — pass.**

### Task C3: 10-K narrative prompt

**Files:**
- Create: `src/auto_research/extract/prompts/ten_k_narrative.py`
- Modify: `tests/unit/test_extract_prompts.py`

- [ ] **Step 1: Add failing test**

```python
def test_ten_k_narrative_prompt_exports() -> None:
    from auto_research.extract.prompts.ten_k_narrative import (
        TEN_K_NARRATIVE_PROMPT, TEN_K_NARRATIVE_PROMPT_VERSION,
    )
    import re
    assert re.fullmatch(r"v\d+", TEN_K_NARRATIVE_PROMPT_VERSION)
    assert "source_quote" in TEN_K_NARRATIVE_PROMPT
    assert "{source_text}" not in TEN_K_NARRATIVE_PROMPT
    # Must cover all narrative TenKOutput fields except financials.
    for field in [
        "guidance_tone", "accrual_flags", "supplier_mentions",
        "customer_mentions", "risk_factor_deltas",
    ]:
        assert field in TEN_K_NARRATIVE_PROMPT, f"missing instruction for {field}"
```

- [ ] **Step 2: Create `ten_k_narrative.py`**

```python
"""10-K narrative-extraction prompt.

Drives the single-shot path (`token_count < SINGLE_SHOT_TOKEN_CUTOFF`)
AND the RAG path. In the RAG branch the worker stuffs the top-5
reranked parents per field into the user-content turn; the prompt
itself does not change between branches.

The prompt covers ONLY narrative TenKOutput fields. `financials` (Item
8) is extracted by a separate worker path from `ParentChunk.table_html`
and has its own prompt + schema.

`language_novelty_score` is NOT in the prompt — it's computed downstream
from the supplier/customer/risk-factor text vs the prior year's
extraction.
"""
from __future__ import annotations

TEN_K_NARRATIVE_PROMPT_VERSION = "v1"

TEN_K_NARRATIVE_PROMPT = """\
You are extracting narrative signals from an SEC 10-K annual report.
The 10-K text (or, in the RAG branch, the retrieved top passages) will
be supplied in the next user message. Focus on Items 1A (Risk Factors),
7 (MD&A), and 7A (Market Risk) — those sections dominate the
language-signal value for downstream signals.

Return a single JSON object matching the TenKOutput schema's narrative
fields (excluding `financials` and `language_novelty_score`, which are
handled separately). Every claim MUST include:
- source_quote: a verbatim substring of the supplied text supporting
  the claim. Preserve original whitespace; do NOT collapse runs of
  whitespace. The substring will be located by whitespace-flexible
  match; if no occurrence is found, OR if more than one occurrence is
  found, the claim is rejected and the output quarantined. Choose
  quotes long and specific enough to be unique in the supplied text.

DO NOT include `source_span` — character offsets are computed in code.
DO NOT populate `financials` or `language_novelty_score`.

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- fiscal_period_end: the period-end date in ISO format (YYYY-MM-DD).
- guidance_tone: a single Claim describing the tone of forward-looking
  language in MD&A (e.g., "cautious; gross-margin headwinds called out
  twice"). Confidence in [0, 1].
- accrual_flags: list of Claims flagging accrual-quality concerns —
  large unbilled receivables, deferred revenue swings, capitalized R&D
  growing faster than revenue, restructuring-charge resets.
- supplier_mentions: list of SupplierMentions naming specific named
  suppliers (e.g., TSMC, Foxconn, Samsung). Each must include:
  - mention_text: the verbatim name as it appears.
  - citation: {source_quote: "..."}.
  Do NOT fabricate or guess `resolved_ticker` / `resolver_confidence` /
  `resolver_reasoning` — leave them null. A separate resolver step runs
  later.
- customer_mentions: list of CustomerMentions (same shape as
  SupplierMentions). Include named customers — typically hyperscaler /
  enterprise customers explicitly called out (NVDA, MSFT, GOOGL, AMZN,
  META) — not vague references ("certain large customers").
- risk_factor_deltas: list of RiskFactorDeltas, each:
  - change_type: EXACTLY one of "added", "removed", "modified" (vs the
    prior year's 10-K).
  - text: the new (or removed/modified) risk-factor language.
  - citation: {source_quote: "..."} anchoring the text in this filing.
  When the prior year is not available, treat all Item 1A items as
  "added".

A Claim is `{"citation": {"source_quote": "..."}, "confidence": 0.0-1.0}`.
A SupplierMention / CustomerMention is `{"mention_text": "...",
"citation": {"source_quote": "..."}, "resolved_ticker": null,
"resolver_confidence": null, "resolver_reasoning": null}`.
A RiskFactorDelta is `{"change_type": "...", "text": "...", "citation":
{"source_quote": "..."}}`.

If a field has no support in the supplied text, return an empty list.
Do not fabricate.

Return ONLY the JSON object. No markdown fences. No commentary.
"""

__all__ = ["TEN_K_NARRATIVE_PROMPT", "TEN_K_NARRATIVE_PROMPT_VERSION"]
```

- [ ] **Step 3: Run new test — pass.**

### Task C4: 10-K financials (Item 8) prompt

**Files:**
- Create: `src/auto_research/extract/prompts/ten_k_financials.py`
- Modify: `tests/unit/test_extract_prompts.py`

- [ ] **Step 1: Add failing test**

```python
def test_ten_k_financials_prompt_exports() -> None:
    from auto_research.extract.prompts.ten_k_financials import (
        TEN_K_FINANCIALS_PROMPT, TEN_K_FINANCIALS_PROMPT_VERSION,
    )
    import re
    assert re.fullmatch(r"v\d+", TEN_K_FINANCIALS_PROMPT_VERSION)
    assert "source_quote" in TEN_K_FINANCIALS_PROMPT
    assert "high" in TEN_K_FINANCIALS_PROMPT
    assert "value_usd" in TEN_K_FINANCIALS_PROMPT
```

- [ ] **Step 2: Create `ten_k_financials.py`**

```python
"""10-K Item 8 financial-statement extraction prompt.

Reads structured line items from a `ParentChunk.table_html` snippet
(the raw outer table HTML). The user-content turn carries the table
HTML — small enough that single-shot is appropriate even for the
RAG branch of the 10-K worker.

Categorical confidence (`high`/`medium`/`low`) per user-feedback
policy: float confidence on table-cell extraction is uncalibrated
noise. Categorical lets a downstream consumer threshold cleanly
(e.g., only use `high`-confidence rows for financial signals).
"""
from __future__ import annotations

TEN_K_FINANCIALS_PROMPT_VERSION = "v1"

TEN_K_FINANCIALS_PROMPT = """\
You are extracting line items from a 10-K Item 8 financial statement
table. The table HTML will be supplied in the next user message.

Return a single JSON object matching the TenKFinancials schema. Each
field is either a FinancialLineItem object or null when the line item
isn't reported in this table. Every FinancialLineItem MUST include:
- value_usd: the dollar value as a number (negatives for losses or
  cash outflows). If the table reports values in thousands or millions,
  scale to dollars in your output (e.g., a table showing "Revenue:
  1,234 (in millions)" → value_usd: 1234000000).
- citation: {source_quote: "..."} — a verbatim substring of the table's
  text (e.g., the cell value plus row label, like
  "Total revenue $1,234"). Preserve whitespace and punctuation.
- confidence: EXACTLY one of "high", "medium", or "low". Use "high"
  when the line label is unambiguous and the value is clearly labelled;
  "medium" when the label is paraphrased or the unit (thousands /
  millions) requires inference; "low" when the cell is at the edge of
  the table or could plausibly refer to a different line.

Line items to populate (return null when not present in the table):
- revenue: total revenue / net revenue / total net sales.
- gross_profit: revenue minus cost of revenue.
- operating_income: operating income / income from operations.
- net_income: net income / net earnings.
- total_assets: total assets at period end.
- total_liabilities: total liabilities at period end.
- stockholders_equity: total stockholders' equity / total equity.
- cash_from_operations: cash provided by (used in) operating activities.
- cash_from_investing: cash provided by (used in) investing activities.
- cash_from_financing: cash provided by (used in) financing activities.

When multiple periods are reported (e.g., current year and prior year
columns), extract ONLY the most recent fiscal period. The current
fiscal period is typically the leftmost or rightmost data column —
use the column header dates to choose.

Return ONLY the JSON object. No markdown fences. No commentary.
"""

__all__ = ["TEN_K_FINANCIALS_PROMPT", "TEN_K_FINANCIALS_PROMPT_VERSION"]
```

- [ ] **Step 3: Run new test — pass.**

### Task C5: Routing-table row for `("ten_k", "financials")`

**Files:**
- Modify: `src/auto_research/_models.py`
- Modify: `tests/unit/test_models.py`

- [ ] **Step 1: Add failing test in `tests/unit/test_models.py`**

```python
def test_routes_ten_k_financials_to_haiku() -> None:
    from auto_research._models import route_model
    # Item 8 table-to-JSON is templated, high-volume pattern recognition
    # per spec §7.3 — Haiku, not Sonnet.
    assert route_model("ten_k", "financials") == "claude-haiku-4-5"
```

- [ ] **Step 2: Add the row to `_ROUTING` in `_models.py`**

```python
    ("ten_k", "financials"): _HAIKU,
```

- [ ] **Step 3: Run the new test — pass.**

Then run the full models test file: `uv run pytest tests/unit/test_models.py -v`. Expected: all pass.

### Task C6: Commit Phase C

- [ ] **Step 1: Commit**

```bash
git add src/auto_research/extract/prompts/eight_k.py \
        src/auto_research/extract/prompts/transcript.py \
        src/auto_research/extract/prompts/ten_k_narrative.py \
        src/auto_research/extract/prompts/ten_k_financials.py \
        src/auto_research/_models.py \
        tests/unit/test_extract_prompts.py \
        tests/unit/test_models.py
git commit -m "feat(extract): prompts for 10-K, transcript, 8-K"
```

---

## Phase D — 8-K and Transcript workers

These are the easy two: both single-shot only, no RAG branch, no
structured-table reader.

### Task D1: 8-K worker

**Files:**
- Create: `src/auto_research/extract/workers/eight_k.py`
- Create: `tests/unit/test_extract_worker_eight_k.py`

- [ ] **Step 1: Write failing tests**

```python
"""Unit tests for the 8-K worker."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.workers.eight_k import extract_eight_k


_SAMPLE_8K = (
    "On January 15, 2026, the company entered into a Material Definitive "
    "Agreement with the Department of Defense for delivery of optical "
    "interconnect systems valued at $42 million."
)


def _make_response(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text, citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=10, output_tokens=10,
            cache_creation=None, cache_creation_input_tokens=None,
            cache_read_input_tokens=None, inference_geo=None,
            server_tool_use=None, service_tier="standard",
        ),
    )


def _fake_client(text: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(text)
    return cast(anthropic.Anthropic, fake)


def _valid_output() -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000007",
        "event_classification": "contract",
        "milestone_mentions": [],
        "dilution_language_flags": [],
    }


def test_extract_eight_k_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client(json.dumps(_valid_output()))
    out = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.event_classification.value == "contract"


def test_extract_eight_k_cache_hit_skips_llm(tmp_path: Path) -> None:
    client = _fake_client(json.dumps(_valid_output()))
    extract_eight_k(raw_doc=_SAMPLE_8K, doc_id="8k-001", cache_root=tmp_path,
                    anthropic_client=client)
    extract_eight_k(raw_doc=_SAMPLE_8K, doc_id="8k-001", cache_root=tmp_path,
                    anthropic_client=client)
    assert client.messages.create.call_count == 1


def test_extract_eight_k_quarantines_hallucinated_quote(tmp_path: Path) -> None:
    bad = _valid_output()
    bad["milestone_mentions"] = [{
        "citation": {"source_quote": "not in the filing"},
        "confidence": 0.9,
    }]
    client = _fake_client(json.dumps(bad))
    out = extract_eight_k(
        raw_doc=_SAMPLE_8K, doc_id="8k-bad",
        cache_root=tmp_path, quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "q" / "eight_k" / "8k-bad.json").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_extract_worker_eight_k.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `eight_k.py`**

```python
"""8-K extraction worker — single-shot."""
from __future__ import annotations

from pathlib import Path

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.eight_k import (
    EIGHT_K_PROMPT, EIGHT_K_PROMPT_VERSION,
)
from auto_research.extract.schemas import EightKOutput
from auto_research.extract.workers._common import run_single_shot_extraction

_WORKER = "eight_k"
_TASK = "event_classification"  # matches EightKOutput.event_classification
_MAX_TOKENS = 4096

_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(
            worker=_WORKER, anthropic_client=anthropic_client
        )
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER)
    return _CLIENT


def extract_eight_k(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> EightKOutput | None:
    return run_single_shot_extraction(
        raw_doc=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        task=_TASK,
        prompt=EIGHT_K_PROMPT,
        prompt_version=EIGHT_K_PROMPT_VERSION,
        output_model=EightKOutput,
        max_tokens=_MAX_TOKENS,
        cache_root=cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT,
        quarantine_root=quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT,
        anthropic_client=anthropic_client,
        client_factory=_get_client,
    )


__all__ = ["extract_eight_k"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_extract_worker_eight_k.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/extract/workers/eight_k.py \
        tests/unit/test_extract_worker_eight_k.py
git commit -m "feat(extract): 8-K extraction worker"
```

### Task D2: Transcript worker

**Files:**
- Create: `src/auto_research/extract/workers/transcript.py`
- Create: `tests/unit/test_extract_worker_transcript.py`

Mirrors Task D1 with `TranscriptOutput`. The valid-output fixture must
include `ticker`, `event_datetime` (ISO string), `prepared_remarks_tone`,
`q_and_a_evasiveness`, and an empty `forward_statements`. The
routing-table key is `prepared_remarks_tone` (Haiku — cheap default).

- [ ] **Step 1: Write failing tests** (analogous to D1, with TranscriptOutput shape)

```python
def _valid_output() -> dict[str, Any]:
    return {
        "ticker": "ACME",
        "event_datetime": "2026-01-30T17:00:00-05:00",
        "prepared_remarks_tone": {
            "citation": {"source_quote": "We had a strong quarter"},
            "confidence": 0.7,
        },
        "q_and_a_evasiveness": {
            "citation": {"source_quote": "we can't comment on that"},
            "confidence": 0.5,
        },
        "forward_statements": [],
    }


_SAMPLE_TRANSCRIPT = (
    "Operator: Welcome. CFO: We had a strong quarter with revenue up 30%. "
    "Q&A — Analyst: How about FY26 margins? CFO: we can't comment on that."
)
```

Then three tests parallel to D1: happy path, cache hit, quarantine on
hallucinated quote.

- [ ] **Step 2: Run tests to verify failure.**

- [ ] **Step 3: Create `transcript.py`** (same shape as `eight_k.py`,
swapping prompt, schema, worker name = `"transcript"`, task =
`"prepared_remarks_tone"`).

- [ ] **Step 4: Run tests to verify they pass.**

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/extract/workers/transcript.py \
        tests/unit/test_extract_worker_transcript.py
git commit -m "feat(extract): transcript extraction worker"
```

---

## Phase E — 10-K worker

The headline complexity of the issue. Three paths in one entry point:

1. **Narrative single-shot** (`count_tokens(raw_doc) < SINGLE_SHOT_TOKEN_CUTOFF`,
   AND no chunkset provided): same shape as 8-K/transcript via
   `run_single_shot_extraction` with `ten_k_narrative` prompt.
2. **Narrative RAG** (`count_tokens(raw_doc) ≥ SINGLE_SHOT_TOKEN_CUTOFF`):
   parse → embed → hybrid retrieve + rerank per narrative field → assemble
   per-field user content → LLM call per field → assemble into
   TenKOutput. The full RAG composition is heavy; per the executing-plans
   discipline of "minimum to make the test pass", land a thin RAG path
   that exercises the branch (count_tokens check + chunkset materialization)
   but is parameterized by an injected `rag_extraction_fn` to keep tests
   hermetic — the real composition is wired through `chunking.parse_filing`
   + `EmbeddingAdapter` at the call site.
3. **Item 8 financials** (always, if a chunkset is provided AND the
   chunkset contains parents with `table_html is not None`): for each
   table parent, call the LLM with `ten_k_financials` prompt + `table_html`
   as user content; keep the first successful result (most 10-Ks have one
   consolidated financials table per statement). The Item 8 path is
   independent of the narrative path's single-shot-vs-RAG split.

### Task E1: 10-K narrative single-shot branch

**Files:**
- Create: `src/auto_research/extract/workers/ten_k.py`
- Create: `tests/unit/test_extract_worker_ten_k.py`

- [ ] **Step 1: Write a failing test for the single-shot path** (small raw doc)

```python
"""Unit tests for the 10-K worker."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.workers.ten_k import extract_ten_k


_SAMPLE_10K = (
    "Item 1A. Risk Factors. Our supply chain depends on TSMC.\n"
    "Item 7. MD&A. We expect cautious growth in fiscal 2026.\n"
)


def _make_response(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text, citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=10, output_tokens=10,
            cache_creation=None, cache_creation_input_tokens=None,
            cache_read_input_tokens=None, inference_geo=None,
            server_tool_use=None, service_tier="standard",
        ),
    )


def _fake_client(text: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(text)
    return cast(anthropic.Anthropic, fake)


def _valid_narrative() -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
        "guidance_tone": {
            "citation": {"source_quote": "cautious growth"},
            "confidence": 0.7,
        },
        "accrual_flags": [],
        "supplier_mentions": [{
            "mention_text": "TSMC",
            "citation": {"source_quote": "TSMC"},
            "resolved_ticker": None,
            "resolver_confidence": None,
            "resolver_reasoning": None,
        }],
        "customer_mentions": [],
        "language_novelty_score": 0.0,
        "risk_factor_deltas": [],
    }


def test_ten_k_single_shot_branch(tmp_path: Path) -> None:
    """Doc under SINGLE_SHOT_TOKEN_CUTOFF and no chunkset → single-shot
    narrative path (one LLM call, validated TenKOutput, no financials)."""
    client = _fake_client(json.dumps(_valid_narrative()))
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.fiscal_period_end == date(2025, 12, 31)
    assert out.financials is None
    assert len(out.supplier_mentions) == 1
    assert out.supplier_mentions[0].mention_text == "TSMC"
    # Single-shot = one Anthropic call.
    assert client.messages.create.call_count == 1
```

- [ ] **Step 2: Run failing test.** Expected: ImportError.

- [ ] **Step 3: Create `ten_k.py` with the single-shot branch**

```python
"""10-K extraction worker — single-shot narrative + RAG narrative + Item 8 financials.

Three paths, one entry point:

- `count_tokens(raw_doc) < SINGLE_SHOT_TOKEN_CUTOFF` and no chunkset
  supplied: single-shot via `run_single_shot_extraction` with the
  narrative prompt; `financials` is None on output.
- Otherwise (long doc): caller supplies a `ChunkSet` and a RAG
  extraction callable (see `extract_ten_k_rag`); the worker runs the
  RAG path for narrative fields.
- If `chunkset` is supplied AND has table parents: extract Item 8
  financials from the first table parent (or skip if none), regardless
  of which narrative path ran.

The chunking + embedding + retrieval steps are NOT executed inside the
worker — they're caller-owned (the backfill orchestrator in #22 will
materialize the chunkset and embedding adapter once per doc). Injecting
the chunkset keeps this worker testable without the full RAG stack.
"""
from __future__ import annotations

from pathlib import Path

import anthropic

from auto_research.extract import cache as content_cache
from auto_research.extract.chunking import (
    SINGLE_SHOT_TOKEN_CUTOFF, ChunkSet, count_tokens,
)
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.guardrails import DEFAULT_QUARANTINE_ROOT
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT, TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.schemas import TenKOutput
from auto_research.extract.workers._common import run_single_shot_extraction

_WORKER = "ten_k"
_NARRATIVE_TASK = "guidance_tone"  # default route key for the narrative single-shot path
_FINANCIALS_TASK = "financials"
_NARRATIVE_MAX_TOKENS = 8192
_FINANCIALS_MAX_TOKENS = 4096

_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    global _CLIENT
    if anthropic_client is not None:
        return make_extraction_client(
            worker=_WORKER, anthropic_client=anthropic_client
        )
    if _CLIENT is None:
        _CLIENT = make_extraction_client(worker=_WORKER)
    return _CLIENT


def extract_ten_k(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
    chunkset: ChunkSet | None = None,
) -> TenKOutput | None:
    """Extract a TenKOutput. Single-shot for short docs (no chunkset),
    RAG-pathway otherwise. Item 8 reads from chunkset table parents.
    """
    cache_root_resolved = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    quarantine_root_resolved = (
        quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT
    )

    if chunkset is None and count_tokens(raw_doc) < SINGLE_SHOT_TOKEN_CUTOFF:
        return run_single_shot_extraction(
            raw_doc=raw_doc,
            doc_id=doc_id,
            worker=_WORKER,
            task=_NARRATIVE_TASK,
            prompt=TEN_K_NARRATIVE_PROMPT,
            prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
            output_model=TenKOutput,
            max_tokens=_NARRATIVE_MAX_TOKENS,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
            client_factory=_get_client,
        )

    # RAG path / Item 8 path landed in later tasks.
    raise NotImplementedError("RAG / Item 8 paths land in Tasks E2-E3.")


__all__ = ["extract_ten_k"]
```

- [ ] **Step 4: Run test — pass.**

### Task E2: 10-K Item 8 financials path

**Files:**
- Modify: `src/auto_research/extract/workers/ten_k.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`

- [ ] **Step 1: Add failing test** — chunkset with one table parent → worker
calls LLM twice (once for narrative on a small raw_doc, once for the
table), output has `financials` populated.

```python
from auto_research.extract.chunking import (
    ChildChunk, ChunkMetadata, ChunkSet, ParentChunk,
)


def _chunkset_with_table(raw: str, table_html: str) -> ChunkSet:
    meta = ChunkMetadata(
        ticker="ACME", filing_date=date(2026, 1, 30),
        fiscal_period="FY2025", doc_type="10-K", doc_id="10k-table-001",
    )
    parent = ParentChunk(
        text=table_html, section_name="item_8",
        char_span=(0, len(raw)), token_count=10,
        table_html=table_html, metadata=meta,
    )
    child = ChildChunk(
        text=table_html, char_span=(0, len(table_html)), token_count=10,
        parent_id="x", section_name="item_8", from_table=True, metadata=meta,
    )
    return ChunkSet(parents=(parent,), children=(child,))


def test_ten_k_item8_financials_extracted_from_table_html(tmp_path: Path) -> None:
    """Worker calls the financials prompt on table_html and populates
    TenKOutput.financials. Narrative still runs single-shot."""
    raw = _SAMPLE_10K
    table = (
        "<table><tr><td>Revenue</td><td>$1,234,567</td></tr>"
        "<tr><td>Net income</td><td>$456,789</td></tr></table>"
    )
    chunkset = _chunkset_with_table(raw, table)

    narrative_payload = json.dumps(_valid_narrative())
    financials_payload = json.dumps({
        "revenue": {
            "value_usd": 1234567.0,
            "citation": {"source_quote": "Revenue"},
            "confidence": "high",
        },
        "gross_profit": None,
        "operating_income": None,
        "net_income": {
            "value_usd": 456789.0,
            "citation": {"source_quote": "Net income"},
            "confidence": "high",
        },
        "total_assets": None, "total_liabilities": None,
        "stockholders_equity": None,
        "cash_from_operations": None, "cash_from_investing": None,
        "cash_from_financing": None,
    })

    fake = MagicMock()
    fake.messages.create.side_effect = [
        _make_response(narrative_payload),
        _make_response(financials_payload),
    ]
    out = extract_ten_k(
        raw_doc=raw, doc_id="10k-table-001",
        cache_root=tmp_path, anthropic_client=cast(anthropic.Anthropic, fake),
        chunkset=chunkset,
    )
    assert out is not None
    assert out.financials is not None
    assert out.financials.revenue is not None
    assert out.financials.revenue.value_usd == 1234567.0
    assert out.financials.revenue.confidence == "high"
    assert out.financials.gross_profit is None
    # Two LLM calls — narrative + Item 8 financials.
    assert fake.messages.create.call_count == 2
```

- [ ] **Step 2: Run test — fail with NotImplementedError.**

- [ ] **Step 3: Implement the Item 8 path**

Refactor `extract_ten_k` so it:
- Runs the narrative path (single-shot or RAG).
- Independently, if chunkset has table parents, runs Item 8 extraction
  on the first table parent's `table_html` via the financials prompt.
- Merges the narrative output and the financials output via
  `model_copy(update={"financials": ...})`.

```python
import dataclasses
from auto_research.extract.prompts.ten_k_financials import (
    TEN_K_FINANCIALS_PROMPT, TEN_K_FINANCIALS_PROMPT_VERSION,
)
from auto_research.extract.schemas import TenKFinancials


def _extract_item8_financials(
    *,
    parent_table_html: str,
    doc_id: str,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None,
) -> TenKFinancials | None:
    """Run the financials prompt against `parent_table_html` via the
    shared single-shot driver. Item 8's raw_doc is the table HTML, so
    the per-row `source_quote` resolution + cache key naturally key off
    the table contents alone (different tables → different keys)."""
    return run_single_shot_extraction(
        raw_doc=parent_table_html,
        doc_id=f"{doc_id}#item8",  # distinct cache key from narrative
        worker=_WORKER,
        task=_FINANCIALS_TASK,
        prompt=TEN_K_FINANCIALS_PROMPT,
        prompt_version=TEN_K_FINANCIALS_PROMPT_VERSION,
        output_model=TenKFinancials,
        max_tokens=_FINANCIALS_MAX_TOKENS,
        cache_root=cache_root,
        quarantine_root=quarantine_root,
        anthropic_client=anthropic_client,
        client_factory=_get_client,
    )


def extract_ten_k(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
    chunkset: ChunkSet | None = None,
) -> TenKOutput | None:
    cache_root_resolved = (
        cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    )
    quarantine_root_resolved = (
        quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT
    )

    # 1. Narrative path: single-shot if short OR no chunkset; otherwise RAG.
    narrative_is_rag = (
        chunkset is not None and count_tokens(raw_doc) >= SINGLE_SHOT_TOKEN_CUTOFF
    )
    if narrative_is_rag:
        narrative = _extract_ten_k_rag(  # implemented in Task E3
            raw_doc=raw_doc, doc_id=doc_id, chunkset=chunkset,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
        )
    else:
        narrative = run_single_shot_extraction(
            raw_doc=raw_doc,
            doc_id=doc_id,
            worker=_WORKER,
            task=_NARRATIVE_TASK,
            prompt=TEN_K_NARRATIVE_PROMPT,
            prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
            output_model=TenKOutput,
            max_tokens=_NARRATIVE_MAX_TOKENS,
            cache_root=cache_root_resolved,
            quarantine_root=quarantine_root_resolved,
            anthropic_client=anthropic_client,
            client_factory=_get_client,
        )
    if narrative is None:
        return None

    # 2. Item 8 financials: independent of narrative path; runs only when
    #    a chunkset is supplied and has a table parent.
    if chunkset is None:
        return narrative
    table_parents = [p for p in chunkset.parents if p.table_html is not None]
    if not table_parents:
        return narrative
    financials = _extract_item8_financials(
        parent_table_html=table_parents[0].table_html,  # type: ignore[arg-type]
        doc_id=doc_id,
        cache_root=cache_root_resolved,
        quarantine_root=quarantine_root_resolved,
        anthropic_client=anthropic_client,
    )
    return narrative.model_copy(update={"financials": financials})
```

For Task E2, leave `_extract_ten_k_rag` as a stub raising
`NotImplementedError("see Task E3")`; the table test doesn't exercise the
RAG branch (the doc is small, so narrative_is_rag is False).

- [ ] **Step 4: Run tests — both E1 and E2 pass.**

### Task E3: 10-K RAG narrative branch

**Files:**
- Modify: `src/auto_research/extract/workers/ten_k.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`

This is the meatier of the three. The acceptance criterion is **branch
coverage** — the test must prove the RAG branch fires for tokens
≥ SINGLE_SHOT_TOKEN_CUTOFF. The minimal implementation is to call the
LLM once per narrative field with a query string derived from the field
name and the top-5 reranked parents stuffed into user content. Per the
"YAGNI" plan rule, that's all we land — the per-field query tuning is
deferred to #20 (eval suite) and #22 (backfill).

The signature accepts a `retrieve_fn` callable that, given a query,
returns a list of `ParentChunk`s. The default is a real hybrid retrieve
+ rerank pipeline; tests inject a stub.

- [ ] **Step 1: Add failing test for the RAG branch**

The test:
- Constructs a long `raw_doc` that pushes `count_tokens(raw_doc)` over
  the cutoff. (Use `"word " * 200_000` — well above 100K tokens.)
- Builds a ChunkSet without table parents (so Item 8 doesn't fire).
- Injects a `retrieve_fn` stub that returns a deterministic
  `list[ParentChunk]` regardless of query.
- Asserts the worker called the LLM multiple times (one per narrative
  field) and that the result is a valid TenKOutput with `financials is None`.

```python
def test_ten_k_rag_branch_fires_above_cutoff(tmp_path: Path) -> None:
    """count_tokens(raw_doc) >= SINGLE_SHOT_TOKEN_CUTOFF AND chunkset
    supplied → RAG branch. The minimum guarantee is that the cutoff
    branch fires; per-field retrieval/composition tuning is in #20/#22."""
    long_raw = "word " * 200_000  # well above 100K tokens
    parent_text = "Item 7 MD&A. We expect cautious growth in fiscal 2026."
    meta = ChunkMetadata(
        ticker="ACME", filing_date=date(2026, 1, 30),
        fiscal_period="FY2025", doc_type="10-K", doc_id="10k-rag-001",
    )
    parent = ParentChunk(
        text=parent_text, section_name="item_7",
        char_span=(0, len(parent_text)), token_count=10,
        table_html=None, metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    # Five LLM responses — one per narrative field (guidance_tone,
    # accrual_flags, supplier_mentions, customer_mentions, risk_factor_deltas).
    # We return the full TenKOutput shape each time and merge in the
    # worker. To keep the test small, we'll assert call_count and
    # check that the final output type is TenKOutput.
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(
        json.dumps(_valid_narrative())
    )

    retrieve_fn_calls: list[str] = []
    def fake_retrieve(query: str) -> list[ParentChunk]:
        retrieve_fn_calls.append(query)
        return [parent]

    out = extract_ten_k(
        raw_doc=long_raw, doc_id="10k-rag-001",
        cache_root=tmp_path, anthropic_client=cast(anthropic.Anthropic, fake),
        chunkset=chunkset, retrieve_fn=fake_retrieve,
    )
    assert out is not None
    assert len(retrieve_fn_calls) >= 5  # one query per narrative field
    assert out.financials is None  # no table parents in this chunkset
```

- [ ] **Step 2: Run test — fail with NotImplementedError from the stub.**

- [ ] **Step 3: Implement `_extract_ten_k_rag` and extend `extract_ten_k`'s signature**

Add `retrieve_fn: RetrieveFn | None = None` to `extract_ten_k`.

```python
from collections.abc import Callable
from auto_research.extract.chunking import ParentChunk

RetrieveFn = Callable[[str], list[ParentChunk]]

_NARRATIVE_RAG_QUERIES: dict[str, str] = {
    "guidance_tone": (
        "What is management's tone on forward growth, gross margin, and "
        "demand in the MD&A?"
    ),
    "accrual_flags": (
        "What are the accrual-quality concerns: unbilled receivables, "
        "deferred revenue, capitalized R&D, restructuring resets?"
    ),
    "supplier_mentions": (
        "Which specific named suppliers (e.g., TSMC, Foxconn, Samsung) "
        "does the company rely on?"
    ),
    "customer_mentions": (
        "Which specific named customers (hyperscalers, large "
        "enterprises) are explicitly called out?"
    ),
    "risk_factor_deltas": (
        "What new, removed, or modified Item 1A risk factors does this "
        "filing disclose?"
    ),
}


def _format_parents_as_context(parents: list[ParentChunk]) -> str:
    """Concatenate parent text with section-name headers — the LLM sees
    a single user-content block with the retrieved passages."""
    return "\n\n".join(
        f"[{p.section_name}]\n{p.text}" for p in parents
    )


def _extract_ten_k_rag(
    *,
    raw_doc: str,
    doc_id: str,
    chunkset: ChunkSet,
    retrieve_fn: RetrieveFn,
    cache_root: Path,
    quarantine_root: Path,
    anthropic_client: anthropic.Anthropic | None,
) -> TenKOutput | None:
    """Per-field RAG narrative extraction.

    For each TenKOutput narrative field, retrieve the top parents,
    format as user content, and run the narrative prompt. The prompt
    asks for the full output schema; the worker takes only the relevant
    field from each call and merges into one TenKOutput. Per-field
    LLM calls trade higher cost for per-field reranker selectivity.

    Spans are resolved against the assembled user_content, not the raw
    doc — INV-2's `source_text[span] == source_quote` holds against
    that user content, which is the text the model actually saw.
    """
    if retrieve_fn is None:
        raise ValueError(
            "RAG branch requires an explicit retrieve_fn; see #22 for the "
            "default composition via hybrid_retrieve + rerank."
        )
    partials: dict[str, Any] = {}
    for field, query in _NARRATIVE_RAG_QUERIES.items():
        parents = retrieve_fn(query)
        user_content = _format_parents_as_context(parents)
        per_field = run_single_shot_extraction(
            raw_doc=user_content,
            doc_id=f"{doc_id}#{field}",  # distinct cache key per field
            worker=_WORKER,
            task=field,
            prompt=TEN_K_NARRATIVE_PROMPT,
            prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
            output_model=TenKOutput,
            max_tokens=_NARRATIVE_MAX_TOKENS,
            cache_root=cache_root,
            quarantine_root=quarantine_root,
            anthropic_client=anthropic_client,
            client_factory=_get_client,
        )
        if per_field is None:
            return None
        partials[field] = getattr(per_field, field)
        # Identity fields (cik, accession_number, fiscal_period_end) come
        # from the first successful call.
        partials.setdefault("cik", per_field.cik)
        partials.setdefault("accession_number", per_field.accession_number)
        partials.setdefault("fiscal_period_end", per_field.fiscal_period_end)
        partials.setdefault("language_novelty_score", per_field.language_novelty_score)

    return TenKOutput(**partials)
```

Update `extract_ten_k` to accept `retrieve_fn` and pass it through to
`_extract_ten_k_rag`.

Make sure `("ten_k", "guidance_tone")`, `("ten_k", "accrual_flags")`,
`("ten_k", "supplier_mentions")`, `("ten_k", "customer_mentions")`,
`("ten_k", "risk_factor_deltas")` already exist in `_models.py` (they do
— see existing `_ROUTING`). No routing-table changes needed.

- [ ] **Step 4: Run E3 test — pass. Re-run E1/E2 — still pass.**

### Task E4: 10-K — citation-grounding on a real fixture

AC bullet 1: "Each worker produces a frozen output passing citation
grounding on a real fixture." The hermetic mocked tests above PROVE the
worker code path, but the AC asks for a real fixture. Use a small,
hand-built fixture (real 10-K text + a fixture-frozen LLM response) so
the test stays hermetic.

**Files:**
- Modify: `tests/unit/test_extract_worker_ten_k.py`

- [ ] **Step 1: Add failing test**

```python
def test_ten_k_real_fixture_passes_citation_grounding(tmp_path: Path) -> None:
    """End-to-end: real 10-K excerpt (Item 7 MD&A) → mocked LLM with a
    frozen valid response → output passes citation grounding (no
    quarantine record written). Acceptance bullet 1."""
    fixture_dir = Path(__file__).parent / "fixtures" / "ten_k"
    raw = (fixture_dir / "sample_item7.txt").read_text()
    frozen_response = (fixture_dir / "sample_item7_output.json").read_text()
    client = _fake_client(frozen_response)
    out = extract_ten_k(
        raw_doc=raw, doc_id="ten-k-fixture-001",
        cache_root=tmp_path, quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is not None
    # Critical: no quarantine record was created.
    assert not (tmp_path / "q").exists() or \
        not any((tmp_path / "q").iterdir())
    # Every Citation.source_quote indexes back to the raw text.
    for path, citation in _walk_all_citations(out):
        start, end = citation.source_span
        assert raw[start:end] == citation.source_quote, f"mismatch at {path}"


def _walk_all_citations(model):
    """Mirror guardrails._walk_citations for assertion-side use."""
    from auto_research.extract.guardrails import _walk_citations
    return list(_walk_citations(model))
```

- [ ] **Step 2: Create the fixture files**

```bash
mkdir -p tests/unit/fixtures/ten_k
```

`tests/unit/fixtures/ten_k/sample_item7.txt`: hand-pasted ~10-line MD&A
excerpt from a public 10-K (NVDA FY2024 10-K Item 7 is fine — public
domain SEC filing). Keep it small enough to read.

`tests/unit/fixtures/ten_k/sample_item7_output.json`: a hand-built TenKOutput
JSON whose `source_quote`s are verbatim from the txt file.

- [ ] **Step 3: Run test — pass.**

### Task E5: Commit Phase E

- [ ] **Step 1: Commit**

```bash
git add src/auto_research/extract/workers/ten_k.py \
        tests/unit/test_extract_worker_ten_k.py \
        tests/unit/fixtures/ten_k/
git commit -m "feat(extract): 10-K extraction worker with hybrid policy + Item 8 path"
```

---

## Phase F — Fixtures for 8-K + transcript citation grounding

To satisfy AC bullet 1 for all three workers, add fixture-based tests
for 8-K and transcript too. These are short — one txt + one json each.

### Task F1: 8-K fixture test

- [ ] **Step 1: Mirror Task E4 for 8-K.** Test asserts the worker output's
citations all index back into the raw 8-K text.

### Task F2: Transcript fixture test

- [ ] **Step 1: Mirror Task E4 for transcript.** Same shape.

### Task F3: Commit Phase F

- [ ] **Step 1: Commit**

```bash
git add tests/unit/fixtures/eight_k/ tests/unit/fixtures/transcript/ \
        tests/unit/test_extract_worker_eight_k.py \
        tests/unit/test_extract_worker_transcript.py
git commit -m "test(extract): citation-grounding fixtures for 8-K and transcript"
```

---

## Phase G — Verification + PR

### Task G1: Full extract test suite + make quick

- [ ] **Step 1: Run targeted suite**

```bash
uv run pytest tests/unit/test_extract_worker_common.py \
              tests/unit/test_extract_worker_s_filings.py \
              tests/unit/test_extract_worker_eight_k.py \
              tests/unit/test_extract_worker_transcript.py \
              tests/unit/test_extract_worker_ten_k.py \
              tests/unit/test_extract_schemas.py \
              tests/unit/test_extract_guardrails.py \
              tests/unit/test_extract_prompts.py \
              tests/unit/test_models.py -v
```

Expected: all pass.

- [ ] **Step 2: make quick (ruff + mypy)**

Run: `make quick`
Expected: clean (no lint or type errors).

- [ ] **Step 3: make check (full unit suite excluding eval/integration)**

Run: `make check`
Expected: clean.

### Task G2: Apply bump-prompt-version skill

Per AC bullet 5. The skill confirms each prompt has a corresponding
version constant. Since these are NEW prompts (no prior version), and
per the user-feedback memory "don't bump *_VERSION tags for prompt/contract
edits during pre-deployment", `v1` is correct for all four — no bump.

- [ ] **Step 1: Invoke the bump-prompt-version skill via the Skill tool.**

Follow the skill's instructions. Expected outcome: confirm `v1` is right
because no downstream worker consumes these prompts yet (the orchestrator
in #22 is the first consumer).

### Task G3: Commit + push + PR

- [ ] **Step 1: Verify no uncommitted changes via `git status`**

Expected: clean working tree, branch ahead of origin by N commits.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/19-extraction-workers
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --title "feat(extract): 10-K, transcript, 8-K worker bodies + prompts (#19)" \
  --body "$(cat <<'EOF'
Closes #19.

## Summary
- Shared scaffolding (`extract/workers/_common.py`) extracted from
  s_filings worker so the four extraction workers share JSON parse,
  span resolution, quarantine routing, and OTel instrumentation.
- New workers: `ten_k`, `transcript`, `eight_k`.
- New prompts: `ten_k_narrative` v1, `ten_k_financials` v1,
  `transcript` v1, `eight_k` v1.
- 10-K hybrid policy: single-shot for `count_tokens(raw_doc) <
  SINGLE_SHOT_TOKEN_CUTOFF`, RAG-per-narrative-field above. Item 8
  financials always extracted from `ParentChunk.table_html` when a
  chunkset is supplied with a table parent.
- Additive schema change: `TenKFinancials` + `FinancialLineItem` (new),
  `TenKOutput.financials: TenKFinancials | None = None` (additive,
  no SCHEMA_VERSION bump).

## Acceptance criteria
- [x] Each worker produces a frozen output passing citation grounding
  on a real fixture — `test_ten_k_real_fixture_passes_citation_grounding`,
  `test_eight_k_real_fixture_passes_citation_grounding`,
  `test_transcript_real_fixture_passes_citation_grounding`.
- [x] Each prompt has its own version constant — `*_PROMPT_VERSION` in
  each `extract/prompts/*.py`. Langfuse registration covered by the
  existing `register_prompt` path (Issue #11); promotion via
  `scripts/promote_prompt.py`.
- [x] Hybrid extraction policy — `test_ten_k_single_shot_branch`,
  `test_ten_k_rag_branch_fires_above_cutoff`.
- [x] 10-K Item 8 financials via typed Pydantic schema, not dense
  retrieval — `test_ten_k_item8_financials_extracted_from_table_html`
  and `TenKFinancials` schema.
- [x] `bump-prompt-version` skill applied — see G2 above.

## Change Contract (Tier 2)
- INV-1 (PIT): N/A.
- INV-2 (citation grounding): preserved. Span resolution +
  validate_or_quarantine are unchanged; the scaffolding refactor
  preserves observable behavior (s_filings test suite passes
  unchanged). New workers route to the same validator.
- INV-6 (version pinning): new prompts pinned at v1; routing-table
  row added for `("ten_k", "financials")`.

## Test plan
- [x] `pytest tests/unit/test_extract_*` — all green.
- [x] `make quick` — clean.
- [x] `make check` — clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Print PR URL** (so the user can review).

---

## Self-review checklist

**Spec coverage.** Each AC bullet maps:
- AC1 (citation grounding on a real fixture): Tasks E4, F1, F2 ✓
- AC2 (per-prompt version constant + Langfuse): Tasks C1-C4 ✓
- AC3 (hybrid policy with branch coverage): Tasks E1, E3 ✓
- AC4 (Item 8 via typed Pydantic schema): Tasks B1, E2 ✓
- AC5 (bump-prompt-version skill applied): Task G2 ✓

**Placeholder scan.** No "TBD" / "TODO" left. Specific file paths,
specific test names, real code in every code step.

**Type consistency.** `run_single_shot_extraction` signature consistent
across A3, A4, D1, D2, E1, E2, E3. `retrieve_fn: RetrieveFn` typed and
used identically in E3's test and the implementation.

**Scope.** One implementation plan, four workers (three new + s_filings
refactor), one PR. No coupling to #20 (eval) or #22 (backfill) beyond
the RAG `retrieve_fn` parameter, which is left injectable for #22 to
wire to the real `hybrid_retrieve + rerank` composition.
