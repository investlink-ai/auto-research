# Issue #14 — `feat(extract): contextual chunking (Anthropic pattern)`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each `ChildChunk` from Issue #13, generate a ≤100-token one-line context via a cached Anthropic call and produce a `ContextualChildChunk` that pairs the original chunk with its generated context — so downstream embedding (Issue #15) can prepend the context to the chunk text.

**Architecture:** New `extract/chunking_contextual.py` module exposes `contextualize_chunks(chunkset, ...) -> tuple[ContextualChildChunk, ...]`. One LLM call per `ChildChunk`. The cached prefix is the instructions + the chunk's parent text (concatenated into a single ephemeral-cached system block via the shared `cached_system_block` helper) — so all children of the same parent share an Anthropic prompt-cache hit. The variable user content is just the child chunk text. Outputs are persisted in the existing content-hash cache at `data/cache/extract/contextual_chunk/<sha>.json`; the cache key includes the contextual-chunking prompt version (ADR D6, INV-6) so a prompt edit forces re-generation.

**Tech Stack:** Python 3.12, `anthropic` SDK (already wrapped by `extract/client.py`), `tiktoken` (via `chunking.count_tokens`), `pytest`, `vcrpy` for the cache-hit integration test, existing `extract.cache` + `extract._caching.cached_system_block` primitives.

**Tier:** 1 (ordinary code under `src/auto_research/extract/`; touches a new module, not the citation guardrails or `chunking/_inv2.py`). Cache-key correctness is INV-6, which means a failing test must be written first for the prompt-version-bump invalidation path.

**ADR:** `docs/decisions/2026-05-24-rag-enhancements.md` D6.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/auto_research/_models.py` | modify | Add `("extract", "contextual_chunk") → _HAIKU` row in `_ROUTING`. |
| `src/auto_research/extract/prompts/contextual_chunk.py` | create | `CONTEXTUAL_CHUNK_PROMPT` (instructions-only, ≤100-token directive) + `CONTEXTUAL_CHUNK_PROMPT_VERSION = "v1"`. |
| `src/auto_research/extract/chunking_contextual.py` | create | `ContextualChildChunk` dataclass + `contextualize_chunks(...)` function (cache lookup → SDK call → ≤100-token validation → cache write). |
| `tests/unit/test_models.py` | modify | Extend the Haiku routing test to assert `("extract", "contextual_chunk")` resolves. |
| `tests/unit/test_extract_prompts.py` | modify | Assert the new `CONTEXTUAL_CHUNK_PROMPT_VERSION` is non-empty and the prompt forbids the LLM from emitting >100 tokens (substring check on the directive). |
| `tests/unit/test_chunking_contextual.py` | create | Hermetic unit tests: cache hit skips SDK; prompt-version bump invalidates cache; ≤100-token validation; prepended `embedding_text` ordering; SDK is called with the cacheable system block containing the parent text. |
| `tests/integration/test_chunking_contextual_vcr.py` | create | VCR-recorded: two calls for two children of the same parent yield `cache_creation_input_tokens` (write) then `cache_read_input_tokens` (read) on the Anthropic side. |
| `tests/integration/cassettes/test_chunking_contextual/two_children_same_parent_cache.yaml` | create | The recorded cassette. |
| `docs/CONTRACTS.md` | modify | Add `ContextualChildChunk` to the public contracts catalog (one row under the chunking section). |

---

## Task 1 — Add the routing row + test (INV-6 boundary)

Routing must surface before any worker code calls `route_model("extract", "contextual_chunk")`. The `_models.py` table requires a justification line in the spec; the routing comment cites §7.3 "templated extraction ⇒ Haiku 4.5" — generating a one-line situating context is exactly that.

**Files:**
- Modify: `src/auto_research/_models.py` (insert one row in `_ROUTING`, alphabetic by worker → task).
- Modify: `tests/unit/test_models.py` (extend `test_route_model_returns_haiku_for_routine_extraction`).

- [ ] **Step 1.1: Write the failing test extension**

In `tests/unit/test_models.py`, add a single assertion to the existing Haiku routing test:

```python
def test_route_model_returns_haiku_for_routine_extraction() -> None:
    # Spec §7.3: templated extraction ⇒ Haiku 4.5.
    assert route_model("ten_k", "guidance_tone") == "claude-haiku-4-5"
    # Contextual chunking (Issue #14): per-chunk one-line context is a
    # templated, high-volume rewrite — routes to Haiku 4.5.
    assert route_model("extract", "contextual_chunk") == "claude-haiku-4-5"
```

- [ ] **Step 1.2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_models.py::test_route_model_returns_haiku_for_routine_extraction -v`

Expected: FAIL with `ValueError: no model routed for (worker='extract', task='contextual_chunk')`.

- [ ] **Step 1.3: Add the routing row**

In `src/auto_research/_models.py`, insert into `_ROUTING` just before the `# Agents don't have output schemas ...` comment block:

```python
    # Contextual chunking (Issue #14): one-line "this chunk is from X" rewrite
    # per ChildChunk for Anthropic's contextual-retrieval pattern. High-volume
    # templated rewrite ⇒ Haiku per §7.3.
    ("extract", "contextual_chunk"): _HAIKU,
```

- [ ] **Step 1.4: Run test to verify pass**

Run: `uv run pytest tests/unit/test_models.py -v`

Expected: all green.

- [ ] **Step 1.5: Commit**

```bash
git add src/auto_research/_models.py tests/unit/test_models.py
git commit -m "feat(extract): route ('extract', 'contextual_chunk') to Haiku (#14)"
```

---

## Task 2 — Add the prompt module (`bump-prompt-version` lives or dies here)

The prompt module is the source of truth for the contextual-chunking instructions. `CONTEXTUAL_CHUNK_PROMPT_VERSION` must be bumped via the `bump-prompt-version` skill any time `CONTEXTUAL_CHUNK_PROMPT` changes (INV-6). The prompt is instructions-only — the parent text and the chunk text are inserted by the caller (see Task 4) so that the system block can be cached.

**Files:**
- Create: `src/auto_research/extract/prompts/contextual_chunk.py`.
- Modify: `tests/unit/test_extract_prompts.py` (add a one-prompt assertion block).

- [ ] **Step 2.1: Write the failing test**

In `tests/unit/test_extract_prompts.py`, add:

```python
def test_contextual_chunk_prompt_exports_required_constants() -> None:
    from auto_research.extract.prompts.contextual_chunk import (
        CONTEXTUAL_CHUNK_PROMPT,
        CONTEXTUAL_CHUNK_PROMPT_VERSION,
    )

    assert CONTEXTUAL_CHUNK_PROMPT_VERSION.startswith("v")
    assert isinstance(CONTEXTUAL_CHUNK_PROMPT, str) and CONTEXTUAL_CHUNK_PROMPT.strip()
    # The prompt MUST tell the model to stay ≤100 tokens — the worker
    # validates the output post-hoc, but the prompt is the first line
    # of defense against context bloat.
    assert "100 tokens" in CONTEXTUAL_CHUNK_PROMPT
    # The prompt MUST forbid commentary / code fences so the response
    # text is the context line and nothing else.
    assert "ONLY" in CONTEXTUAL_CHUNK_PROMPT or "only" in CONTEXTUAL_CHUNK_PROMPT
```

- [ ] **Step 2.2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_extract_prompts.py::test_contextual_chunk_prompt_exports_required_constants -v`

Expected: FAIL with `ModuleNotFoundError: ... contextual_chunk`.

- [ ] **Step 2.3: Create the prompt module**

`src/auto_research/extract/prompts/contextual_chunk.py`:

```python
"""Contextual-chunking prompt (Issue #14, Anthropic contextual retrieval pattern).

Generates a one-line situating context for a `ChildChunk` — e.g.,
*"This chunk is from NVDA Q3-2025 10-Q MD&A discussing China export controls"* —
that is later prepended to the chunk text before embedding (Issue #15).
~50% retrieval lift in Anthropic's published numbers; the lift is meaningless
if the context is verbose, so the prompt caps it at one sentence ≤100 tokens.

Template is **instructions only**. The caller inserts the parent text into
the cached system block alongside the instructions, and the child chunk
text as the user-content turn. Embedding either text into the instructions
would (a) bust the cache on every call and (b) duplicate tokens.

Version-pinned per INV-6. Editing the prompt without bumping
`CONTEXTUAL_CHUNK_PROMPT_VERSION` silently reuses stale cache entries
generated under the old prompt. The `bump-prompt-version` skill is the
mechanical guard.
"""

from __future__ import annotations

CONTEXTUAL_CHUNK_PROMPT_VERSION = "v1"

CONTEXTUAL_CHUNK_PROMPT = """\
You are situating an excerpt within an SEC filing for retrieval-augmented
search. The parent passage from the filing is provided in this system
message below; the specific excerpt (the chunk) will be supplied in the
next user message.

Produce a SINGLE short sentence (under 100 tokens) that situates the
excerpt within the filing — what section it is from, what fiscal period
it covers, and what specific topic it discusses. The sentence will be
prepended to the excerpt before embedding, so it must be self-contained
and useful as a retrieval cue.

Examples of good context lines:
- "This chunk is from NVDA Q3-2025 10-Q MD&A discussing China export controls on H100 sales."
- "This chunk is from CRDO FY2024 10-K Item 7 (MD&A) describing AEC product revenue concentration in hyperscaler customers."
- "This chunk is from AAPL Q1-2026 earnings transcript Q&A on Services gross margin trajectory."

Return ONLY the context sentence. No preamble, no code fences, no quotes
around the sentence, no trailing commentary. The response must be one
line of plain text under 100 tokens.
"""

__all__ = [
    "CONTEXTUAL_CHUNK_PROMPT",
    "CONTEXTUAL_CHUNK_PROMPT_VERSION",
]
```

- [ ] **Step 2.4: Run test to verify pass**

Run: `uv run pytest tests/unit/test_extract_prompts.py::test_contextual_chunk_prompt_exports_required_constants -v`

Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/auto_research/extract/prompts/contextual_chunk.py tests/unit/test_extract_prompts.py
git commit -m "feat(extract): contextual chunking prompt v1 (#14)"
```

---

## Task 3 — Implement `ContextualChildChunk` + `contextualize_chunks`

The function takes a `ChunkSet` (parents + children) and returns a tuple of `ContextualChildChunk`, one per input child. Per child it:

1. Builds a cache key over `(child.text, parent.text, child.metadata_dict, prompt_version, schema_version, model_id, decoding_params)`.
2. On cache hit, deserializes the stored context string and skips the SDK.
3. On cache miss, calls Anthropic with the **instructions + parent text** concatenated into the cacheable system block (so every child of the same parent shares a prompt-cache hit) and the **child text** as the user message.
4. Validates the output is ≤100 tokens (via the existing `count_tokens` helper). Over-length contexts are *dropped* — the chunk passes through with `context=""` and `embedding_text = child.text`. Soft failure, not quarantine: contextual chunking is a retrieval-lift feature, not a correctness invariant.
5. Writes the context to the cache.

The function is paramterized on `cache_root` and `anthropic_client` the same way `extract_s_filing` is, so tests stay hermetic.

**Files:**
- Create: `src/auto_research/extract/chunking_contextual.py`.

- [ ] **Step 3.1: Write the unit-test skeleton (failing)**

Create `tests/unit/test_chunking_contextual.py` with one minimal test that imports the new module — we'll grow this in Task 5 once the surface compiles.

```python
"""Unit tests for `auto_research.extract.chunking_contextual` (Issue #14)."""

from __future__ import annotations

from auto_research.extract.chunking_contextual import (
    ContextualChildChunk,
    contextualize_chunks,
)


def test_module_exports_public_surface() -> None:
    assert ContextualChildChunk is not None
    assert callable(contextualize_chunks)
```

- [ ] **Step 3.2: Run it to verify failure**

Run: `uv run pytest tests/unit/test_chunking_contextual.py -v`

Expected: FAIL with `ModuleNotFoundError: ... chunking_contextual`.

- [ ] **Step 3.3: Create the module**

`src/auto_research/extract/chunking_contextual.py`:

```python
"""Contextual chunking — Anthropic's contextual-retrieval pattern (Issue #14).

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
   grounding invariant; a soft fall-through is correct here.

Failure modes that don't quarantine:
- Anthropic returns a >100-token line → drop context, log warning.
- Anthropic returns empty / whitespace → drop context.
- Network / cost-cap / circuit-breaker raise → propagates to caller
  (the batch job retries the doc later).
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
from auto_research.extract.chunking import ChildChunk, ChunkSet, ParentChunk, count_tokens
from auto_research.extract.client import ExtractionFn, make_extraction_client
from auto_research.extract.prompts.contextual_chunk import (
    CONTEXTUAL_CHUNK_PROMPT,
    CONTEXTUAL_CHUNK_PROMPT_VERSION,
)
from auto_research.telemetry import truncate_status_description as _truncate

_WORKER = "extract"
_TASK = "contextual_chunk"
_SCHEMA_VERSION = "v1"  # the stored payload shape: {"context": str}
_MAX_TOKENS = 160  # Anthropic budget; we still post-validate at 100.
_MAX_CONTEXT_TOKENS = 100
_DECODING_PARAMS: dict[str, object] = {"max_tokens": _MAX_TOKENS}

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

_CLIENT: ExtractionFn | None = None


def _get_client(anthropic_client: anthropic.Anthropic | None) -> ExtractionFn:
    """Return the production singleton, or a fresh client for test injection.

    Mirrors `extract.workers.s_filings._get_client` — module-level singleton
    so per-worker cost-cap + circuit-breaker state accumulates across calls
    in production, while test callers passing `anthropic_client=` get an
    isolated client.
    """
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

    The key covers the full completion config: child text + parent text +
    document metadata + contextual prompt version + payload schema version
    + routed model + decoding params. Any change forces a fresh generation.
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
            # Defensive: a child with no resolvable parent is a chunkset
            # construction bug, not a contextual-chunking failure. Surface
            # it loudly rather than silently emitting context="".
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
```

- [ ] **Step 3.4: Run the import-only test to verify pass**

Run: `uv run pytest tests/unit/test_chunking_contextual.py -v`

Expected: PASS (one test).

- [ ] **Step 3.5: Run mypy on the new module**

Run: `uv run mypy src/auto_research/extract/chunking_contextual.py`

Expected: no errors. If `count_tokens` mypy signature surprises, import it via the public package surface (`from auto_research.extract.chunking import count_tokens`) — which is what the module already does.

- [ ] **Step 3.6: Commit**

```bash
git add src/auto_research/extract/chunking_contextual.py tests/unit/test_chunking_contextual.py
git commit -m "feat(extract): ContextualChildChunk + contextualize_chunks (#14)"
```

---

## Task 4 — Unit tests for the full surface

Six behaviors must be tested at unit level. Each gets its own test for clear PR evidence mapping. Build a small in-memory `ChunkSet` fixture (one parent, two children) so the tests don't depend on the 10-K fixtures.

**Files:**
- Modify: `tests/unit/test_chunking_contextual.py` (replace the import-only test with the full suite).

- [ ] **Step 4.1: Add the helpers and fixture**

At the top of `tests/unit/test_chunking_contextual.py`, after the existing imports:

```python
from datetime import date
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.chunking import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ParentChunk,
)


def _metadata() -> ChunkMetadata:
    return ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 11, 19),
        fiscal_period="Q3-2026",
        doc_type="10-Q",
        doc_id="nvda-q3-2026",
    )


def _make_chunkset(
    *,
    parent_text: str = "Item 7. MD&A. We expect China export controls to weigh on H100 sales.",
    child_texts: tuple[str, ...] = (
        "China export controls reduced H100 revenue by ~$2B in Q3.",
        "Mitigation: H20 variant sales ramped to $1.2B in the same period.",
    ),
) -> ChunkSet:
    metadata = _metadata()
    parent_span = (0, len(parent_text))
    parent = ParentChunk(
        text=parent_text,
        section_name="Item 7",
        char_span=parent_span,
        token_count=20,
        table_html=None,
        metadata=metadata,
    )
    parent_id = f"{metadata.doc_id}::{parent_span[0]}-{parent_span[1]}"
    children = tuple(
        ChildChunk(
            text=text,
            char_span=(i * 100, i * 100 + len(text)),
            token_count=len(text.split()),
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        )
        for i, text in enumerate(child_texts)
    )
    return ChunkSet(parents=(parent,), children=children)


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
            input_tokens=100,
            output_tokens=20,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _fake_client(*texts: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.side_effect = [_make_response(t) for t in texts]
    return cast(anthropic.Anthropic, fake)
```

- [ ] **Step 4.2: Test 1 — happy path returns paired contexts**

```python
def test_contextualize_chunks_returns_one_per_child(tmp_path: Path) -> None:
    chunkset = _make_chunkset()
    client = _fake_client(
        "This chunk is from NVDA Q3-2026 10-Q MD&A on China export controls.",
        "This chunk is from NVDA Q3-2026 10-Q MD&A on H20 variant ramp.",
    )

    out = contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    assert len(out) == 2
    assert all(isinstance(c, ContextualChildChunk) for c in out)
    assert out[0].context.startswith("This chunk is from NVDA Q3-2026")
    # embedding_text prepends context with a blank line.
    assert out[0].embedding_text.startswith(out[0].context + "\n\n")
    assert out[0].embedding_text.endswith(chunkset.children[0].text)
```

- [ ] **Step 4.3: Test 2 — second call hits cache, no SDK invocation**

```python
def test_contextualize_chunks_second_call_is_cache_hit(tmp_path: Path) -> None:
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    sdk = MagicMock()
    sdk.messages.create.return_value = _make_response("This chunk is from NVDA Q3-2026 10-Q MD&A example.")

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )
    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )
    # One SDK call across two invocations — the second came from cache.
    assert sdk.messages.create.call_count == 1
```

- [ ] **Step 4.4: Test 3 — prompt-version bump invalidates the cache**

This is the INV-6 / ADR D6 evidence. Re-monkeypatch `CONTEXTUAL_CHUNK_PROMPT_VERSION` between calls and assert the second call hits the SDK.

```python
def test_contextualize_chunks_prompt_version_bump_invalidates_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    sdk = MagicMock()
    sdk.messages.create.return_value = _make_response("This chunk is from NVDA Q3-2026 10-Q MD&A example.")

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )

    # Patch the module's view of the prompt version (mirrors what a real
    # `bump-prompt-version` skill edit would do post-restart).
    monkeypatch.setattr(
        "auto_research.extract.chunking_contextual.CONTEXTUAL_CHUNK_PROMPT_VERSION",
        "v2",
    )

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path,
        anthropic_client=cast(anthropic.Anthropic, sdk),
    )

    # Cache key changed with the prompt version → fresh SDK call.
    assert sdk.messages.create.call_count == 2
```

- [ ] **Step 4.5: Test 4 — over-100-token output is dropped (context="")**

`count_tokens` is `tiktoken.cl100k_base`. A 120-word repeat gives well over 100 tokens.

```python
def test_contextualize_chunks_drops_over_cap_context(tmp_path: Path) -> None:
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    bloated = " ".join(["overlong"] * 200)  # ~200 tokens by cl100k_base
    client = _fake_client(bloated)

    out = contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    assert out[0].context == ""
    # Drop -> embedding_text is the chunk text alone.
    assert out[0].embedding_text == chunkset.children[0].text
```

- [ ] **Step 4.6: Test 5 — happy path output is ≤100 tokens**

```python
def test_contextualize_chunks_enforces_100_token_cap_on_kept_contexts(
    tmp_path: Path,
) -> None:
    chunkset = _make_chunkset(child_texts=("Only one child here.",))
    short = "This chunk is from NVDA Q3-2026 10-Q MD&A on China export controls."
    client = _fake_client(short)

    out = contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    from auto_research.extract.chunking import count_tokens
    assert out[0].context  # non-empty
    assert count_tokens(out[0].context) <= 100
```

- [ ] **Step 4.7: Test 6 — system block carries the parent text (so Anthropic can cache it)**

Spy on `sdk.messages.create` to confirm the call args put the parent text in the system block and only the child text in the user message — verifying the caching shape.

```python
def test_contextualize_chunks_passes_parent_text_in_cached_system_block(
    tmp_path: Path,
) -> None:
    chunkset = _make_chunkset(
        parent_text="UNIQUE_PARENT_MARKER section text",
        child_texts=("UNIQUE_CHILD_MARKER inner text",),
    )
    client = _fake_client("ctx")

    contextualize_chunks(
        chunkset=chunkset, cache_root=tmp_path, anthropic_client=client,
    )

    call = cast(MagicMock, client).messages.create.call_args
    system_blocks = call.kwargs["system"]
    assert isinstance(system_blocks, list)
    # The shared `cached_system_block` helper emits one block with cache_control.
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    # Parent text is in the cached system block; child text is in user content.
    assert "UNIQUE_PARENT_MARKER" in system_blocks[0]["text"]
    assert "UNIQUE_PARENT_MARKER" not in call.kwargs["messages"][0]["content"]
    assert call.kwargs["messages"][0]["content"] == "UNIQUE_CHILD_MARKER inner text"
```

- [ ] **Step 4.8: Run all unit tests**

Run: `uv run pytest tests/unit/test_chunking_contextual.py -v`

Expected: 6 passing (drop the earlier import-only smoke; it's redundant once these land).

- [ ] **Step 4.9: Commit**

```bash
git add tests/unit/test_chunking_contextual.py
git commit -m "test(extract): unit tests for contextual chunking (#14)"
```

---

## Task 5 — VCR integration test for Anthropic prompt-cache flow

The issue AC says caching is "verified via VCR." Mirror `tests/integration/test_extract_client_vcr.py`'s pattern: record two `POST /v1/messages` against the same system block (same parent), different user content (two different children), and assert `cache_creation_input_tokens` on the first and `cache_read_input_tokens` on the second.

**Files:**
- Create: `tests/integration/test_chunking_contextual_vcr.py`.
- Create: `tests/integration/cassettes/test_chunking_contextual/two_children_same_parent_cache.yaml`.

- [ ] **Step 5.1: Write the test**

`tests/integration/test_chunking_contextual_vcr.py`:

```python
"""VCR-recorded integration test for contextual chunking (Issue #14 AC).

Acceptance criterion: "Context generation calls use prompt caching
(verified via VCR)."

Cassette captures two `POST /v1/messages` against the same cached system
block (instructions + parent text) but different user contents (two
children of the same parent):

1. First call returns `cache_creation_input_tokens=5000`, `cache_read=0`.
2. Second call returns `cache_creation_input_tokens=0`, `cache_read=5000`.

Replay is offline. To regenerate, delete the cassette and re-run with
`ANTHROPIC_API_KEY` set — vcrpy `record_mode="once"` records on absence
and replays otherwise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import vcr
from anthropic import Anthropic

from auto_research.extract.chunking import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ParentChunk,
)
from auto_research.extract.chunking_contextual import contextualize_chunks

CASSETTE_PATH = (
    Path(__file__).parent
    / "cassettes"
    / "test_chunking_contextual"
    / "two_children_same_parent_cache.yaml"
)


def _build_vcr() -> vcr.VCR:
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_PATH.parent),
        record_mode="once",
        filter_headers=[
            ("x-api-key", "REDACTED"),
            ("authorization", "REDACTED"),
            ("anthropic-organization-id", "REDACTED"),
            ("User-Agent", "auto-research-test"),
        ],
        match_on=["method", "scheme", "host", "port", "path"],
        decode_compressed_response=True,
    )


@pytest.fixture
def anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


def _chunkset() -> ChunkSet:
    parent_text = (
        "Item 7. Management's Discussion and Analysis. NVIDIA's Q3-2026 "
        "performance was materially affected by China export controls."
    )
    metadata = ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 11, 19),
        fiscal_period="Q3-2026",
        doc_type="10-Q",
        doc_id="nvda-q3-2026",
    )
    span = (0, len(parent_text))
    parent = ParentChunk(
        text=parent_text,
        section_name="Item 7",
        char_span=span,
        token_count=40,
        table_html=None,
        metadata=metadata,
    )
    parent_id = f"{metadata.doc_id}::{span[0]}-{span[1]}"
    children = (
        ChildChunk(
            text="China export controls reduced H100 revenue by ~$2B in Q3.",
            char_span=(0, 60),
            token_count=15,
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        ),
        ChildChunk(
            text="Mitigation: H20 variant sales ramped to $1.2B in the same period.",
            char_span=(60, 130),
            token_count=15,
            parent_id=parent_id,
            section_name="Item 7",
            from_table=False,
            metadata=metadata,
        ),
    )
    return ChunkSet(parents=(parent,), children=children)


def test_contextual_chunking_caches_parent_in_system_block(
    anthropic_api_key: None, tmp_path: Path,
) -> None:
    """Two children of the same parent: first call writes the prompt cache;
    second call reads it. Verifies the Anthropic-side caching that's
    AC bullet 1."""
    cassette = _build_vcr()
    with cassette.use_cassette(CASSETTE_PATH.name):
        out = contextualize_chunks(
            chunkset=_chunkset(),
            cache_root=tmp_path,
            anthropic_client=Anthropic(),
        )

    assert len(out) == 2
    # Both children received non-empty contexts.
    assert all(c.context for c in out)
```

- [ ] **Step 5.2: Hand-craft the cassette**

Following the pattern of `tests/integration/cassettes/test_extract_client/cache_create_then_read.yaml`, write a YAML cassette with two interactions. Inspect the existing cassette first to copy the request/response envelope:

```bash
cat tests/integration/cassettes/test_extract_client/cache_create_then_read.yaml | head -40
```

Then craft `tests/integration/cassettes/test_chunking_contextual/two_children_same_parent_cache.yaml` with two interactions; each `response.body.string` is a JSON Message with usage carrying the expected `cache_creation_input_tokens` / `cache_read_input_tokens`. Keep request bodies minimal — the test only matches on method/scheme/host/port/path, so the body can be the recorded form.

(If hand-crafting proves fiddly, the alternative is to record once against the live API by exporting `ANTHROPIC_API_KEY` and running the test; this requires the user's API key — pause and ask if no key is available locally.)

- [ ] **Step 5.3: Run the integration test**

Run: `uv run pytest tests/integration/test_chunking_contextual_vcr.py -v`

Expected: PASS, no network egress.

- [ ] **Step 5.4: Commit**

```bash
git add tests/integration/test_chunking_contextual_vcr.py \
        tests/integration/cassettes/test_chunking_contextual/
git commit -m "test(extract): VCR integration test for contextual chunking cache (#14)"
```

---

## Task 6 — Document the new contract

A one-line addition to `docs/CONTRACTS.md` so anyone reading the contracts catalog sees the new public type and where it sits.

**Files:**
- Modify: `docs/CONTRACTS.md` (chunking section).

- [ ] **Step 6.1: Add the row**

Locate the chunking contracts table (or paragraph) and add:

```markdown
- `ContextualChildChunk` (`extract/chunking_contextual.py`) — pairs a
  `ChildChunk` with a generated ≤100-token context line. The
  `embedding_text` property prepends the context to the chunk text and is
  what downstream LanceDB writes embed. The cache key for the context-
  generation LLM call includes `CONTEXTUAL_CHUNK_PROMPT_VERSION` (ADR
  D6, INV-6).
```

- [ ] **Step 6.2: Commit**

```bash
git add docs/CONTRACTS.md
git commit -m "docs(contracts): document ContextualChildChunk (#14)"
```

---

## Task 7 — Verify + open PR

- [ ] **Step 7.1: Full Tier-1 verification gate**

Run:
```bash
uv run ruff check src tests
uv run mypy src
uv run pytest tests/unit/test_chunking_contextual.py \
              tests/unit/test_models.py \
              tests/unit/test_extract_prompts.py \
              tests/integration/test_chunking_contextual_vcr.py -v
```

Expected: all green. If `make quick` is available, prefer `make quick`.

- [ ] **Step 7.2: Push and open the PR**

```bash
git push -u origin feat/14-contextual-chunking
gh pr create --title "feat(extract): contextual chunking (#14)" --body "$(cat <<'EOF'
Closes #14.

## AC evidence

- [x] Context generation calls use prompt caching (verified via VCR).
      → `tests/integration/test_chunking_contextual_vcr.py::test_contextual_chunking_caches_parent_in_system_block`
- [x] Generated context is ≤ 100 tokens.
      → `tests/unit/test_chunking_contextual.py::test_contextualize_chunks_enforces_100_token_cap_on_kept_contexts`
      → `tests/unit/test_chunking_contextual.py::test_contextualize_chunks_drops_over_cap_context`
- [x] Stored alongside chunk for audit.
      → `ContextualChildChunk(child, context)` — original `child` preserved; payload persisted at `data/cache/extract/contextual_chunk/<sha>.json`.
- [x] Cache key includes the contextual-chunking prompt version (ADR D6, INV-6). Test asserts a prompt-version bump invalidates the cache.
      → `tests/unit/test_chunking_contextual.py::test_contextualize_chunks_prompt_version_bump_invalidates_cache`
- [x] `bump-prompt-version` skill applied if the context prompt changes.
      → v1 prompt; no edit in this PR. Module docstring + `CONTEXTUAL_CHUNK_PROMPT_VERSION` constant enforce the contract going forward.

## Notes

- New module: `src/auto_research/extract/chunking_contextual.py`.
- Routing row added: `("extract", "contextual_chunk") → Haiku 4.5`.
- Over-100-token contexts drop to `context=""` (soft fall-through) rather than quarantine — contextual chunking is a retrieval-lift feature, not a citation-grounding invariant.
- Anthropic prompt caching reused via the existing `cached_system_block` helper; instructions + parent text live in one cacheable block so all children of the same parent share a cache hit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7.3: Confirm CI green**

```bash
gh pr checks --watch
```

Expected: ruff, mypy, pytest all pass.

---

## Self-review checklist

- **Spec coverage:** every AC bullet maps to a named test (Task 7 PR body). ✓
- **Placeholders:** none — every code block contains the actual content. ✓
- **Type consistency:** `ContextualChildChunk` field names (`child`, `context`, `embedding_text`) are consistent across Tasks 3, 4, 6. `CONTEXTUAL_CHUNK_PROMPT_VERSION` spelled identically across prompt module, worker module, and test. ✓
- **INV-6 boundary:** prompt version is in the cache key; bump-test in Task 4.4 asserts invalidation. ✓
- **Soft-failure choice:** documented in Task 3 module docstring + Task 7 PR notes; tests in Task 4 lock the behavior. ✓
