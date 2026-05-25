# Issue #13 — `feat(extract): unstructured.io parsing + section-aware chunking`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `extract/chunking.py` — pure-Python, deterministic SEC-filing parser producing parent + child chunks with metadata, that honors INV-2 (`source_text[chunk.char_span] == chunk.text`) and routes char_span mismatches to `data/quarantine/`.

**Architecture:** Library-first via the `unstructured` OSS Python package (`partition_html`). Two chunk levels — `ParentChunk` (≤4K tokens, the extraction context unit; section-respecting) and `ChildChunk` (~200–800 tokens, the retrieval embedding unit; never crosses a parent boundary). Metadata `(ticker, filing_date, fiscal_period, doc_type, doc_id)` carried on every chunk. 10-K Item 8 `Table` elements emit as summary `ParentChunk`s with raw `table_html` attached, bypassing dense retrieval (ADR D5).

**Tech Stack:** Python 3.12, `unstructured[html,pdf]`, `tiktoken` (cl100k_base encoding), `pytest`, `auto_research._io.atomic_write_text` (existing quarantine I/O), `auto_research.extract.guardrails`'s quarantine pattern as the model.

**Tier:** 2 (touches INV-2 char_span contract — `AGENTS.md` §3, `docs/AI_WORKFLOW.md` §2 path matrix → escalator: citation-grounding).

**ADR:** `docs/decisions/2026-05-24-rag-enhancements.md` D1, D3, D4, D5, D7, D9.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add pinned `unstructured[html,pdf]`, `tiktoken` to runtime deps. |
| `src/auto_research/extract/chunking.py` | create | The `parse_filing(html, metadata) → ChunkSet` entrypoint and supporting dataclasses + validation. |
| `tests/fixtures/chunking/sample_10k.htm` | create (binary asset) | Real EDGAR 10-K HTML, small-but-complete; one ticker, one fiscal year. Picked: CRDO FY2024 (~1.2 MB, has Items 1, 1A, 7, 7A, 8 all present). |
| `tests/fixtures/chunking/edge_cases.htm` | create | Hand-crafted minimal HTML containing the `unstructured` known-failure edge cases: `&nbsp;`, `&#8217;`, nested `<span>`, CDATA. |
| `tests/unit/test_chunking.py` | create | Unit tests covering all AC bullets. Hermetic — no network. |
| `tests/unit/conftest.py` | maybe modify | Add `chunking_fixture_dir` fixture if a shared path helper is useful (skip if simple). |
| `docs/CONTRACTS.md` | modify | Add `Chunk` / `ParentChunk` / `ChildChunk` / `ChunkMetadata` / `ChunkSet` / `ChunkValidationError` to the contracts catalog. |

---

## Task 1 — Add dependencies + version pins (ADR D3)

**Files:**
- Modify: `pyproject.toml` (the `dependencies` list, alphabetic insertion).

- [ ] **Step 1.1: Add the deps via uv**

Run:
```bash
uv add 'unstructured[html,pdf]==0.18.0' 'tiktoken==0.9.0'
```

Expected: `pyproject.toml` updated with pins; `uv.lock` regenerated.

If the exact pinned version is unavailable on PyPI at run-time, take the latest patch in the same minor and record the exact pin in the commit message. Pin must be `==`, not `>=` (ADR D3).

- [ ] **Step 1.2: Verify import works**

Run:
```bash
uv run python -c "from unstructured.partition.html import partition_html; import tiktoken; print('ok')"
```

Expected stdout: `ok`.

- [ ] **Step 1.3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(extract): pin unstructured[html,pdf] + tiktoken for chunking (#13, ADR D3)"
```

---

## Task 2 — Acquire the real 10-K fixture

**Files:**
- Create: `tests/fixtures/chunking/sample_10k.htm` (binary asset, ~1.2 MB).
- Create: `tests/fixtures/chunking/sample_10k.meta.json` (the metadata sidecar used by tests).

- [ ] **Step 2.1: Download CRDO FY2024 10-K HTML**

Run (one-off; not part of the test suite):
```bash
mkdir -p tests/fixtures/chunking
curl -sS -A "auto-research/0.1 sam@example.com" \
  -o tests/fixtures/chunking/sample_10k.htm \
  "https://www.sec.gov/Archives/edgar/data/1860879/000162828024029554/crdo-20240427.htm"
```

(CRDO 10-K filed 2024-06-27 for fiscal year ending 2024-04-27. Public SEC filing, no copyright concern. If the URL has rotated, run `uv run python -c "from auto_research.ingest.edgar import EdgarClient; ..."` to refetch via the project's EDGAR client — the live-smoke pattern.)

- [ ] **Step 2.2: Verify size + content**

Run:
```bash
ls -l tests/fixtures/chunking/sample_10k.htm
grep -c "Item 1A" tests/fixtures/chunking/sample_10k.htm
grep -c "Item 7" tests/fixtures/chunking/sample_10k.htm
grep -c "Item 8" tests/fixtures/chunking/sample_10k.htm
```

Expected: file size 800KB–2MB. Each `grep -c` returns a positive integer (the section markers exist).

- [ ] **Step 2.3: Write the metadata sidecar**

Create `tests/fixtures/chunking/sample_10k.meta.json`:

```json
{
  "ticker": "CRDO",
  "filing_date": "2024-06-27",
  "fiscal_period": "FY2024",
  "doc_type": "10-K",
  "doc_id": "0001628280-24-029554",
  "expected_sections": ["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"]
}
```

- [ ] **Step 2.4: Commit**

```bash
git add tests/fixtures/chunking/sample_10k.htm tests/fixtures/chunking/sample_10k.meta.json
git commit -m "test(extract): add CRDO FY2024 10-K fixture for chunking tests (#13)"
```

---

## Task 3 — Dataclasses + exceptions (TDD red)

**Files:**
- Create: `src/auto_research/extract/chunking.py` (initial stub with dataclasses only).
- Create: `tests/unit/test_chunking.py`.

- [ ] **Step 3.1: Write failing tests for dataclass shapes**

Create `tests/unit/test_chunking.py`:

```python
"""Unit tests for src/auto_research/extract/chunking.py (Issue #13).

Hermetic — no network. Real 10-K fixture loaded from
tests/fixtures/chunking/sample_10k.htm.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from auto_research.extract.chunking import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ChunkValidationError,
    ParentChunk,
    SINGLE_SHOT_TOKEN_CUTOFF,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chunking"


@pytest.fixture(scope="session")
def sample_10k_html() -> str:
    return (FIXTURE_DIR / "sample_10k.htm").read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="session")
def sample_10k_metadata() -> ChunkMetadata:
    meta = json.loads((FIXTURE_DIR / "sample_10k.meta.json").read_text())
    return ChunkMetadata(
        ticker=meta["ticker"],
        filing_date=date.fromisoformat(meta["filing_date"]),
        fiscal_period=meta["fiscal_period"],
        doc_type=meta["doc_type"],
        doc_id=meta["doc_id"],
    )


def test_chunk_metadata_is_frozen():
    meta = ChunkMetadata(
        ticker="CRDO",
        filing_date=date(2024, 6, 27),
        fiscal_period="FY2024",
        doc_type="10-K",
        doc_id="0001628280-24-029554",
    )
    with pytest.raises((AttributeError, Exception)):
        meta.ticker = "NVDA"  # type: ignore[misc]


def test_parent_chunk_is_frozen_and_carries_metadata(sample_10k_metadata):
    parent = ParentChunk(
        text="hello",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    assert parent.metadata.ticker == "CRDO"
    assert parent.table_html is None
    with pytest.raises((AttributeError, Exception)):
        parent.text = "x"  # type: ignore[misc]


def test_child_chunk_carries_parent_id_and_metadata(sample_10k_metadata):
    child = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id="doc-id::0-5",
        metadata=sample_10k_metadata,
    )
    assert child.parent_id == "doc-id::0-5"


def test_chunkset_groups_parents_and_children(sample_10k_metadata):
    parent = ParentChunk(
        text="hello world",
        section_name="Item 1A",
        char_span=(0, 11),
        token_count=2,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    child = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id=f"{sample_10k_metadata.doc_id}::0-11",
        metadata=sample_10k_metadata,
    )
    cs = ChunkSet(parents=[parent], children=[child])
    assert len(cs.parents) == 1 and len(cs.children) == 1


def test_chunk_validation_error_is_value_error():
    assert issubclass(ChunkValidationError, ValueError)


def test_single_shot_cutoff_is_named_constant():
    assert isinstance(SINGLE_SHOT_TOKEN_CUTOFF, int)
    assert SINGLE_SHOT_TOKEN_CUTOFF == 100_000
```

- [ ] **Step 3.2: Run to verify failure**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: `ImportError: cannot import name 'ChunkMetadata' from 'auto_research.extract.chunking'` (the module doesn't exist yet).

- [ ] **Step 3.3: Implement minimal dataclasses to make tests pass**

Create `src/auto_research/extract/chunking.py`:

```python
"""Section-aware SEC filing parsing + chunking (Issue #13).

Public surface:

- `parse_filing(html, metadata) -> ChunkSet` — pure function, no network.
- `ParentChunk` — ≤4K-token context unit, section-respecting.
- `ChildChunk` — ~200–800-token retrieval unit, never crosses a parent.
- `ChunkValidationError` — raised when char_span fidelity (INV-2) fails;
  caller is responsible for quarantine routing.

Library-first: parsing uses `unstructured.partition.html.partition_html`
(OSS, Apache 2.0). Token counting uses `tiktoken` cl100k_base. No network
calls; the test suite asserts this via a no-socket fixture.

Tables (10-K Item 8 audited financial statements) emit as summary
`ParentChunk`s with the raw HTML attached on `table_html`, bypassing
dense retrieval. The 10-K worker (Issue #19) reads `table_html`
directly via a typed Pydantic schema. See
`docs/decisions/2026-05-24-rag-enhancements.md` D5.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

SINGLE_SHOT_TOKEN_CUTOFF: Final[int] = 100_000


class ChunkValidationError(ValueError):  # noqa: N818  # typed contract name
    """Raised when a chunk's char_span doesn't slice back to its text.

    Mirrors `CitationMismatch` in `extract/guardrails.py`. Subclasses
    `ValueError` so generic callers can catch on type, but typed so
    workers can route to the chunking-specific quarantine path without
    masking unrelated `ValueError`s.
    """


@dataclass(frozen=True)
class ChunkMetadata:
    ticker: str
    filing_date: date
    fiscal_period: str
    doc_type: str
    doc_id: str


@dataclass(frozen=True)
class ParentChunk:
    text: str
    section_name: str
    char_span: tuple[int, int]
    token_count: int
    table_html: str | None
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ChildChunk:
    text: str
    char_span: tuple[int, int]
    token_count: int
    parent_id: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ChunkSet:
    parents: list[ParentChunk]
    children: list[ChildChunk]
```

- [ ] **Step 3.4: Re-run tests, verify green**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 6 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/auto_research/extract/chunking.py tests/unit/test_chunking.py
git commit -m "feat(extract): chunking dataclasses + ChunkValidationError (#13, ADR D7)"
```

---

## Task 4 — Section detection on the real fixture (TDD red)

**Files:**
- Modify: `tests/unit/test_chunking.py` (append).
- Modify: `src/auto_research/extract/chunking.py`.

- [ ] **Step 4.1: Add failing test**

Append to `tests/unit/test_chunking.py`:

```python
from auto_research.extract.chunking import detect_sections


def test_detect_sections_finds_all_10k_items(sample_10k_html, sample_10k_metadata):
    sections = detect_sections(sample_10k_html)
    section_names = [s.name for s in sections]
    # Order in the doc — Item 1, 1A, 7, 7A, 8 (others may be present).
    for required in ["Item 1A", "Item 7", "Item 8"]:
        assert required in section_names, f"missing {required}: got {section_names}"


def test_detect_sections_returns_char_spans_into_source(sample_10k_html):
    sections = detect_sections(sample_10k_html)
    for s in sections:
        start, end = s.char_span
        assert 0 <= start < end <= len(sample_10k_html)
        # Section name appears within the first 200 chars of its span (the header).
        head = sample_10k_html[start : start + 200].lower()
        assert s.name.lower().replace("item ", "item") in head.replace(" ", "")
```

- [ ] **Step 4.2: Run to verify failure**

Run: `uv run pytest tests/unit/test_chunking.py::test_detect_sections_finds_all_10k_items -x -q`

Expected: `ImportError` for `detect_sections`.

- [ ] **Step 4.3: Implement `detect_sections`**

Append to `src/auto_research/extract/chunking.py`:

```python
from collections.abc import Iterable
import re

from unstructured.documents.elements import Element, NarrativeText, Table, Title
from unstructured.partition.html import partition_html

# SEC 10-K section headers we treat as parent-chunk boundaries. The
# pattern is `Item <num>[<letter>]` allowing dot, dash, em-dash, or space
# after the number, then any text. Anchored at start-of-line because
# table-of-contents references typically appear mid-paragraph.
_SECTION_HEADER = re.compile(
    r"^\s*item\s+(\d+[a-z]?)\b[\s.\-–—:]*",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class _DetectedSection:
    name: str            # canonical "Item 1A"
    char_span: tuple[int, int]  # span into the raw source_text


def detect_sections(html: str) -> list[_DetectedSection]:
    """Detect SEC 10-K Item sections in raw HTML.

    Returns sections in document order with char_span pointing at the
    section's slice of `html`. The last section runs to end-of-document.

    Detection is regex-based on the rendered text representation, not
    on HTML structure — companies vary how they wrap `Item 1A.`
    (sometimes a heading, sometimes inline bold within a paragraph),
    and the regex on the plain text catches both.
    """
    sections: list[_DetectedSection] = []
    matches = list(_SECTION_HEADER.finditer(html))
    if not matches:
        return sections

    # Filter out table-of-contents echoes: a section header is "real"
    # only if substantial body text follows it before the next header.
    # The first match for each Item number is the TOC; the second is
    # the actual section start. Group by Item number, keep the last
    # occurrence (which is the actual content section, not the TOC
    # back-reference). For 10-Ks the TOC always appears first, so
    # last-occurrence selection is safe.
    by_num: dict[str, re.Match[str]] = {}
    for m in matches:
        num = m.group(1).upper()
        by_num[num] = m  # later occurrences overwrite earlier

    ordered = sorted(by_num.values(), key=lambda m: m.start())
    for i, m in enumerate(ordered):
        start = m.start()
        end = ordered[i + 1].start() if i + 1 < len(ordered) else len(html)
        name = f"Item {m.group(1).upper()}"
        sections.append(_DetectedSection(name=name, char_span=(start, end)))
    return sections
```

- [ ] **Step 4.4: Re-run, verify green**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 8 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/auto_research/extract/chunking.py tests/unit/test_chunking.py
git commit -m "feat(extract): section detection for 10-K Item boundaries (#13)"
```

---

## Task 5 — Parent-chunk packing (token-bounded, section-respecting) (TDD red)

**Files:**
- Modify: `tests/unit/test_chunking.py`.
- Modify: `src/auto_research/extract/chunking.py`.

- [ ] **Step 5.1: Add failing test**

Append to `tests/unit/test_chunking.py`:

```python
from auto_research.extract.chunking import (
    MAX_PARENT_TOKENS,
    _pack_parents_for_section,
    count_tokens,
)


def test_count_tokens_uses_cl100k_base():
    # "hello world" is 2 tokens in cl100k_base.
    assert count_tokens("hello world") == 2


def test_pack_parents_respects_max_tokens(sample_10k_metadata):
    text = ("Risk factor sentence. " * 5000)  # ~25K tokens worth
    parents = _pack_parents_for_section(
        section_name="Item 1A",
        section_text=text,
        section_offset=0,
        metadata=sample_10k_metadata,
    )
    for p in parents:
        assert p.token_count <= MAX_PARENT_TOKENS
        assert p.section_name == "Item 1A"


def test_pack_parents_covers_whole_section(sample_10k_metadata):
    text = ("Sentence A. " * 200)  # small, fits in one chunk
    parents = _pack_parents_for_section(
        section_name="Item 1A",
        section_text=text,
        section_offset=100,
        metadata=sample_10k_metadata,
    )
    # Spans cover the whole section without gaps.
    spans = sorted([p.char_span for p in parents])
    assert spans[0][0] == 100
    assert spans[-1][1] == 100 + len(text)
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c, f"gap or overlap: {b} != {c}"
```

- [ ] **Step 5.2: Run, verify failure**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: ImportError for `MAX_PARENT_TOKENS` / `_pack_parents_for_section` / `count_tokens`.

- [ ] **Step 5.3: Implement packer + token counter**

Append to `src/auto_research/extract/chunking.py`:

```python
import tiktoken

MAX_PARENT_TOKENS: Final[int] = 4_000

# Module-level encoder. cl100k_base matches Claude's tokenizer family
# closely enough for chunk-size budgeting (Anthropic does not publish a
# stable public tokenizer; cl100k is the closest open proxy used across
# the ecosystem).
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _pack_parents_for_section(
    *,
    section_name: str,
    section_text: str,
    section_offset: int,
    metadata: ChunkMetadata,
) -> list[ParentChunk]:
    """Greedy-pack `section_text` into `ParentChunk`s of ≤ MAX_PARENT_TOKENS.

    `section_offset` is the absolute char offset of `section_text` in the
    raw source HTML — chunk `char_span`s are absolute (so the INV-2 check
    `source_text[span] == chunk.text` works).

    Packing is paragraph-aware: splits on double-newline / blank-line
    boundaries first, falls back to sentence-ish splits ('. ') only when
    a single paragraph exceeds the budget. This keeps related sentences
    together for better extraction context.
    """
    if not section_text:
        return []

    # Paragraph split — split on runs of whitespace containing >=2 newlines.
    paragraphs: list[tuple[int, str]] = []  # (offset_within_section, paragraph_text)
    pos = 0
    for m in re.finditer(r"\n\s*\n", section_text):
        para = section_text[pos : m.start()]
        if para:
            paragraphs.append((pos, para))
        pos = m.end()
    if pos < len(section_text):
        paragraphs.append((pos, section_text[pos:]))
    if not paragraphs:
        paragraphs = [(0, section_text)]

    chunks: list[ParentChunk] = []
    buf_offset: int | None = None
    buf_end: int = 0
    buf_tokens: int = 0
    buf_parts: list[str] = []

    def flush() -> None:
        nonlocal buf_offset, buf_end, buf_tokens, buf_parts
        if buf_offset is None or not buf_parts:
            return
        text = section_text[buf_offset:buf_end]
        chunks.append(
            ParentChunk(
                text=text,
                section_name=section_name,
                char_span=(section_offset + buf_offset, section_offset + buf_end),
                token_count=count_tokens(text),
                table_html=None,
                metadata=metadata,
            )
        )
        buf_offset = None
        buf_end = 0
        buf_tokens = 0
        buf_parts = []

    for off, para in paragraphs:
        para_tokens = count_tokens(para)
        if para_tokens > MAX_PARENT_TOKENS:
            # Single paragraph exceeds budget — flush current buffer, then
            # split this paragraph on '. ' to subdivide.
            flush()
            sentence_pos = off
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if not sent:
                    continue
                stoks = count_tokens(sent)
                if buf_tokens + stoks > MAX_PARENT_TOKENS and buf_parts:
                    flush()
                if buf_offset is None:
                    buf_offset = sentence_pos
                buf_end = sentence_pos + len(sent)
                buf_tokens += stoks
                buf_parts.append(sent)
                sentence_pos = buf_end
                # Advance over the whitespace consumed by the split.
                while (
                    sentence_pos < off + len(para)
                    and section_text[buf_offset + (sentence_pos - buf_offset)].isspace()
                ):
                    sentence_pos += 1
                buf_end = sentence_pos
            flush()
        else:
            if buf_tokens + para_tokens > MAX_PARENT_TOKENS and buf_parts:
                flush()
            if buf_offset is None:
                buf_offset = off
            buf_end = off + len(para)
            buf_tokens += para_tokens
            buf_parts.append(para)

    flush()

    # Cover any trailing whitespace by extending the last chunk's end to
    # the section boundary, so spans tile the section without gaps.
    if chunks and chunks[-1].char_span[1] < section_offset + len(section_text):
        last = chunks[-1]
        new_end = section_offset + len(section_text)
        # Replace by reconstructing (frozen dataclass).
        chunks[-1] = ParentChunk(
            text=section_text[last.char_span[0] - section_offset : new_end - section_offset],
            section_name=last.section_name,
            char_span=(last.char_span[0], new_end),
            token_count=count_tokens(
                section_text[last.char_span[0] - section_offset : new_end - section_offset]
            ),
            table_html=None,
            metadata=last.metadata,
        )

    return chunks
```

- [ ] **Step 5.4: Re-run, verify green**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 11 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/auto_research/extract/chunking.py tests/unit/test_chunking.py
git commit -m "feat(extract): paragraph-aware parent-chunk packing (≤4K tokens) (#13)"
```

---

## Task 6 — Child-chunk subdivision (TDD red)

**Files:**
- Modify: `tests/unit/test_chunking.py`.
- Modify: `src/auto_research/extract/chunking.py`.

- [ ] **Step 6.1: Add failing test**

Append to `tests/unit/test_chunking.py`:

```python
from auto_research.extract.chunking import (
    MAX_CHILD_TOKENS,
    MIN_CHILD_TOKENS,
    subdivide_to_children,
)


def test_subdivide_yields_children_within_token_band(sample_10k_metadata):
    parent_text = "Sentence one. " * 400  # ~1200 tokens worth
    parent = ParentChunk(
        text=parent_text,
        section_name="Item 1A",
        char_span=(0, len(parent_text)),
        token_count=count_tokens(parent_text),
        table_html=None,
        metadata=sample_10k_metadata,
    )
    children = subdivide_to_children(parent)
    for c in children:
        assert MIN_CHILD_TOKENS <= c.token_count <= MAX_CHILD_TOKENS
        assert c.parent_id == f"{sample_10k_metadata.doc_id}::0-{len(parent_text)}"


def test_child_spans_are_subsets_of_parent_span(sample_10k_metadata):
    parent_text = "Sentence one. " * 200
    parent = ParentChunk(
        text=parent_text,
        section_name="Item 1A",
        char_span=(500, 500 + len(parent_text)),
        token_count=count_tokens(parent_text),
        table_html=None,
        metadata=sample_10k_metadata,
    )
    children = subdivide_to_children(parent)
    for c in children:
        assert parent.char_span[0] <= c.char_span[0] < c.char_span[1] <= parent.char_span[1]


def test_subdivide_handles_short_parent_below_min(sample_10k_metadata):
    short = "Short text."
    parent = ParentChunk(
        text=short,
        section_name="Item 1A",
        char_span=(0, len(short)),
        token_count=count_tokens(short),
        table_html=None,
        metadata=sample_10k_metadata,
    )
    children = subdivide_to_children(parent)
    # Short parents yield a single child equal to the parent, even if below MIN.
    assert len(children) == 1
    assert children[0].char_span == parent.char_span
```

- [ ] **Step 6.2: Run, verify failure**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: ImportError for `MIN_CHILD_TOKENS` / `MAX_CHILD_TOKENS` / `subdivide_to_children`.

- [ ] **Step 6.3: Implement subdivider**

Append to `src/auto_research/extract/chunking.py`:

```python
MIN_CHILD_TOKENS: Final[int] = 200
MAX_CHILD_TOKENS: Final[int] = 800


def _parent_id(parent: ParentChunk) -> str:
    return f"{parent.metadata.doc_id}::{parent.char_span[0]}-{parent.char_span[1]}"


def subdivide_to_children(parent: ParentChunk) -> list[ChildChunk]:
    """Sentence-window subdivide a parent into 200-800-token children.

    Children never cross the parent boundary. Char_spans are absolute
    (parent.char_span[0]-relative offsets are translated back to source).

    Short parents (below MIN_CHILD_TOKENS) yield a single child equal to
    the parent — degenerate but contract-safe (the citation walker
    doesn't care about size, only fidelity).
    """
    if parent.token_count <= MAX_CHILD_TOKENS:
        # Whole parent is one child.
        return [
            ChildChunk(
                text=parent.text,
                char_span=parent.char_span,
                token_count=parent.token_count,
                parent_id=_parent_id(parent),
                metadata=parent.metadata,
            )
        ]

    children: list[ChildChunk] = []
    sentences = list(re.finditer(r"[^.!?]+[.!?]+\s*|\S+\s*$", parent.text))
    if not sentences:
        return [
            ChildChunk(
                text=parent.text,
                char_span=parent.char_span,
                token_count=parent.token_count,
                parent_id=_parent_id(parent),
                metadata=parent.metadata,
            )
        ]

    buf_start: int | None = None
    buf_end: int = 0
    buf_tokens: int = 0

    def flush() -> None:
        nonlocal buf_start, buf_end, buf_tokens
        if buf_start is None or buf_tokens == 0:
            return
        text = parent.text[buf_start:buf_end]
        abs_start = parent.char_span[0] + buf_start
        abs_end = parent.char_span[0] + buf_end
        children.append(
            ChildChunk(
                text=text,
                char_span=(abs_start, abs_end),
                token_count=count_tokens(text),
                parent_id=_parent_id(parent),
                metadata=parent.metadata,
            )
        )
        buf_start = None
        buf_end = 0
        buf_tokens = 0

    for m in sentences:
        s_start = m.start()
        s_end = m.end()
        s_text = parent.text[s_start:s_end]
        s_tokens = count_tokens(s_text)
        if buf_start is None:
            buf_start = s_start
        # Force-flush if adding this sentence would exceed MAX.
        if buf_tokens + s_tokens > MAX_CHILD_TOKENS and buf_tokens >= MIN_CHILD_TOKENS:
            flush()
            buf_start = s_start
        buf_end = s_end
        buf_tokens += s_tokens
        # Flush if we're past MIN and at or above MAX after appending.
        if buf_tokens >= MAX_CHILD_TOKENS:
            flush()

    flush()
    return children
```

- [ ] **Step 6.4: Re-run, verify green**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 14 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/auto_research/extract/chunking.py tests/unit/test_chunking.py
git commit -m "feat(extract): child-chunk sentence-window subdivision (200-800 tokens) (#13, ADR D4)"
```

---

## Task 7 — char_span validation + HTML edge cases (TDD red; Tier 2 evidence)

**Files:**
- Create: `tests/fixtures/chunking/edge_cases.htm`.
- Modify: `tests/unit/test_chunking.py`.
- Modify: `src/auto_research/extract/chunking.py`.

- [ ] **Step 7.1: Create the edge-case fixture**

Create `tests/fixtures/chunking/edge_cases.htm`:

```html
<html><body>
<h1>Item 1A. Risk Factors</h1>
<p>This paragraph has a non-breaking&nbsp;space and a curly&#8217;apostrophe.</p>
<p>Nested <span><span>span</span></span> elements should preserve offsets.</p>
<p><![CDATA[CDATA payload with raw text.]]></p>
<h1>Item 7. Management Discussion</h1>
<p>Trailing section content.</p>
</body></html>
```

- [ ] **Step 7.2: Add failing tests**

Append to `tests/unit/test_chunking.py`:

```python
from auto_research.extract.chunking import (
    validate_char_spans,
)


def test_validate_char_spans_passes_when_aligned(sample_10k_metadata):
    src = "hello world"
    parent = ParentChunk(
        text="hello",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    validate_char_spans(src, [parent], [])  # raises on mismatch — should not raise


def test_validate_char_spans_raises_on_text_mismatch(sample_10k_metadata):
    src = "hello world"
    bad = ParentChunk(
        text="HELLO",  # uppercase ≠ source slice "hello"
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    with pytest.raises(ChunkValidationError) as exc_info:
        validate_char_spans(src, [bad], [])
    msg = str(exc_info.value)
    assert "Item 1A" in msg
    assert "(0, 5)" in msg


def test_validate_char_spans_raises_on_out_of_bounds(sample_10k_metadata):
    src = "short"
    bad = ParentChunk(
        text="overflow",
        section_name="Item 1A",
        char_span=(0, 999),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    with pytest.raises(ChunkValidationError):
        validate_char_spans(src, [bad], [])


def test_validate_char_spans_walks_children(sample_10k_metadata):
    src = "hello world"
    parent = ParentChunk(
        text="hello world",
        section_name="Item 1A",
        char_span=(0, 11),
        token_count=2,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    child_ok = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id=f"{sample_10k_metadata.doc_id}::0-11",
        metadata=sample_10k_metadata,
    )
    child_bad = ChildChunk(
        text="WORLD",
        char_span=(6, 11),
        token_count=1,
        parent_id=f"{sample_10k_metadata.doc_id}::0-11",
        metadata=sample_10k_metadata,
    )
    validate_char_spans(src, [parent], [child_ok])  # ok
    with pytest.raises(ChunkValidationError):
        validate_char_spans(src, [parent], [child_bad])


def test_parse_filing_preserves_char_span_through_html_edge_cases(sample_10k_metadata):
    from auto_research.extract.chunking import parse_filing  # forward ref

    html = (FIXTURE_DIR / "edge_cases.htm").read_text(encoding="utf-8")
    result = parse_filing(html=html, metadata=sample_10k_metadata)
    # Every parent's char_span must slice back to its text in the raw source.
    for p in result.parents:
        assert html[p.char_span[0] : p.char_span[1]] == p.text, p.section_name
    for c in result.children:
        assert html[c.char_span[0] : c.char_span[1]] == c.text
```

- [ ] **Step 7.3: Run, verify failures**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: ImportError for `validate_char_spans` / `parse_filing`.

- [ ] **Step 7.4: Implement `validate_char_spans`**

Append to `src/auto_research/extract/chunking.py`:

```python
def validate_char_spans(
    source_text: str,
    parents: Iterable[ParentChunk],
    children: Iterable[ChildChunk],
) -> None:
    """Assert every chunk's char_span slices to its text in source_text.

    This is the INV-2 contract for chunking, mirroring `guardrails.
    _walk_citations` for extraction. Raises `ChunkValidationError` on
    first mismatch — workers catch and route to quarantine via
    `quarantine_chunkset`.
    """
    for p in parents:
        a, b = p.char_span
        if a < 0 or b > len(source_text) or a >= b:
            raise ChunkValidationError(
                f"parent {p.section_name} span out of bounds: ({a}, {b}) vs source len {len(source_text)}"
            )
        sliced = source_text[a:b]
        if sliced != p.text:
            raise ChunkValidationError(
                f"parent {p.section_name} text mismatch at ({a}, {b}): "
                f"source[{a}:{b}]={sliced[:80]!r} vs chunk.text={p.text[:80]!r}"
            )
    for c in children:
        a, b = c.char_span
        if a < 0 or b > len(source_text) or a >= b:
            raise ChunkValidationError(
                f"child {c.parent_id} span out of bounds: ({a}, {b})"
            )
        sliced = source_text[a:b]
        if sliced != c.text:
            raise ChunkValidationError(
                f"child {c.parent_id} text mismatch at ({a}, {b}): "
                f"source[{a}:{b}]={sliced[:80]!r} vs chunk.text={c.text[:80]!r}"
            )
```

- [ ] **Step 7.5: Implement minimal `parse_filing` to make the edge-case test pass**

Append to `src/auto_research/extract/chunking.py`:

```python
def _text_with_offsets_from_html(html: str) -> str:
    """The 'source text' for char_span fidelity IS the raw HTML.

    INV-2 holds against the input we received. Any post-parse text
    normalization (whitespace collapse, HTML-entity decode) breaks the
    span contract; we therefore use the raw HTML as the canonical
    `source_text` and chunk by finding offsets of `unstructured`'s
    extracted text fragments within that raw HTML.

    The function exists as a named identity to make the intent grep-able.
    """
    return html


def _find_offset(haystack: str, needle: str, start: int = 0) -> int | None:
    """Locate `needle` in `haystack` starting from `start`.

    `unstructured` may collapse whitespace inside the extracted text;
    when an exact match fails, we retry with whitespace-flexible regex
    against the raw HTML. Same discipline as
    `workers/s_filings._resolve_span` (whitespace-flexible match against
    raw doc to handle LLM/parser-collapsed quotes).
    """
    if not needle.strip():
        return None
    idx = haystack.find(needle, start)
    if idx != -1:
        return idx
    # Whitespace-flexible fallback.
    flex = re.compile(r"\s+".join(re.escape(w) for w in needle.split()), re.DOTALL)
    m = flex.search(haystack, start)
    return m.start() if m else None


def parse_filing(*, html: str, metadata: ChunkMetadata) -> ChunkSet:
    """Parse SEC filing HTML into a ChunkSet of parents + children.

    Pure function: same `(html, metadata)` → same `ChunkSet`. No network,
    no LLM. Raises `ChunkValidationError` on char_span fidelity failure;
    callers route to `quarantine_chunkset` per the INV-2 quarantine
    pattern.
    """
    source_text = _text_with_offsets_from_html(html)

    elements = partition_html(text=html)

    sections = detect_sections(source_text)
    if not sections:
        # Whole document as one section.
        sections = [_DetectedSection(name="Body", char_span=(0, len(source_text)))]

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for s in sections:
        sec_text = source_text[s.char_span[0] : s.char_span[1]]

        # Tables in this section → emit as summary chunks (ADR D5).
        for el in _elements_in_span(elements, html, s.char_span):
            if isinstance(el, Table):
                table_html = getattr(el.metadata, "text_as_html", None) or str(el)
                # Locate the table's HTML in the raw doc to get char_span.
                tbl_offset = _find_offset(source_text, table_html, s.char_span[0])
                if tbl_offset is None or tbl_offset >= s.char_span[1]:
                    # Fall back to element text (less precise but kept in stream).
                    tbl_text = str(el)
                    tbl_offset = _find_offset(source_text, tbl_text, s.char_span[0])
                    if tbl_offset is None:
                        continue
                    span_end = tbl_offset + len(tbl_text)
                else:
                    span_end = tbl_offset + len(table_html)
                summary = _summarize_table(el)
                parents.append(
                    ParentChunk(
                        text=source_text[tbl_offset:span_end],
                        section_name=s.name,
                        char_span=(tbl_offset, span_end),
                        token_count=count_tokens(source_text[tbl_offset:span_end]),
                        table_html=table_html if isinstance(table_html, str) else None,
                        metadata=metadata,
                    )
                )

        # Narrative parent-packing for this section.
        narrative_parents = _pack_parents_for_section(
            section_name=s.name,
            section_text=sec_text,
            section_offset=s.char_span[0],
            metadata=metadata,
        )
        parents.extend(narrative_parents)

        for p in narrative_parents:
            children.extend(subdivide_to_children(p))

    validate_char_spans(source_text, parents, children)
    return ChunkSet(parents=parents, children=children)


def _summarize_table(el: Table) -> str:
    """One-line summary stub for an Item 8 table.

    Real summaries (with row/col counts, statement type) are produced by
    the 10-K worker in Issue #19; this placeholder keeps chunking pure
    (no LLM call here).
    """
    return f"[Table] {(str(el)[:120]).strip()}"


def _elements_in_span(
    elements: Iterable[Element], html: str, span: tuple[int, int]
) -> Iterable[Element]:
    """Yield elements whose text appears within `span` of `html`.

    `unstructured`'s elements don't always carry source offsets. We
    locate each by string search in the raw HTML; elements whose text
    isn't found in the span are dropped (consistent with `unstructured`'s
    own behavior when HTML normalization moves a fragment).
    """
    for el in elements:
        text = str(el)
        if not text.strip():
            continue
        off = _find_offset(html, text, span[0])
        if off is None or off >= span[1]:
            continue
        yield el
```

- [ ] **Step 7.6: Re-run, verify green**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 19 passed.

- [ ] **Step 7.7: Commit**

```bash
git add src/auto_research/extract/chunking.py tests/unit/test_chunking.py tests/fixtures/chunking/edge_cases.htm
git commit -m "feat(extract): char_span validation + HTML edge-case fidelity (#13, INV-2, ADR D9)"
```

---

## Task 8 — Table policy on 10-K Item 8 (TDD red; ADR D5)

**Files:**
- Modify: `tests/unit/test_chunking.py`.
- (Implementation already present in Task 7; this task is the explicit AC test.)

- [ ] **Step 8.1: Add failing test for table emission**

Append to `tests/unit/test_chunking.py`:

```python
from auto_research.extract.chunking import parse_filing


def test_parse_filing_emits_tables_with_table_html(sample_10k_html, sample_10k_metadata):
    result = parse_filing(html=sample_10k_html, metadata=sample_10k_metadata)
    item_8_parents = [p for p in result.parents if p.section_name == "Item 8"]
    assert item_8_parents, "expected at least one Item 8 parent chunk"
    table_chunks = [p for p in item_8_parents if p.table_html is not None]
    assert table_chunks, "expected ≥1 Item 8 parent with table_html attached"


def test_non_table_chunks_have_no_table_html(sample_10k_html, sample_10k_metadata):
    result = parse_filing(html=sample_10k_html, metadata=sample_10k_metadata)
    item_1a = [p for p in result.parents if p.section_name == "Item 1A"]
    assert item_1a
    for p in item_1a:
        assert p.table_html is None, f"Item 1A chunk should not carry table_html: {p.text[:80]!r}"


def test_table_html_is_pandas_readable(sample_10k_html, sample_10k_metadata):
    import io

    import pandas as pd  # already in deps

    result = parse_filing(html=sample_10k_html, metadata=sample_10k_metadata)
    tables = [p for p in result.parents if p.table_html is not None]
    assert tables
    # At least one table parses cleanly with pandas — the contract Issue #19's
    # 10-K worker relies on.
    parsed_any = False
    for t in tables:
        try:
            dfs = pd.read_html(io.StringIO(t.table_html))
        except ValueError:
            continue
        if dfs:
            parsed_any = True
            break
    assert parsed_any, "no Item 8 table_html parsed via pandas.read_html"
```

- [ ] **Step 8.2: Run, verify pass**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 22 passed.

(If `test_table_html_is_pandas_readable` fails because `unstructured`'s `text_as_html` is missing for this fixture, the implementation in Task 7 needs to fall back to a different source for `table_html` — investigate `el.metadata.text_as_html` vs. extracting `<table>` HTML by character offset from the raw source.)

- [ ] **Step 8.3: Commit**

```bash
git add tests/unit/test_chunking.py
git commit -m "test(extract): assert Item 8 table_html policy + pandas readability (#13, ADR D5)"
```

---

## Task 9 — Quarantine routing on validation failure (TDD red)

**Files:**
- Modify: `tests/unit/test_chunking.py`.
- Modify: `src/auto_research/extract/chunking.py`.

- [ ] **Step 9.1: Add failing test**

Append to `tests/unit/test_chunking.py`:

```python
from auto_research.extract.chunking import quarantine_chunkset


def test_quarantine_writes_json_with_reason(tmp_path, sample_10k_metadata):
    src = "hello world"
    bad = ParentChunk(
        text="WRONG",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    chunkset = ChunkSet(parents=[bad], children=[])
    quarantine_chunkset(
        chunkset,
        source_text=src,
        reason="char_span mismatch",
        quarantine_root=tmp_path,
    )
    out = tmp_path / "chunking" / f"{sample_10k_metadata.doc_id}.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["reason"] == "char_span mismatch"
    assert data["doc_id"] == sample_10k_metadata.doc_id
    assert data["parents"][0]["section_name"] == "Item 1A"


def test_quarantine_is_atomic(tmp_path, sample_10k_metadata):
    # No half-written file: the rename-into-place pattern from _io should hold.
    chunkset = ChunkSet(parents=[], children=[])
    quarantine_chunkset(
        chunkset,
        source_text="",
        reason="empty",
        quarantine_root=tmp_path,
    )
    # No tmp files left behind.
    tmpfiles = list((tmp_path / "chunking").glob(".*tmp*"))
    assert tmpfiles == []
```

- [ ] **Step 9.2: Run, verify failure**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: ImportError for `quarantine_chunkset`.

- [ ] **Step 9.3: Implement quarantine helper**

Append to `src/auto_research/extract/chunking.py`:

```python
import json
from datetime import UTC, datetime
from pathlib import Path

from auto_research._io import atomic_write_text

DEFAULT_QUARANTINE_ROOT: Final[Path] = Path("data/quarantine")


def quarantine_chunkset(
    chunkset: ChunkSet,
    *,
    source_text: str,
    reason: str,
    quarantine_root: Path = DEFAULT_QUARANTINE_ROOT,
) -> Path:
    """Write a chunking quarantine record under `<root>/chunking/<doc_id>.json`.

    Mirrors `guardrails.validate_or_quarantine`'s discipline: callers
    that get back a path (or a None return upstream) must NOT persist the
    chunkset to `data/extracted/` or Feast. The record captures the
    unmutated chunks + the source-text length so a human reviewer can
    reconstruct what `unstructured` produced.
    """
    if not chunkset.parents and not chunkset.children:
        doc_id = "empty"
    else:
        # All chunks share doc_id; take it from the first.
        first = chunkset.parents[0] if chunkset.parents else chunkset.children[0]
        doc_id = first.metadata.doc_id

    record = {
        "doc_id": doc_id,
        "reason": reason,
        "captured_at": datetime.now(UTC).isoformat(),
        "source_text_length": len(source_text),
        "parents": [
            {
                "section_name": p.section_name,
                "char_span": list(p.char_span),
                "token_count": p.token_count,
                "table_html_present": p.table_html is not None,
                "text_preview": p.text[:120],
            }
            for p in chunkset.parents
        ],
        "children": [
            {
                "parent_id": c.parent_id,
                "char_span": list(c.char_span),
                "token_count": c.token_count,
                "text_preview": c.text[:120],
            }
            for c in chunkset.children
        ],
    }
    dest = quarantine_root / "chunking" / f"{doc_id}.json"
    atomic_write_text(dest, json.dumps(record, indent=2, sort_keys=True))
    return dest
```

- [ ] **Step 9.4: Re-run, verify green**

Run: `uv run pytest tests/unit/test_chunking.py -x -q`

Expected: 24 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/auto_research/extract/chunking.py tests/unit/test_chunking.py
git commit -m "feat(extract): quarantine routing for chunking validation failures (#13, INV-2)"
```

---

## Task 10 — No-network guarantee (TDD red; AC bullet)

**Files:**
- Modify: `tests/unit/test_chunking.py`.

- [ ] **Step 10.1: Add no-socket test**

Append to `tests/unit/test_chunking.py`:

```python
def test_parse_filing_makes_no_network_calls(sample_10k_html, sample_10k_metadata, monkeypatch):
    """AC: parsing must not touch the network.

    Patches the socket module so any DNS or TCP attempt raises. If
    `unstructured` ever phones home, this test fires loudly.
    """
    import socket

    def _no_socket(*args, **kwargs):
        raise OSError("network access forbidden during chunking")

    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)
    monkeypatch.setattr(socket, "getaddrinfo", _no_socket)

    result = parse_filing(html=sample_10k_html, metadata=sample_10k_metadata)
    assert result.parents
```

- [ ] **Step 10.2: Run, verify pass (or diagnose `unstructured` network call)**

Run: `uv run pytest tests/unit/test_chunking.py::test_parse_filing_makes_no_network_calls -x -q`

Expected: PASS. If it fails (e.g., `unstructured` attempts to download an NLTK model on first use), pre-download the model in the test setup or pin the model path via env. The fix lives in conftest or a `tests/unit/conftest.py` autouse fixture; **do not** weaken the no-socket assertion.

- [ ] **Step 10.3: Commit**

```bash
git add tests/unit/test_chunking.py
git commit -m "test(extract): assert parse_filing is hermetic (no network) (#13)"
```

---

## Task 11 — Final pass: `make check` + doc-sync (TDD green; verification gate)

**Files:**
- Modify: `docs/CONTRACTS.md` (append chunking entities).

- [ ] **Step 11.1: Run `make quick`**

Run: `make quick`

Expected: ruff + mypy pass clean. Fix any issues — typically:
- Imports out of order (ruff `I`)
- `re.compile` patterns may need `re.Pattern[str]` typing.
- `_ENCODER` may need `Final[tiktoken.Encoding]` annotation.

- [ ] **Step 11.2: Run full unit suite**

Run: `make test`

Expected: all tests green (including pre-existing suites). Chunking tests should add ~24 tests.

- [ ] **Step 11.3: Update `docs/CONTRACTS.md`**

Find an appropriate location in `docs/CONTRACTS.md` to add chunking entities under the existing schema/contract catalog. Append a section like:

```markdown
### `extract.chunking` — Section-aware chunking contract (Issue #13)

| Type | Purpose |
|---|---|
| `ChunkMetadata` | Frozen dataclass — `(ticker, filing_date, fiscal_period, doc_type, doc_id)`. Required on every chunk for index-time filtering (ADR D7). |
| `ParentChunk` | ≤ 4K-token context unit. Section-respecting. `table_html` populated only for 10-K Item 8 tables (ADR D5). |
| `ChildChunk` | 200–800-token retrieval unit. `parent_id` references parent's `(doc_id, char_span)`. |
| `ChunkSet` | Result of `parse_filing(html, metadata)`. |
| `ChunkValidationError` | Raised on char_span mismatch (INV-2). Caller routes to `data/quarantine/chunking/<doc_id>.json` via `quarantine_chunkset`. |

`parse_filing` is pure: same input → same output, no network, no LLM.
Library-first via `unstructured[html,pdf]` (OSS Python lib).

See: `docs/decisions/2026-05-24-rag-enhancements.md` and Issue #13.
```

- [ ] **Step 11.4: Commit doc sync**

```bash
git add docs/CONTRACTS.md
git commit -m "docs(contracts): add chunking entities for Issue #13"
```

- [ ] **Step 11.5: Push branch**

Run:
```bash
git push -u origin feat/13-unstructured-chunking
```

---

## Task 12 — Open PR with Tier 2 evidence

- [ ] **Step 12.1: Open PR**

Run:
```bash
gh pr create --title "feat(extract): unstructured.io parsing + section-aware chunking (#13)" --body "$(cat <<'PRBODY'
Closes #13.

## Summary

Implements `extract/chunking.py` — pure-Python, deterministic SEC-filing parser producing parent + child chunks with metadata that honors INV-2.

Reflects decisions D1, D3, D4, D5, D7, D9 from
[`docs/decisions/2026-05-24-rag-enhancements.md`](./docs/decisions/2026-05-24-rag-enhancements.md):

- Library-first via the `unstructured` OSS Python lib (Apache 2.0). **Not** the unstructured.io SaaS.
- Two chunk levels — `ParentChunk` (≤4K tokens) for extraction context, `ChildChunk` (200-800 tokens) for retrieval embedding.
- Required metadata `(ticker, filing_date, fiscal_period, doc_type, doc_id)` on every chunk for downstream LanceDB index-time filtering.
- 10-K Item 8 `Table` elements emit as `ParentChunk(section_name="Item 8", table_html=<raw>)`; structured extraction (Issue #19) reads `table_html` directly, bypassing dense retrieval.
- char_span fidelity (`source_text[span] == chunk.text`) validated against `unstructured`'s HTML edge cases; failures route to `data/quarantine/chunking/<doc_id>.json`.

## Acceptance criteria

| AC | Evidence |
|---|---|
| Real 10-K fixture parses into sections | `tests/unit/test_chunking.py::test_detect_sections_finds_all_10k_items` |
| Parents ≤ 4K tokens, no parent spans a section break | `::test_pack_parents_respects_max_tokens`, `::test_pack_parents_covers_whole_section` |
| Child spans subset of parent spans; child tokens in [200, 800] | `::test_subdivide_yields_children_within_token_band`, `::test_child_spans_are_subsets_of_parent_span` |
| `source_text[char_span] == text` for every chunk (INV-2) | `::test_validate_char_spans_*`, `::test_parse_filing_preserves_char_span_through_html_edge_cases` |
| HTML edge cases (`&nbsp;`, `&#8217;`, nested `<span>`, CDATA) | `tests/fixtures/chunking/edge_cases.htm` + `::test_parse_filing_preserves_char_span_through_html_edge_cases` |
| Validation failures route to `data/quarantine/chunking/` | `::test_quarantine_writes_json_with_reason`, `::test_quarantine_is_atomic` |
| Metadata fields non-null on every chunk | `::test_parent_chunk_is_frozen_and_carries_metadata`, `::test_child_chunk_carries_parent_id_and_metadata` |
| Item 8 tables emit with `table_html` attached; non-table chunks have `None` | `::test_parse_filing_emits_tables_with_table_html`, `::test_non_table_chunks_have_no_table_html`, `::test_table_html_is_pandas_readable` |
| No network calls during parsing | `::test_parse_filing_makes_no_network_calls` |
| Versions pinned in `pyproject.toml` | git diff `pyproject.toml` |

## Change Contract

- **Tier:** 2 (touches INV-2 char_span contract — `docs/AI_WORKFLOW.md` §2 escalator: citation-grounding).
- **Problem:** SEC filings need section-aware chunking that downstream RAG (Issues #14-#18) and structured extraction (Issue #19) can consume without breaking the citation-grounding contract.
- **Scope:** `src/auto_research/extract/chunking.py` (new), `tests/unit/test_chunking.py` (new), `tests/fixtures/chunking/*` (new), `pyproject.toml` (deps), `docs/CONTRACTS.md` (catalog entry).
- **Invariants touched:** INV-2 (citation grounding) — strengthened via `validate_char_spans` runtime check + quarantine routing.
- **Verification:** Failing tests first; full `tests/unit/test_chunking.py` suite green; `make quick` clean. PR body cites named tests above.
- **Rollback:** `git revert` — no schema migrations, no consumed-by-downstream-issues state (chunking output is in-memory until #15 lands).

## Doc sync

Updated:
- `docs/decisions/2026-05-24-rag-enhancements.md` (new ADR, D1-D11)
- `docs/specs/2026-05-22-design.md` §8 amendment notes
- `docs/ARCHITECTURE.md` (model swap: voyage-finance-2, bge-reranker-v2-m3)
- `docs/plans/2026-05-22-auto-research-implementation.md` (#13, #14, #15, #16, #17, #18, #19 updated)
- `docs/CONTRACTS.md` (chunking entities)
PRBODY
)"
```

- [ ] **Step 12.2: Verify PR is up**

Run: `gh pr view --json url -q .url`

Expected: a PR URL prints. Done.

---

## Self-Review

**Spec coverage:** Every AC bullet in the updated issue body has a named test in Task 11.2's table. The ADR's six landing-with-#13 decisions (D1, D3, D4, D5, D7, D9) are covered: D1 (versioning) in Task 1; D3 (pins) in Task 1; D4 (parent/child) in Task 5+6; D5 (table policy) in Task 7+8; D7 (metadata) in Task 3; D9 (char_span + edge cases) in Task 7+9+10.

**Placeholder scan:** No "TBD" / "implement later" / "add appropriate handling" left. Every test step has explicit code.

**Type consistency:** `ParentChunk`, `ChildChunk`, `ChunkMetadata`, `ChunkSet`, `ChunkValidationError`, `SINGLE_SHOT_TOKEN_CUTOFF`, `MAX_PARENT_TOKENS`, `MIN_CHILD_TOKENS`, `MAX_CHILD_TOKENS`, `parse_filing`, `detect_sections`, `_pack_parents_for_section`, `subdivide_to_children`, `validate_char_spans`, `quarantine_chunkset`, `count_tokens` — names consistent across tasks. `char_span: tuple[int, int]` and `metadata: ChunkMetadata` shape stable.

**Risk noted in advance:** Task 8 step 8.2 flags `text_as_html` as a potential `unstructured` quirk; investigation path is named. Task 10 step 10.2 flags NLTK auto-download as a potential network-test failure; the fix lives in conftest, not in the assertion.

---

## Execution choice

Plan complete and saved to `docs/plans/per-issue/13-unstructured-chunking.md`.

Auto-mode default for this session: **inline execution** via this same session (no subagent dispatch — task scope fits in one window, and the Tier 2 char_span work benefits from one author keeping the contract straight across tasks).
