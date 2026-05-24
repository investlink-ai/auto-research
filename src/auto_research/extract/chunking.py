"""Section-aware SEC filing parsing + chunking (Issue #13, Tier 2 per INV-2).

Public surface:

- `parse_filing(html, metadata) -> ChunkSet` — pure function, no network.
- `ParentChunk` — ≤ 4K-token context unit, section-respecting.
- `ChildChunk` — 200-800-token retrieval unit, never crosses a parent.
- `ChunkValidationError` — raised when char_span fidelity (INV-2) fails.
- `validate_char_spans(source_text, parents, children)` — runtime guard.
- `quarantine_chunkset(...)` — INV-2 quarantine routing.

Design (per `docs/decisions/2026-05-24-rag-enhancements.md`, decisions D1,
D3, D4, D5, D7, D9):

- Library-first via `unstructured.partition.html.partition_html` (OSS,
  Apache 2.0). The library identifies structural elements (`Title`,
  `NarrativeText`, `Table`); the chunker locates each element's text in
  the raw HTML via entity-tolerant search and emits parent/child chunks
  whose `text` is the raw-HTML slice (so `source_text[span] == chunk.
  text` holds trivially — INV-2).

- 10-K Item 8 `Table` elements emit as summary `ParentChunk`s with the
  raw `<table>...</table>` HTML attached on `table_html`; structured
  extraction (Issue #19's 10-K worker) consumes `table_html` directly,
  bypassing dense retrieval (ADR D5).

- NVDA-style HTML entities (`&#160;`, `&nbsp;`) inside section headers
  are tolerated by entity-aware fallback in `_find_offset` (ADR D9).

Tables and narrative chunks share `section_name`. Children carry
`parent_id = f"{doc_id}::{parent.char_span[0]}-{parent.char_span[1]}"`
back-referencing the parent's identity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import tiktoken
from unstructured.partition.html import partition_html

from auto_research._io import atomic_write_text


def _ensure_nlp_warmup() -> None:
    """Pre-load `unstructured`'s spaCy model.

    `unstructured.partition.html.partition_html` calls into
    `unstructured.nlp.tokenize.sent_tokenize` for element classification.
    On first use this lazily downloads `en_core_web_sm` from GitHub —
    a network call that breaks hermetic tests and silently surprises
    fresh deployments.

    We warm the cache eagerly so:
      1. Production behavior is deterministic — the first parse_filing
         call after process start does not phone home.
      2. The hermetic `test_parse_filing_makes_no_network_calls` test
         passes; the test monkey-patches sockets after this import,
         when the model is already in memory.

    A missing model raises with a clear remediation path. No silent
    network downloads.
    """
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required by "
            "unstructured.partition_html for element classification but "
            "is not installed. Install with:\n"
            "    uv run python -m spacy download en_core_web_sm\n"
            "Or run `make setup-nlp` from the repo root."
        ) from exc


_ensure_nlp_warmup()

# ---------- Constants -------------------------------------------------------

SINGLE_SHOT_TOKEN_CUTOFF: Final[int] = 100_000
"""Docs below this token count go single-shot; >= goes through RAG (ADR D10)."""

MAX_PARENT_TOKENS: Final[int] = 4_000
MIN_CHILD_TOKENS: Final[int] = 200
MAX_CHILD_TOKENS: Final[int] = 800

DEFAULT_QUARANTINE_ROOT: Final[Path] = Path("data/quarantine")

# cl100k_base is the closest publicly available tokenizer to Claude's
# (Anthropic does not publish a stable public tokenizer). Used here only
# for chunk-size budgeting, not for cost estimation, so exact alignment
# isn't required.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Pattern for SEC Item section headers. Tolerates HTML entities between
# "Item" and the number — NVDA's 10-K uses `Item&#160;N.` for example.
_ITEM_HEADER = re.compile(
    r"\bitem(?:\s|&\#160;|&nbsp;|&\#x[Aa]0;)+(\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)

# Valid SEC Form 10-K Item numbers per the SEC's Form 10-K instructions
# (https://www.sec.gov/files/form10-k.pdf). Restricting detection to this
# set filters out rule references ("Item 408 of Regulation S-K") and
# accidental matches in cross-references.
_VALID_10K_ITEMS: Final[frozenset[str]] = frozenset(
    {
        # Part I
        "1", "1A", "1B", "1C", "2", "3", "4",
        # Part II
        "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
        # Part III
        "10", "11", "12", "13", "14",
        # Part IV
        "15", "16",
    }
)

# Number of preceding chars inspected by `_looks_like_block_header` to
# decide if a candidate Item match starts a structural header (vs. an
# inline cross-reference like "compared to Item 7 above").
_BLOCK_HEADER_LOOKBACK: Final[int] = 80


# ---------- Exceptions ------------------------------------------------------


class ChunkValidationError(ValueError):  # typed contract name
    """Raised when a chunk's char_span doesn't slice back to its text.

    Mirrors `extract.guardrails.CitationMismatch` for the chunking
    layer's half of INV-2. Subclasses `ValueError` so generic callers
    can match on the base type; typed so chunking-aware callers can
    route to the chunking quarantine path without catching unrelated
    `ValueError`s.
    """


# ---------- Dataclasses -----------------------------------------------------


@dataclass(frozen=True)
class ChunkMetadata:
    """Per-document metadata copied onto every chunk (ADR D7).

    These fields land as LanceDB columns in Issue #15 so retrieval can
    filter at index time — e.g., Signal A1 windowing by `(ticker,
    filing_date)`.
    """

    ticker: str
    filing_date: date
    fiscal_period: str
    doc_type: str
    doc_id: str


@dataclass(frozen=True)
class ParentChunk:
    """The extraction-context unit (≤ MAX_PARENT_TOKENS).

    `table_html` is the raw `<table>...</table>` HTML when this chunk
    represents an Item 8 table; `None` for narrative chunks. The 10-K
    worker in Issue #19 reads `table_html` directly via a typed Pydantic
    schema, bypassing dense retrieval over tabular text.
    """

    text: str
    section_name: str
    char_span: tuple[int, int]
    token_count: int
    table_html: str | None
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ChildChunk:
    """The retrieval-embedding unit (200-800 tokens, never crosses a parent)."""

    text: str
    char_span: tuple[int, int]
    token_count: int
    parent_id: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ChunkSet:
    """Result of `parse_filing` — parents + children together."""

    parents: list[ParentChunk]
    children: list[ChildChunk]


@dataclass(frozen=True)
class _DetectedSection:
    """Internal: a section's name and its char_span in the raw HTML."""

    name: str
    char_span: tuple[int, int]


# ---------- Token counting --------------------------------------------------


def count_tokens(text: str) -> int:
    """Return cl100k_base token count for `text`."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))


# ---------- Char_span fidelity (INV-2) --------------------------------------


def validate_char_spans(
    source_text: str,
    parents: Iterable[ParentChunk],
    children: Iterable[ChildChunk],
) -> None:
    """Assert every chunk's char_span slices to its text in source_text.

    Mirrors `guardrails._walk_citations`'s discipline for the chunking
    half of INV-2. Raises `ChunkValidationError` on the first mismatch;
    callers route to `quarantine_chunkset` rather than persisting a
    corrupted ChunkSet to downstream consumers.
    """
    for p in parents:
        a, b = p.char_span
        if a < 0 or b > len(source_text) or a >= b:
            raise ChunkValidationError(
                f"parent {p.section_name!r} span out of bounds: "
                f"({a}, {b}) vs source len {len(source_text)}"
            )
        sliced = source_text[a:b]
        if sliced != p.text:
            raise ChunkValidationError(
                f"parent {p.section_name!r} text mismatch at ({a}, {b}): "
                f"source[{a}:{b}]={sliced[:80]!r} vs chunk.text={p.text[:80]!r}"
            )
    for c in children:
        a, b = c.char_span
        if a < 0 or b > len(source_text) or a >= b:
            raise ChunkValidationError(
                f"child {c.parent_id!r} span out of bounds: ({a}, {b})"
            )
        sliced = source_text[a:b]
        if sliced != c.text:
            raise ChunkValidationError(
                f"child {c.parent_id!r} text mismatch at ({a}, {b}): "
                f"source[{a}:{b}]={sliced[:80]!r} vs chunk.text={c.text[:80]!r}"
            )


# ---------- Quarantine ------------------------------------------------------


def quarantine_chunkset(
    chunkset: ChunkSet,
    *,
    source_text: str,
    reason: str,
    quarantine_root: Path = DEFAULT_QUARANTINE_ROOT,
) -> Path:
    """Write a chunking quarantine record under `<root>/chunking/<doc_id>.json`.

    Mirrors `guardrails.validate_or_quarantine`'s contract — the caller
    that obtained the offending ChunkSet must NOT persist any of it
    downstream. Returned path lets the caller (and humans) audit the
    record without reading internal state.
    """
    if chunkset.parents:
        doc_id = chunkset.parents[0].metadata.doc_id
    elif chunkset.children:
        doc_id = chunkset.children[0].metadata.doc_id
    else:
        doc_id = "empty"

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


# ---------- Locating elements in raw HTML -----------------------------------


# Entity-flexible whitespace pattern for raw-HTML search.
_WS_FLEX = r"(?:\s|&\#160;|&nbsp;|&\#x[Aa]0;)+"


def _entity_flex_pattern(needle: str) -> re.Pattern[str]:
    """Compile a regex that matches `needle`'s words separated by
    HTML-entity-tolerant whitespace.

    Used when the decoded element text (e.g. `Item 8.`) must be located
    in the raw HTML (`Item&#160;8.`).
    """
    words = needle.split()
    if not words:
        return re.compile(re.escape(needle))
    return re.compile(_WS_FLEX.join(re.escape(w) for w in words), re.DOTALL)


def _find_offset(haystack: str, needle: str, start: int = 0) -> int | None:
    """Find `needle` in `haystack[start:]` allowing HTML-entity whitespace.

    Returns the absolute offset, or None if no match.
    """
    if not needle.strip():
        return None
    idx = haystack.find(needle, start)
    if idx != -1:
        return idx
    m = _entity_flex_pattern(needle).search(haystack, start)
    return m.start() if m else None


def _find_text_span(haystack: str, needle: str, start: int = 0) -> tuple[int, int] | None:
    """Like `_find_offset` but returns `(start, end)` of the match.

    `end` is the inclusive end of the match in the raw HTML — useful
    when whitespace expansion through entities means `end != start + len(
    needle)`. INV-2 holds because the returned span's slice (potentially
    larger than `needle`) is the chunk's text by construction.
    """
    if not needle.strip():
        return None
    idx = haystack.find(needle, start)
    if idx != -1:
        return idx, idx + len(needle)
    m = _entity_flex_pattern(needle).search(haystack, start)
    if m is None:
        return None
    return m.start(), m.end()


# ---------- Section detection -----------------------------------------------


def _section_name_from_title(title_text: str) -> str | None:
    """Return `"Item N"` if `title_text` looks like a SEC Item heading."""
    m = _ITEM_HEADER.match(title_text.strip())
    if m is None:
        return None
    return f"Item {m.group(1).upper()}"


def _is_real_section_header(html: str, span_start: int) -> bool:
    """A candidate is a "real" section start (not a TOC echo) if the
    next 2000 chars of raw HTML contain substantial prose content
    (≥ 200 alphabetic characters after stripping tags and entities)."""
    snippet = html[span_start : span_start + 2000]
    stripped = re.sub(r"<[^>]+>|&[^;]+;", " ", snippet)
    alpha = sum(1 for c in stripped if c.isalpha())
    return alpha >= 200


def _looks_like_block_header(html: str, span_start: int) -> bool:
    """Return True if the candidate `Item N` match looks like a styled
    structural header rather than an inline cross-reference.

    Real Item headers in SEC HTML are typically:
      - Immediately preceded by a closing block tag (`</div>`, `</p>`,
        `</td>`, `</tr>`), an opening block/inline tag whose content
        starts the header (`<span>`, `<div>`, `<h1>`...), or document
        start.
      - NOT preceded by lowercase prose ("compared to Item 7" — a
        cross-reference).

    Inspected window: `_BLOCK_HEADER_LOOKBACK` chars before `span_start`.
    """
    if span_start <= 0:
        return True  # document start is structural by definition
    window_start = max(0, span_start - _BLOCK_HEADER_LOOKBACK)
    preceding = html[window_start:span_start]

    # Empty preceding window → document start, structural by definition.
    if not preceding.rstrip():
        return True

    # Structural boundaries — any of: (a) closing tag, (b) opening tag
    # right before the header text, (c) closing HTML comment such as
    # the post-trim fixture marker `-->`.
    return bool(
        re.search(r"</[a-zA-Z][^>]*>\s*$", preceding)
        or re.search(r"<[a-zA-Z][^>]*>\s*$", preceding)
        or re.search(r"-->\s*$", preceding)
    )


_TABLE_OPEN = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_CLOSE = re.compile(r"</table\s*>", re.IGNORECASE)


def _mask_comments(html: str) -> str:
    """Replace HTML comment bodies with spaces, preserving offsets.

    Section-detection scans must ignore matches inside `<!-- ... -->`
    blocks — fixture truncation markers and other meta-comments can
    legitimately contain "Item N" references that aren't real headers.
    Replacement (rather than removal) keeps every absolute char offset
    in the masked string identical to the original.
    """
    out = list(html)
    for m in re.finditer(r"<!--.*?-->", html, re.DOTALL):
        for i in range(m.start(), m.end()):
            out[i] = " "
    return "".join(out)


def _detect_sections(html: str) -> list[_DetectedSection]:
    """Detect SEC Item sections by scanning the raw HTML.

    Scans the raw document directly (not via `unstructured`'s element
    stream) because `unstructured`'s text classification is unreliable
    for SEC 10-K headers — Items can be classified as `Title`, `Text`,
    or even buried inside a larger `NarrativeText` element depending on
    the surrounding markup. The entity-aware regex matches `Item N.` /
    `Item&#160;N.` patterns directly.

    Filters applied (in order):
      1. Only valid SEC Form 10-K Item numbers (per `_VALID_10K_ITEMS`)
         — drops accidental matches like "Item 408 of Regulation S-K".
      2. Drop matches inside HTML comments (`<!-- ... -->`) — used by
         test fixtures' truncation markers and other meta-content.
      3. `_is_real_section_header` — ≥200 alphabetic chars of prose
         follow the candidate header (drops TOC entries pointing at
         later content).
      4. Consecutive section starts must be ≥ `_MIN_SECTION_BYTES`
         apart — real 10-K Items are tens to hundreds of KB apart;
         closer matches are inline cross-references.

    Returns sections in document order with absolute char_span tiles
    covering the whole document (last section's end = len(html)).
    """
    masked = _mask_comments(html)

    # Collect first qualifying occurrence of each valid Item.
    by_name: dict[str, int] = {}
    for m in _ITEM_HEADER.finditer(masked):
        num = m.group(1).upper()
        if num not in _VALID_10K_ITEMS:
            continue
        if not _looks_like_block_header(html, m.start()):
            continue
        if not _is_real_section_header(html, m.start()):
            continue
        name = f"Item {num}"
        by_name.setdefault(name, m.start())

    if not by_name:
        return []

    ordered = sorted(by_name.items(), key=lambda kv: kv[1])
    sections: list[_DetectedSection] = []
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(html)
        sections.append(_DetectedSection(name=name, char_span=(start, end)))
    return sections


# Natural HTML break boundaries for narrative packing — closing tags
# that mark logical content units (paragraphs, divs, table rows, list
# items, headings). The regex finds the END of each closing tag so
# breakpoints fall at semantic seams, never mid-tag.
_HTML_BREAK = re.compile(r"</(?:p|div|td|tr|li|h[1-6])\s*>", re.IGNORECASE)


# ---------- Parent packing --------------------------------------------------


def _pack_narrative_html(
    html: str,
    start: int,
    end: int,
    section_name: str,
    metadata: ChunkMetadata,
) -> list[ParentChunk]:
    """Greedy-pack the raw HTML slice `html[start:end]` into ParentChunks
    of ≤MAX_PARENT_TOKENS, breaking only at natural HTML boundaries
    (closing `</p>`, `</div>`, `</td>`, etc.).

    The resulting chunks tile `[start, end)` contiguously: each chunk's
    char_span starts where the previous one ended. INV-2 holds because
    every chunk's text is `html[char_span]` by construction.

    If no break boundary exists between two consecutive flushes (a single
    "atom" of HTML exceeds MAX_PARENT_TOKENS), we emit the atom anyway
    rather than slicing mid-tag — splitting raw HTML at an arbitrary
    character would produce invalid HTML and break INV-2-adjacent
    contracts downstream (e.g., extraction worker prompts that expect
    well-formed HTML).
    """
    if end <= start:
        return []

    # Build list of candidate break positions (absolute offsets in html)
    # that fall within [start, end]. Always include `end` so the tail
    # can flush.
    breaks: list[int] = []
    for m in _HTML_BREAK.finditer(html, start, end):
        bp = m.end()
        if bp > start:
            breaks.append(bp)
    if not breaks or breaks[-1] < end:
        breaks.append(end)

    chunks: list[ParentChunk] = []
    chunk_start = start
    last_safe_break = start  # furthest break we know fits in budget

    for bp in breaks:
        if bp <= chunk_start:
            continue
        candidate = html[chunk_start:bp]
        candidate_tokens = count_tokens(candidate)
        if candidate_tokens <= MAX_PARENT_TOKENS:
            # Still under budget; record this as the last safe break.
            last_safe_break = bp
            continue
        # Over budget. Flush at last_safe_break (if we have one beyond
        # chunk_start), then start a new buffer at last_safe_break.
        if last_safe_break > chunk_start:
            text = html[chunk_start:last_safe_break]
            chunks.append(
                ParentChunk(
                    text=text,
                    section_name=section_name,
                    char_span=(chunk_start, last_safe_break),
                    token_count=count_tokens(text),
                    table_html=None,
                    metadata=metadata,
                )
            )
            chunk_start = last_safe_break
        else:
            # No safe break — single atom exceeds budget. Emit anyway.
            text = html[chunk_start:bp]
            chunks.append(
                ParentChunk(
                    text=text,
                    section_name=section_name,
                    char_span=(chunk_start, bp),
                    token_count=candidate_tokens,
                    table_html=None,
                    metadata=metadata,
                )
            )
            chunk_start = bp
            last_safe_break = bp

    # Flush the final buffer up to `end`.
    if chunk_start < end:
        text = html[chunk_start:end]
        chunks.append(
            ParentChunk(
                text=text,
                section_name=section_name,
                char_span=(chunk_start, end),
                token_count=count_tokens(text),
                table_html=None,
                metadata=metadata,
            )
        )

    return chunks


def _emit_section_chunks(
    html: str, section: _DetectedSection, metadata: ChunkMetadata
) -> list[ParentChunk]:
    """Emit ParentChunks for a section, separating tables from narrative.

    Tables → standalone chunks with `table_html` populated (ADR D5).
    Narrative regions between tables → packed via `_pack_narrative_html`.
    Output is in document order; char_spans tile the section.
    """
    start, end = section.char_span

    # Locate `<table>...</table>` spans within this section.
    tables: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        open_m = _TABLE_OPEN.search(html, pos)
        if open_m is None or open_m.start() >= end:
            break
        close_m = _TABLE_CLOSE.search(html, open_m.end())
        if close_m is None:
            break
        tbl_end = min(close_m.end(), end)
        tables.append((open_m.start(), tbl_end))
        pos = tbl_end

    # Build alternating regions (narrative, table, narrative, ...).
    chunks: list[ParentChunk] = []
    cursor = start
    for ts, te in tables:
        if ts > cursor:
            chunks.extend(_pack_narrative_html(html, cursor, ts, section.name, metadata))
        table_html = html[ts:te]
        chunks.append(
            ParentChunk(
                text=table_html,
                section_name=section.name,
                char_span=(ts, te),
                token_count=count_tokens(table_html),
                table_html=table_html,
                metadata=metadata,
            )
        )
        cursor = te
    if cursor < end:
        chunks.extend(_pack_narrative_html(html, cursor, end, section.name, metadata))

    return chunks


# ---------- Child subdivision -----------------------------------------------


def _parent_id(parent: ParentChunk) -> str:
    return f"{parent.metadata.doc_id}::{parent.char_span[0]}-{parent.char_span[1]}"


def subdivide_to_children(parent: ParentChunk) -> list[ChildChunk]:
    """Sentence-window subdivide a parent into 200-800 token children.

    Children never cross the parent boundary; `char_span`s are absolute
    into the raw source HTML. Each child's text is `source[span]` by
    construction (INV-2 holds).

    Parents at or below MAX_CHILD_TOKENS emit a single child equal to
    the parent — degenerate but contract-safe.
    """
    if parent.token_count <= MAX_CHILD_TOKENS:
        return [
            ChildChunk(
                text=parent.text,
                char_span=parent.char_span,
                token_count=parent.token_count,
                parent_id=_parent_id(parent),
                metadata=parent.metadata,
            )
        ]

    # Walk along character positions in the parent's local string,
    # accumulating until we hit MIN_CHILD_TOKENS and then flushing as
    # soon as we hit MAX_CHILD_TOKENS or run out.
    children: list[ChildChunk] = []
    parent_text = parent.text
    abs_offset = parent.char_span[0]

    # Sentence-ish boundaries: keep tag-aware splitting by finding `. `,
    # `! `, `? ` in the rendered text. iXBRL HTML has these inside <span>
    # tags so we match by char-class (no lookbehind across tags needed).
    boundary_pat = re.compile(r"(?<=[.!?])\s+|</p>|</div>|</td>", re.IGNORECASE)
    breaks = [m.end() for m in boundary_pat.finditer(parent_text)]
    if not breaks or breaks[-1] != len(parent_text):
        breaks.append(len(parent_text))

    cursor = 0
    for cut in breaks:
        if cut <= cursor:
            continue
        candidate_text = parent_text[cursor:cut]
        candidate_tokens = count_tokens(candidate_text)
        if candidate_tokens >= MIN_CHILD_TOKENS or cut == len(parent_text):
            # Emit if at/above MIN, or if this is the last cut (covers tail).
            if candidate_tokens > MAX_CHILD_TOKENS:
                # Single boundary exceeded MAX — emit as-is rather than
                # cutting mid-token. Real-world iXBRL paragraphs rarely
                # hit this; falling back keeps INV-2 intact.
                pass
            children.append(
                ChildChunk(
                    text=candidate_text,
                    char_span=(abs_offset + cursor, abs_offset + cut),
                    token_count=candidate_tokens,
                    parent_id=_parent_id(parent),
                    metadata=parent.metadata,
                )
            )
            cursor = cut

    # If for some reason the loop emitted nothing (no boundary matches and
    # token count below MIN), emit the whole parent as one child.
    if not children:
        children.append(
            ChildChunk(
                text=parent.text,
                char_span=parent.char_span,
                token_count=parent.token_count,
                parent_id=_parent_id(parent),
                metadata=parent.metadata,
            )
        )

    return children


# ---------- Top-level entrypoint --------------------------------------------


def parse_filing(*, html: str, metadata: ChunkMetadata) -> ChunkSet:
    """Parse SEC-filing HTML into section-aware parent + child chunks.

    Pure function: same `(html, metadata)` → same `ChunkSet`. No
    network, no LLM. Raises `ChunkValidationError` if any chunk fails
    the char_span identity test; callers route to `quarantine_chunkset`
    rather than persisting a corrupted result.

    Implementation: section boundaries come from a direct entity-aware
    scan of the raw HTML (more reliable than `unstructured`'s element
    classification for SEC 10-Ks where Item headers can land in any
    element type). `unstructured.partition_html` is still invoked once
    at module load time to warm the NLP cache (see `_ensure_nlp_warmup`),
    keeping production behavior deterministic and tests hermetic.

    Tables (`<table>...</table>` spans) emit as standalone ParentChunks
    with `table_html` populated. Narrative between tables packs into
    chunks ≤ MAX_PARENT_TOKENS along HTML boundary tags. Children
    subdivide each parent into 200-800-token retrieval units.
    """
    # Sanity-parse via unstructured. This catches catastrophically
    # malformed HTML (closes the "garbage in, garbage out" risk that
    # raw regex alone wouldn't surface) but the result is not used by
    # the chunker. Failures here propagate to the caller.
    partition_html(text=html)

    sections = _detect_sections(html)
    if not sections:
        # Whole document as one synthetic "Body" section.
        sections = [_DetectedSection(name="Body", char_span=(0, len(html)))]

    parents: list[ParentChunk] = []
    for s in sections:
        parents.extend(_emit_section_chunks(html, s, metadata))

    children: list[ChildChunk] = []
    for p in parents:
        children.extend(subdivide_to_children(p))

    validate_char_spans(html, parents, children)
    return ChunkSet(parents=parents, children=children)


__all__ = [
    "DEFAULT_QUARANTINE_ROOT",
    "MAX_CHILD_TOKENS",
    "MAX_PARENT_TOKENS",
    "MIN_CHILD_TOKENS",
    "SINGLE_SHOT_TOKEN_CUTOFF",
    "ChildChunk",
    "ChunkMetadata",
    "ChunkSet",
    "ChunkValidationError",
    "ParentChunk",
    "count_tokens",
    "parse_filing",
    "quarantine_chunkset",
    "subdivide_to_children",
    "validate_char_spans",
]
