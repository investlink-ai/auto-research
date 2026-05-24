"""Section-aware SEC filing parsing + chunking (Tier 2 per INV-2).

Public surface:

- `parse_filing(html, metadata) -> ChunkSet` — pure function, no network.
- `ParentChunk` — ≤ 4K-token context unit, section-respecting.
- `ChildChunk` — 200-800-token retrieval unit, never crosses a parent.
- `ChunkValidationError` — raised when char_span fidelity (INV-2) fails.
- `validate_char_spans(source_text, parents, children)` — runtime guard.
- `quarantine_chunkset(...)` — INV-2 quarantine routing.
- `validate_or_quarantine_chunkset(...)` — single-call routing helper.

Design recorded in `docs/decisions/2026-05-24-rag-enhancements.md`.

- Section detection uses an entity-aware regex on the raw HTML
  (tolerates `&#160;` / `&nbsp;` between `Item` and the number, since
  iXBRL filings commonly emit non-breaking-space entities at section
  headers). `unstructured.partition.html` is imported transitively for
  downstream callers; this module does NOT invoke it.

- 10-K Item 8 tables emit as standalone `ParentChunk`s with the raw
  `<table>...</table>` HTML attached on `table_html`. Downstream
  structured extraction reads `table_html` via a typed Pydantic
  schema, bypassing dense retrieval. Nested `<table>` depth is tracked
  so `table_html` always covers the full outer table.

- chunk.text is `html[char_span]` by construction. INV-2 holds via
  this identity; `validate_char_spans` is a defense-in-depth runtime
  check, not the load-bearing guarantee.

Children carry `parent_id = "{doc_id}::{parent.char_span[0]}-{parent
.char_span[1]}"`, `section_name` (copied from the parent so LanceDB
can filter at index time without a parent JOIN), and `from_table`
(True iff the child was emitted under a table parent — table parents
are atomic, so the child equals the parent in that case).
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

from auto_research._io import atomic_write_text

# Module-level cache: once a successful warmup completes, subsequent
# `_ensure_nlp_warmup()` calls are no-ops. The flag is process-local so
# pytest-xdist workers each warm once, but a single test run does not
# re-pay the spaCy-load cost on every parse_filing call.
_NLP_WARMED: bool = False


def _ensure_nlp_warmup() -> None:
    """Warm `unstructured`'s spaCy model on first use.

    `unstructured.partition.html.partition_html` calls into
    `unstructured.nlp.tokenize.sent_tokenize` for element classification.
    On first use this lazily downloads `en_core_web_sm` from GitHub —
    a network call that breaks hermetic tests and silently surprises
    fresh deployments.

    We warm the cache on the first `parse_filing` call so:
      1. Production behavior is deterministic — within a process, the
         first parse pays the warmup cost; subsequent parses don't.
      2. The hermetic `test_parse_filing_makes_no_network_calls` test
         passes — the warmup runs once via a conftest autouse fixture
         BEFORE the socket monkey-patch, so by the time `parse_filing`
         is called under the patch, the model is already in memory.
      3. Module import is fast — no eager spaCy load at import time, so
         tooling that imports the module (mypy via inference,
         pytest-xdist worker bootstrap, IDE plugins) does not require
         the spaCy model to be installed just to read the module.

    Idempotent via the `_NLP_WARMED` flag. A missing model raises
    `RuntimeError` with a clear remediation path; no silent network
    downloads.
    """
    global _NLP_WARMED
    if _NLP_WARMED:
        return
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
    _NLP_WARMED = True

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
# `Item` itself is case-insensitive (real filings vary "Item"/"ITEM"); the
# entity alternatives are NOT — HTML5 entity names like `&nbsp;` are
# case-sensitive at the spec level (browsers reject `&NBSP;`). Composing
# with `(?i:item)` makes the leading word case-insensitive while keeping
# entity matching strict.
_ITEM_HEADER = re.compile(
    r"\b(?i:item)(?:\s|&\#160;|&nbsp;|&\#x[Aa]0;)+(\d+[A-Za-z]?)\b",
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

    These fields land as LanceDB columns downstream so retrieval can
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
    worker reads `table_html` directly via a typed Pydantic schema,
    bypassing dense retrieval over tabular text.
    """

    text: str
    section_name: str
    char_span: tuple[int, int]
    token_count: int
    table_html: str | None
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ChildChunk:
    """The retrieval-embedding unit (200-800 tokens, never crosses a parent).

    `section_name` is copied from the parent so the LanceDB schema
    downstream can filter at index time without a parent JOIN (ADR D7
    + D11). `from_table` flags fragments that originated under a
    table parent — the hybrid retriever pairs this with `table_html`
    on parents per ADR D5.
    """

    text: str
    char_span: tuple[int, int]
    token_count: int
    parent_id: str
    section_name: str
    from_table: bool
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ChunkSet:
    """Result of `parse_filing` — parents + children together.

    Fields are `tuple` rather than `list` so the frozen-dataclass guarantee
    extends to the contents — `chunkset.parents.append(...)` and similar
    in-place mutations are mechanically rejected. Callers that need to
    materialize a working list call `list(chunkset.parents)`.
    """

    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]


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
    doc_id: str,
    source_text: str,
    reason: str,
    quarantine_root: Path = DEFAULT_QUARANTINE_ROOT,
) -> Path:
    """Write a chunking quarantine record under `<root>/chunking/<doc_id>.json`.

    Mirrors `guardrails.validate_or_quarantine`'s contract — the caller
    that obtained the offending ChunkSet must NOT persist any of it
    downstream. Returned path lets the caller (and humans) audit the
    record without reading internal state.

    `doc_id` is required (not derived from chunkset contents) — empty
    ChunkSets are reachable via the public API (`parse_filing(html='')`
    produces one) and deriving `doc_id='empty'` for those caused
    silent overwrites at `<root>/chunking/empty.json`. The caller
    always knows which document failed; the quarantine record must
    carry that identity verbatim.

    Raises `ValueError` if `doc_id` is empty / whitespace-only.
    """
    if not doc_id or not doc_id.strip():
        raise ValueError(
            "quarantine_chunkset requires a non-empty doc_id — empty chunksets "
            "are still tied to a specific document and must be filed under it"
        )

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
                "section_name": c.section_name,
                "from_table": c.from_table,
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


def validate_or_quarantine_chunkset(
    chunkset: ChunkSet,
    *,
    source_text: str,
    doc_id: str,
    quarantine_root: Path = DEFAULT_QUARANTINE_ROOT,
) -> ChunkSet | None:
    """Single-call routing helper mirroring `guardrails.validate_or_quarantine`.

    Runs `validate_char_spans`; on success returns the `ChunkSet`, on
    `ChunkValidationError` writes the chunking quarantine record and
    returns `None`. Callers that get `None` MUST NOT persist any part of
    the result downstream (LanceDB, Feast, extracted JSONL) — that's
    how silent INV-2 degradation gets in. Mirrors the entire entry-
    point contract of `extract.guardrails.validate_or_quarantine` so
    callers can write the same pattern on both halves of INV-2:

        chunkset = parse_filing(html=raw, metadata=meta)
        chunkset = validate_or_quarantine_chunkset(
            chunkset, source_text=raw, doc_id=meta.doc_id,
        )
        if chunkset is None:
            return  # already quarantined; do not persist

    `doc_id` is required because `quarantine_chunkset` requires it.
    """
    try:
        validate_char_spans(source_text, chunkset.parents, chunkset.children)
    except ChunkValidationError as exc:
        quarantine_chunkset(
            chunkset,
            doc_id=doc_id,
            source_text=source_text,
            reason=str(exc),
            quarantine_root=quarantine_root,
        )
        return None
    return chunkset


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


# How many bytes after the candidate header to skip before counting
# prose density — the header itself ("Item 1A. Risk Factors") can easily
# contribute 50+ alpha chars and falsely boost the density score for TOC
# echoes packed adjacent to other Item names. The skip is short enough to
# still capture a real section's opening sentence inside the 2000-char
# window.
_HEADER_DENSITY_SKIP_BYTES: Final[int] = 200
_HEADER_DENSITY_WINDOW_BYTES: Final[int] = 2_000
_HEADER_DENSITY_MIN_ALPHA: Final[int] = 200


def _is_real_section_header(html: str, span_start: int) -> bool:
    """A candidate is a "real" section start (not a TOC echo) if the
    raw HTML BEYOND the header's own text contains substantial prose
    content (≥ `_HEADER_DENSITY_MIN_ALPHA` alphabetic characters after
    tag/entity stripping).

    The window starts `_HEADER_DENSITY_SKIP_BYTES` past `span_start` so
    the header's own letters ("Risk Factors", "Management Discussion")
    do not bleed into the threshold. This guards against (a) TOC entries
    where surrounding Item names contribute high alpha density without
    real prose, and (b) inline cross-references where the formatted
    `<span>Item N</span>` is followed by continuing sentence text but no
    actual section body.
    """
    window_start = span_start + _HEADER_DENSITY_SKIP_BYTES
    snippet = html[window_start : window_start + _HEADER_DENSITY_WINDOW_BYTES]
    stripped = re.sub(r"<[^>]+>|&[^;]+;", " ", snippet)
    alpha = sum(1 for c in stripped if c.isalpha())
    return alpha >= _HEADER_DENSITY_MIN_ALPHA


# Block-level HTML tags that mark structural boundaries in SEC filings.
# Section headers typically open inside or immediately after one of
# these; inline formatting tags (`<span>`, `<b>`, `<i>`, `<a>`, `<em>`,
# `<strong>`) do NOT count — accepting them as structural causes inline
# cross-references like `... see <span class="bold">Item 7</span> ...`
# to false-positive as section starts.
_BLOCK_TAGS = "(?:div|p|td|tr|li|h[1-6]|section|article|header|footer|main|body|table|tbody|thead|tfoot)"

_BLOCK_CLOSE_RE = re.compile(rf"</{_BLOCK_TAGS}[^>]*>\s*$", re.IGNORECASE)
_BLOCK_OPEN_RE = re.compile(rf"<{_BLOCK_TAGS}[^>]*>\s*$", re.IGNORECASE)
_COMMENT_END_RE = re.compile(r"-->\s*$")


def _looks_like_block_header(html: str, span_start: int) -> bool:
    """Return True if the candidate `Item N` match looks like a styled
    structural header rather than an inline cross-reference.

    Real Item headers in SEC HTML are immediately preceded by one of:
      (a) a closing BLOCK tag (`</div>`, `</p>`, `</td>`, `</h*>`, etc.) —
          structural boundary;
      (b) an opening BLOCK tag whose content starts the header text;
      (c) a closing HTML comment (`-->`) — test-fixture truncation
          markers and similar meta-content;
      (d) document start.

    Inline formatting tags (`<span>`, `<b>`, `<i>`, `<a>`, `<em>`,
    `<strong>`) are explicitly excluded — they appear inside running
    prose and are not structural. This avoids false positives from
    cross-references like `... compared to <span>Item 7</span> ...`.

    Inspected window: `_BLOCK_HEADER_LOOKBACK` chars before `span_start`.
    """
    if span_start <= 0:
        return True  # document start is structural by definition
    window_start = max(0, span_start - _BLOCK_HEADER_LOOKBACK)
    preceding = html[window_start:span_start]

    # Empty preceding window → document start, structural by definition.
    if not preceding.rstrip():
        return True

    return bool(
        _BLOCK_CLOSE_RE.search(preceding)
        or _BLOCK_OPEN_RE.search(preceding)
        or _COMMENT_END_RE.search(preceding)
    )


_TABLE_OPEN = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_CLOSE = re.compile(r"</table\s*>", re.IGNORECASE)


def _find_matching_table_close(html: str, after: int) -> int | None:
    """Return the offset just past the `</table>` that closes the
    outermost `<table>` opened at `after - 1`-ish.

    Tracks nested `<table>` depth so a parent table that contains
    inner tables (legitimate in SEC iXBRL layouts — outer-shell table
    used for column alignment with inner financial-statement tables) is
    not truncated at the first inner `</table>`. Without depth
    tracking, `table_html` ends at the inner close and the outer
    table's remaining rows leak into a following narrative chunk —
    breaking the well-formed-HTML invariant downstream extraction
    relies on (`pandas.read_html(table_html)`).

    `after` is the offset just past the OPENING `<table ...>` whose
    matching close we want; depth starts at 1 (the outer open already
    consumed by the caller).
    """
    depth = 1
    pos = after
    while True:
        open_m = _TABLE_OPEN.search(html, pos)
        close_m = _TABLE_CLOSE.search(html, pos)
        if close_m is None:
            return None
        if open_m is not None and open_m.start() < close_m.start():
            depth += 1
            pos = open_m.end()
            continue
        depth -= 1
        if depth == 0:
            return close_m.end()
        pos = close_m.end()


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
    (closing `</p>`, `</div>`, `</td>`, etc.) that lie OUTSIDE HTML
    comments.

    The resulting chunks tile `[start, end)` contiguously: each chunk's
    char_span starts where the previous one ended. INV-2 holds because
    every chunk's text is `html[char_span]` by construction.

    Comment-internal break tags (`<!-- example: </p> -->`) are NOT
    treated as boundaries — `_pack_narrative_html` reads its boundary
    scan from `_mask_comments(html)` so it agrees with `_detect_sections`
    about what counts as document structure.

    If no break boundary exists between two consecutive flushes (a single
    "atom" of HTML exceeds MAX_PARENT_TOKENS), we emit the atom anyway
    rather than slicing mid-tag — splitting raw HTML at an arbitrary
    character would produce invalid HTML and break INV-2-adjacent
    contracts downstream (e.g., extraction worker prompts that expect
    well-formed HTML).
    """
    if end <= start:
        return []

    # Build candidate break positions from the comment-masked HTML so
    # that `</p>`/`</div>`/etc. inside a comment block do not register
    # as boundaries. The masked string preserves offsets (comments
    # replaced by spaces), so the resulting `bp` values are valid char
    # offsets into the original `html`.
    masked = _mask_comments(html)
    breaks: list[int] = []
    for m in _HTML_BREAK.finditer(masked, start, end):
        bp = m.end()
        if bp > start:
            breaks.append(bp)
    if not breaks or breaks[-1] < end:
        breaks.append(end)

    chunks: list[ParentChunk] = []
    chunk_start = start
    last_safe_break = start  # furthest break we know fits in budget

    def _emit(span_start: int, span_end: int) -> None:
        text = html[span_start:span_end]
        chunks.append(
            ParentChunk(
                text=text,
                section_name=section_name,
                char_span=(span_start, span_end),
                token_count=count_tokens(text),
                table_html=None,
                metadata=metadata,
            )
        )

    bp_index = 0
    while bp_index < len(breaks):
        bp = breaks[bp_index]
        if bp <= chunk_start:
            bp_index += 1
            continue
        candidate_tokens = count_tokens(html[chunk_start:bp])
        if candidate_tokens <= MAX_PARENT_TOKENS:
            last_safe_break = bp
            bp_index += 1
            continue

        # Over budget at this break. Flush whatever's safe, then RE-ENTER
        # the loop without advancing bp_index — the break that just
        # triggered the over-budget condition may itself be the new
        # chunk's first safe boundary, and we'd lose it if we just
        # continued past it.
        if last_safe_break > chunk_start:
            _emit(chunk_start, last_safe_break)
            chunk_start = last_safe_break
            # Do NOT advance bp_index — re-evaluate the same `bp` against
            # the new chunk_start. This fixes the safe-break-loss bug
            # where intermediate breaks between flushes were skipped.
            continue

        # No safe break in the current buffer — a single atom exceeds
        # MAX_PARENT_TOKENS. Splitting raw HTML mid-tag would invalidate
        # the markup; emit the atom as-is. The narrative-cap test
        # tolerates these because they are unavoidable for legitimate
        # filings (a single iXBRL paragraph spanning >4K tokens has no
        # internal break tag the chunker can use).
        _emit(chunk_start, bp)
        chunk_start = bp
        last_safe_break = bp
        bp_index += 1

    # Flush the final buffer up to `end`.
    if chunk_start < end:
        _emit(chunk_start, end)

    return chunks


def _emit_section_chunks(
    html: str, section: _DetectedSection, metadata: ChunkMetadata
) -> list[ParentChunk]:
    """Emit ParentChunks for a section, separating tables from narrative.

    Tables → standalone chunks with `table_html` populated (ADR D5).
    Narrative regions between tables → packed via `_pack_narrative_html`.
    Output is in document order; char_spans tile the section.

    Section-crossing tables (a `<table>` opens within the section but
    `</table>` closes after) are NOT emitted as table chunks — clamping
    their span to the section end would produce malformed `table_html`
    (open `<table>` with no close), violating ADR D5's well-formed-HTML
    contract. Cross-boundary tables fall through into the narrative
    pass: the open `<table>` is absorbed into a narrative chunk and the
    `</table>` lands in the next section's narrative chunk. Downstream
    consumers see the unbalanced HTML in narrative text and can choose
    to skip or quarantine — they do NOT see a `table_html` field
    claiming well-formed markup.
    """
    start, end = section.char_span

    # Locate outer `<table>...</table>` spans fully inside [start, end).
    # Nested-table depth is tracked by `_find_matching_table_close`, so
    # `table_html` always covers the complete outer table — never
    # truncates at an inner `</table>`. Cross-boundary tables are
    # skipped (see docstring).
    tables: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        open_m = _TABLE_OPEN.search(html, pos)
        if open_m is None or open_m.start() >= end:
            break
        tbl_end = _find_matching_table_close(html, open_m.end())
        if tbl_end is None:
            break
        if tbl_end > end:
            # Outer table crosses the section boundary. Skip emission
            # as a table chunk; the open `<table>` falls into this
            # section's narrative and the close into the next section's
            # narrative. Advance past the open so the loop doesn't spin.
            pos = open_m.end()
            continue
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


def _single_child_from_parent(parent: ParentChunk, *, from_table: bool) -> ChildChunk:
    """Build a single ChildChunk equal to its parent.

    Used in two cases: (1) parents small enough to fit one child, and
    (2) table parents — splitting a table mid-row would invalidate the
    HTML, so the child equals the parent and the `from_table` flag
    tells downstream retrieval to filter accordingly per ADR D5.
    """
    return ChildChunk(
        text=parent.text,
        char_span=parent.char_span,
        token_count=parent.token_count,
        parent_id=_parent_id(parent),
        section_name=parent.section_name,
        from_table=from_table,
        metadata=parent.metadata,
    )


def subdivide_to_children(parent: ParentChunk) -> list[ChildChunk]:
    """Subdivide a parent into 200-800 token children (ADR D4).

    Children never cross the parent boundary; `char_span`s are
    absolute into the raw source HTML. Each child's text is
    `source[span]` by construction (INV-2 holds). `section_name` and
    `from_table` are copied from the parent so LanceDB and the hybrid
    retriever can filter at the child level without a JOIN back to
    the parent.

    Table parents (`table_html is not None`) emit a single child equal
    to the parent — never sub-split (ADR D5). Splitting `<td>` seams
    produces fragments like `<table>...<td>cell</td>` with no closing
    `</table>`, which violates D5's well-formed-HTML invariant.

    Narrative parents at or below MAX_CHILD_TOKENS also emit a single
    child equal to the parent (degenerate but contract-safe).

    Narrative parents where the boundary regex cannot produce a split
    that stays under MAX_CHILD_TOKENS (e.g., a single iXBRL paragraph
    with no sentence punctuation across thousands of tokens) raise
    `ChunkValidationError`. The caller routes to `quarantine_chunkset`
    rather than silently emitting oversized children that break the
    documented contract.
    """
    # ADR D5: tables are atomic — single child equal to the parent.
    if parent.table_html is not None:
        return [_single_child_from_parent(parent, from_table=True)]

    # Degenerate small-parent case: one child equal to the parent.
    if parent.token_count <= MAX_CHILD_TOKENS:
        return [_single_child_from_parent(parent, from_table=False)]

    parent_text = parent.text
    abs_offset = parent.char_span[0]

    # Sentence-ish + block-tag boundaries. Each match's `.end()` is a
    # candidate cut point. iXBRL HTML uses `<span>` wrappers around
    # words, but punctuation typically sits outside spans, so the
    # char-class `[.!?]` followed by whitespace catches sentence ends
    # reliably. Block closers (`</p>` etc.) catch larger boundaries.
    boundary_pat = re.compile(r"(?<=[.!?])\s+|</p>|</div>|</td>|</li>", re.IGNORECASE)
    breaks = [m.end() for m in boundary_pat.finditer(parent_text)]
    if not breaks or breaks[-1] != len(parent_text):
        breaks.append(len(parent_text))

    children: list[ChildChunk] = []
    cursor = 0
    last_safe_cut = 0
    bp_index = 0
    while bp_index < len(breaks):
        cut = breaks[bp_index]
        if cut <= cursor:
            bp_index += 1
            continue
        candidate_tokens = count_tokens(parent_text[cursor:cut])
        if candidate_tokens <= MAX_CHILD_TOKENS:
            # Still under budget. Flush only when at/past MIN or at tail.
            if candidate_tokens >= MIN_CHILD_TOKENS or cut == len(parent_text):
                # Defer the flush to see if a later break gives a fuller
                # (but still-under-budget) child. Only commit once the
                # next break would exceed budget OR we reach the tail.
                last_safe_cut = cut
                bp_index += 1
                # If this is the tail, flush now.
                if cut == len(parent_text):
                    children.append(
                        ChildChunk(
                            text=parent_text[cursor:cut],
                            char_span=(abs_offset + cursor, abs_offset + cut),
                            token_count=candidate_tokens,
                            parent_id=_parent_id(parent),
                            section_name=parent.section_name,
                            from_table=False,
                            metadata=parent.metadata,
                        )
                    )
                    cursor = cut
                continue
            # Under MIN — keep accumulating; don't move cursor yet.
            last_safe_cut = cut
            bp_index += 1
            continue
        # Over budget at this cut.
        if last_safe_cut > cursor:
            # Flush at last safe cut, then re-evaluate the same `cut`.
            flushed_text = parent_text[cursor:last_safe_cut]
            children.append(
                ChildChunk(
                    text=flushed_text,
                    char_span=(abs_offset + cursor, abs_offset + last_safe_cut),
                    token_count=count_tokens(flushed_text),
                    parent_id=_parent_id(parent),
                    section_name=parent.section_name,
                    from_table=False,
                    metadata=parent.metadata,
                )
            )
            cursor = last_safe_cut
            # Re-evaluate the same bp_index against the new cursor.
            continue
        # No safe cut available — single span between two consecutive
        # boundaries exceeds MAX_CHILD_TOKENS. Splitting mid-tag breaks
        # INV-2-adjacent contracts; quarantine instead.
        raise ChunkValidationError(
            f"child subdivision failed for parent {parent.section_name!r} "
            f"at parent-relative offsets ({cursor}, {cut}): single span of "
            f"{candidate_tokens} tokens exceeds MAX_CHILD_TOKENS={MAX_CHILD_TOKENS} "
            "with no available boundary to split on. Caller routes to "
            "quarantine_chunkset rather than emitting an oversized child."
        )

    # Tail flush: if anything is buffered past the last emitted cursor,
    # emit it. This catches the case where the last break was below MIN
    # but represents the end of the document.
    if cursor < len(parent_text):
        tail_text = parent_text[cursor:]
        tail_tokens = count_tokens(tail_text)
        if tail_tokens > MAX_CHILD_TOKENS:
            raise ChunkValidationError(
                f"child subdivision failed for parent {parent.section_name!r} "
                f"tail at parent-relative offset {cursor}: {tail_tokens} tokens "
                f"exceeds MAX_CHILD_TOKENS={MAX_CHILD_TOKENS} with no boundary."
            )
        children.append(
            ChildChunk(
                text=tail_text,
                char_span=(abs_offset + cursor, abs_offset + len(parent_text)),
                token_count=tail_tokens,
                parent_id=_parent_id(parent),
                section_name=parent.section_name,
                from_table=False,
                metadata=parent.metadata,
            )
        )

    # Defensive fallback — if for some reason the loop emitted nothing,
    # return a single child equal to the parent (preserves INV-2 even
    # though it doesn't satisfy the child-size contract). This branch
    # is unreachable given the logic above but documents the contract.
    if not children:
        children.append(_single_child_from_parent(parent, from_table=False))

    return children


# ---------- Top-level entrypoint --------------------------------------------


def parse_filing(*, html: str, metadata: ChunkMetadata) -> ChunkSet:
    """Parse SEC-filing HTML into section-aware parent + child chunks.

    Pure function (modulo the one-time spaCy warmup): same `(html,
    metadata)` → same `ChunkSet`. No network, no LLM. Raises
    `ChunkValidationError` if any chunk fails the char_span identity
    test; callers route to `quarantine_chunkset` rather than persisting
    a corrupted result.

    Implementation: section boundaries come from a direct entity-aware
    scan of the raw HTML (more reliable than `unstructured`'s element
    classification for SEC 10-Ks where Item headers can land in any
    element type). `unstructured.partition_html` is imported as a
    transitively-available dependency for downstream issues; this
    function does NOT invoke it (a previous "sanity-parse" call was
    removed in code review as it neither validated nor warmed —
    `partition_html` returns gracefully on garbage input, see code
    review P2-14).

    Tables (`<table>...</table>` spans) emit as standalone ParentChunks
    with `table_html` populated. Narrative between tables packs into
    chunks ≤ MAX_PARENT_TOKENS along HTML boundary tags. Children
    subdivide each parent into 200-800-token retrieval units; table
    parents emit a single child equal to the parent (ADR D5).
    """
    _ensure_nlp_warmup()

    sections = _detect_sections(html)
    if not sections:
        # Whole document as one synthetic "Body" section.
        sections = [_DetectedSection(name="Body", char_span=(0, len(html)))]

    parents_list: list[ParentChunk] = []
    for s in sections:
        parents_list.extend(_emit_section_chunks(html, s, metadata))

    children_list: list[ChildChunk] = []
    for p in parents_list:
        children_list.extend(subdivide_to_children(p))

    validate_char_spans(html, parents_list, children_list)
    return ChunkSet(parents=tuple(parents_list), children=tuple(children_list))


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
    "validate_or_quarantine_chunkset",
]
