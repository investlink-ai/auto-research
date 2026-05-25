"""Shared section-detection primitives.

Regex tokens, header-context heuristics, and entity-aware helpers used
by all form-specific detectors. Kept in one module so that adding a new
form (10-Q, 8-K, S-1) does not duplicate header-classification logic.
"""

from __future__ import annotations

import re
from typing import Final

# ---------- Item-header regex ----------------------------------------------

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

# Number of preceding chars inspected by `_looks_like_block_header` to
# decide if a candidate Item match starts a structural header (vs. an
# inline cross-reference like "compared to Item 7 above").
_BLOCK_HEADER_LOOKBACK: Final[int] = 80

# Lookback when checking if a candidate position sits inside an open
# `<a>` tag (TOC link). 400 chars covers the longest `<a style="...">`
# opening attribute lists in SEC iXBRL.
_ANCHOR_LOOKBACK: Final[int] = 400


# ---------- Bare-title fallback --------------------------------------------

# Bare section-title → Item number mapping. Some filers (HON, BE, NRG,
# CEG, QUBT) put section bodies under their canonical SEC title only
# ("RISK FACTORS"), without the "Item 1A." prefix. The periodic detector
# runs a fallback pass for any Item NOT found via the prefix scan,
# looking for the bare title in a styled-header context.
#
# The catalog mirrors SEC Form 10-K canonical titles. Longer titles
# come first within an Item so the longer string wins when it's a
# superset (avoids "Management's Discussion and Analysis" being
# shadowed by "Management's Discussion" if both appear).
_BARE_TITLE_TO_ITEM: Final[dict[str, str]] = {
    "Risk Factors": "1A",
    "Unresolved Staff Comments": "1B",
    "Cybersecurity": "1C",
    "Properties": "2",
    "Legal Proceedings": "3",
    "Mine Safety Disclosures": "4",
    "Management's Discussion and Analysis of Financial Condition "
    "and Results of Operations": "7",
    "Management's Discussion and Analysis": "7",
    "Quantitative and Qualitative Disclosures About Market Risk": "7A",
    "Quantitative and Qualitative Disclosures": "7A",
    "Financial Statements and Supplementary Data": "8",
    "Controls and Procedures": "9A",
}

# Pre-compile bare-title patterns once. Word-boundaries on both sides
# so "Risk Factors" doesn't match inside "Risk Factors Summary".
_BARE_TITLE_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(rf"\b{re.escape(title)}\b", re.IGNORECASE), item)
    for title, item in sorted(
        _BARE_TITLE_TO_ITEM.items(), key=lambda kv: -len(kv[0])
    )
]


# ---------- Title-pattern + density windows -------------------------------

# How many bytes after the candidate header to skip before counting
# prose density — the header itself ("Item 1A. Risk Factors") can easily
# contribute 50+ alpha chars and falsely boost the density score for TOC
# echoes packed adjacent to other Item names. The skip is short enough to
# still capture a real section's opening sentence inside the 2000-char
# window.
#
# The threshold (80) is tuned to admit table-heavy sections like Item 8
# (where AMD's filing jumps straight into Consolidated Statements with
# ~150 alpha chars in 2 KB — almost all numbers and table markup) while
# still rejecting TOC entries (typically <50 alpha chars in 2 KB —
# Item name + page number padded with leader dots/whitespace). The
# title-pattern check (`. <Capitalized Title>` after the Item number)
# does most of the false-positive filtering; this density check is
# defense-in-depth against TOC entries that happen to carry title text.
_HEADER_DENSITY_SKIP_BYTES: Final[int] = 200
_HEADER_DENSITY_WINDOW_BYTES: Final[int] = 2_000
_HEADER_DENSITY_MIN_ALPHA: Final[int] = 80

# A real Item header is followed by a separator then `<SECTION TITLE>`
# — period, colon, em-dash, en-dash, or hyphen — then optional
# whitespace, then a capitalized title (Risk Factors, Management's
# Discussion, etc.). Cross-references say "Item 7 above" / "Item 7 of
# this Form" — no separator, lowercase preposition. Inspected after
# stripping HTML tags and entities so the title check works regardless
# of how the filer styled the heading. The 800-char raw-HTML window is
# sized to capture the closing `>` of multi-line `<p style="...">`
# openings common in MSFT-template filings (style attribute can run
# 100+ characters before the tag closes).
_HEADER_TITLE_LOOKAHEAD: Final[int] = 800

# Separator chars accepted in title pattern: period, colon, em dash
# (U+2014), en dash (U+2013), hyphen-minus. The non-ASCII dashes are
# intentional separators used by filers like BE and NRG
# (`Item 1A&#8212;Risk Factors`). Defined via `\uXXXX` escape
# sequences so the source file stays pure ASCII (ruff's RUF001
# ambiguous-character lint flags literal em/en dashes as visually
# confusable with hyphen-minus, and escape form sidesteps it).
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"
_TITLE_PATTERN = re.compile(
    "^\\s*[.:" + _EM_DASH + _EN_DASH + "-]\\s*[A-Z]"
)

# Common HTML-entity dash variants. These must be preserved through
# the entity-strip in `_is_real_section_header` because they ARE the
# header separator in some filings (BE uses `Item 1A&#8212;Risk
# Factors`, etc.). Mapped to the literal Unicode dash chars (via the
# `_EM_DASH` / `_EN_DASH` escapes above) so the title-pattern's
# character class sees them.
_DASH_ENTITY_MAP: Final[dict[str, str]] = {
    "&#8212;": _EM_DASH,
    "&#8211;": _EN_DASH,
    "&mdash;": _EM_DASH,
    "&ndash;": _EN_DASH,
}


# ---------- Block-tag context for header validation -----------------------

# Block-level HTML tags that mark structural boundaries in SEC filings.
# Section headers typically open inside or immediately after one of
# these; unstyled inline formatting tags (`<span>`, `<a>`, `<em>`) do
# NOT count alone — accepting them as structural causes inline cross-
# references like `... see <span>Item 7</span> ...` to false-positive
# as section starts.
_BLOCK_TAGS = "(?:div|p|td|tr|li|h[1-6]|section|article|header|footer|main|body|table|tbody|thead|tfoot)"

_BLOCK_CLOSE_RE = re.compile(rf"</{_BLOCK_TAGS}[^>]*>\s*$", re.IGNORECASE)
_BLOCK_OPEN_RE = re.compile(rf"<{_BLOCK_TAGS}[^>]*>\s*$", re.IGNORECASE)
_COMMENT_END_RE = re.compile(r"-->\s*$")

# Many filers (AMD, AVGO, others) style their section headers with an
# inline `<span style="font-weight:700">Item N. ...</span>` rather than
# a block tag. The unstyled `<span>` is rejected by `_BLOCK_OPEN_RE`
# (which doesn't list span/b/strong/font); the styled one is accepted
# here. Cross-references inside running prose use unstyled `<span>` so
# they stay rejected.
#
# Anchored only at the closing `>` of the styling tag — the lookback
# window (80 chars) is shorter than typical `<span style="...">` opening
# tags in iXBRL HTML (~100 chars of attributes), so we cannot require
# the `<span` opening literally. The signal we rely on is the substring
# `font-weight:700` (or 800/900/bold) followed by `">` within the
# lookback window. False positives would require the literal text
# `font-weight:700">` to appear outside an HTML tag, which doesn't
# happen in normal SEC content.
_STYLED_HEADER_RE = re.compile(
    r"font-weight\s*[:=]\s*['\"]?(?:700|800|900|bold|bolder)['\"]?[^>]*>\s*$",
    re.IGNORECASE,
)


# ---------- Entity-flexible HTML search ------------------------------------

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


def _section_name_from_title(title_text: str) -> str | None:
    """Return `"Item N"` if `title_text` looks like a SEC Item heading."""
    m = _ITEM_HEADER.match(title_text.strip())
    if m is None:
        return None
    return f"Item {m.group(1).upper()}"


# ---------- Comment masking ------------------------------------------------


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


# ---------- Header-context classifiers -------------------------------------


def _is_in_open_anchor(html: str, pos: int) -> bool:
    """Return True if `pos` sits inside an open `<a ...>` tag (TOC link).

    Scans back `_ANCHOR_LOOKBACK` chars: if the nearest `<a` opening
    occurs AFTER the nearest `</a>` closing, we're inside an anchor.
    SEC filings put TOC links inside `<a href="#...">` — bare section
    titles inside those anchors are navigation, not real headers.
    """
    window_start = max(0, pos - _ANCHOR_LOOKBACK)
    window = html[window_start:pos]
    # `<a ` (with whitespace) — avoids matching `<abbr>`, `<address>`, etc.
    last_open = max(window.rfind("<a "), window.rfind("<a\t"), window.rfind("<a\n"))
    last_close = window.rfind("</a>")
    return last_open > last_close


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
        or _STYLED_HEADER_RE.search(preceding)
    )


def _is_real_section_header(html: str, span_start: int) -> bool:
    """A candidate is a "real" section start when both:

    1. The next `_HEADER_DENSITY_WINDOW_BYTES` of raw HTML (skipping
       the first `_HEADER_DENSITY_SKIP_BYTES`) contain ≥
       `_HEADER_DENSITY_MIN_ALPHA` alphabetic characters of prose
       (drops TOC echoes that have no real section body).
    2. The text immediately following the candidate (after tag/entity
       stripping) looks like `. <CAPITALIZED TITLE>` — the section-
       title pattern. Cross-references like `Item 7 of this Form` or
       `Item 7 above` fail this check because they lack the period +
       title structure.
    """
    # Where does the Item-N match end in the raw HTML? Step past the
    # number and any trailing entities/whitespace to find the start of
    # whatever follows.
    after = _ITEM_HEADER.match(html, span_start) or _ITEM_HEADER.search(html, span_start)
    if after is None:
        return False
    title_window = html[after.end() : after.end() + _HEADER_TITLE_LOOKAHEAD]
    # Strip HTML tags first, then map dash entities to their literal
    # character (so the title pattern can see them as separators),
    # then collapse the remaining entities to whitespace.
    title_clean = re.sub(r"<[^>]+>", "", title_window)
    for entity, char in _DASH_ENTITY_MAP.items():
        title_clean = title_clean.replace(entity, char)
    title_clean = re.sub(r"&[^;]+;", " ", title_clean)
    if not _TITLE_PATTERN.match(title_clean):
        return False

    window_start = span_start + _HEADER_DENSITY_SKIP_BYTES
    snippet = html[window_start : window_start + _HEADER_DENSITY_WINDOW_BYTES]
    stripped = re.sub(r"<[^>]+>|&[^;]+;", " ", snippet)
    alpha = sum(1 for c in stripped if c.isalpha())
    return alpha >= _HEADER_DENSITY_MIN_ALPHA
