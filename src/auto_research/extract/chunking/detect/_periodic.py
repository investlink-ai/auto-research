"""Section detection for SEC periodic forms (10-K today; 10-Q later).

The detector scans the raw HTML directly (not via `unstructured`'s
element stream) because `unstructured`'s text classification is
unreliable for SEC 10-K headers — Items can be classified as `Title`,
`Text`, or even buried inside a larger `NarrativeText` element depending
on the surrounding markup. The entity-aware regex matches `Item N.` /
`Item&#160;N.` patterns directly.

Item-number whitelist is form-specific. The default (`_VALID_10K_ITEMS`)
covers Form 10-K Items per the SEC's Form 10-K instructions
(https://www.sec.gov/files/form10-k.pdf). Issue #19 will register the
same detector for 10-Q with a different whitelist; the function accepts
`valid_items` as an optional override so the registry can carry one
detector instance with closure-bound item sets per form.
"""

from __future__ import annotations

import re
from typing import Final

from .._types import _DetectedSection
from ._common import (
    _BARE_TITLE_PATTERNS,
    _HEADER_DENSITY_MIN_ALPHA,
    _HEADER_DENSITY_WINDOW_BYTES,
    _ITEM_HEADER,
    _is_in_open_anchor,
    _is_real_section_header,
    _looks_like_block_header,
    _mask_comments,
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


def detect_sections_periodic(
    html: str,
    *,
    valid_items: frozenset[str] = _VALID_10K_ITEMS,
) -> list[_DetectedSection]:
    """Detect SEC Item sections by scanning the raw HTML.

    Filters applied (in order):
      1. Only valid Item numbers (per `valid_items`) — drops accidental
         matches like "Item 408 of Regulation S-K".
      2. Drop matches inside HTML comments (`<!-- ... -->`) — used by
         test fixtures' truncation markers and other meta-content.
      3. `_is_real_section_header` — ≥200 alphabetic chars of prose
         follow the candidate header (drops TOC entries pointing at
         later content).

    Returns sections in document order with absolute char_span tiles
    covering the whole document (last section's end = len(html)).
    """
    masked = _mask_comments(html)

    # Collect first qualifying occurrence of each valid Item.
    #
    # Filter strategy:
    #   1. Whitelist filter (`valid_items`) — drops rule references
    #      like "Item 408 of Regulation S-K".
    #   2. Title-pattern check (`_is_real_section_header`) — requires
    #      ". <Capitalized Title>" after the Item-N number. This is
    #      the primary filter; it rejects cross-references like
    #      "Item 7 above" / "Item 7 of this Form" mechanically.
    #
    # We do NOT require `_looks_like_block_header` (block/styled
    # preceding tag). Filers' header styling varies dramatically —
    # NVDA uses `<div>` wrappers, AMD/AVGO use bold styled `<span>`,
    # MSFT puts headers in `<td>` cells of layout tables with
    # `font-weight:normal`. A structural pre-filter that handles all
    # templates would need per-filer special cases; the title-pattern
    # check filters out cross-references regardless of styling.
    by_name: dict[str, int] = {}
    for m in _ITEM_HEADER.finditer(masked):
        num = m.group(1).upper()
        if num not in valid_items:
            continue
        if not _is_real_section_header(html, m.start()):
            continue
        name = f"Item {num}"
        by_name.setdefault(name, m.start())

    # Fallback: bare-title detection. Some filers (HON, BE, NRG, CEG,
    # QUBT) mark section bodies with the canonical SEC title only
    # ("RISK FACTORS") instead of "Item 1A. Risk Factors". For each
    # Item not already detected via the prefix scan, look for the
    # canonical bare title in a styled-header context that isn't
    # inside a TOC anchor.
    for pattern, num in _BARE_TITLE_PATTERNS:
        name = f"Item {num}"
        if name in by_name:
            continue
        for m in pattern.finditer(masked):
            pos = m.start()
            if _is_in_open_anchor(html, pos):
                continue
            if not _looks_like_block_header(html, pos):
                continue
            # Density check: substantial prose follows the title.
            # We use the same window/skip as the prefix path —
            # the candidate isn't bleeding into the threshold here
            # (bare title is short; rendered text starts past it).
            window_start = m.end()
            snippet = html[window_start : window_start + _HEADER_DENSITY_WINDOW_BYTES]
            stripped = re.sub(r"<[^>]+>|&[^;]+;", " ", snippet)
            if sum(1 for c in stripped if c.isalpha()) < _HEADER_DENSITY_MIN_ALPHA:
                continue
            by_name[name] = pos
            break

    if not by_name:
        return []

    ordered = sorted(by_name.items(), key=lambda kv: kv[1])
    sections: list[_DetectedSection] = []
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(html)
        sections.append(_DetectedSection(name=name, char_span=(start, end)))
    return sections
