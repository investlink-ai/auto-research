"""Narrative packing + child subdivision.

Two pure-text functions whose outputs INV-2 holds for by construction
(every emitted chunk's `text == html[char_span]`):

- `_pack_narrative_html` — greedy-pack an HTML slice into ParentChunks
  ≤ MAX_PARENT_TOKENS along natural HTML break boundaries.
- `subdivide_to_children` — split a parent into 200-800-token children
  along sentence + block-tag boundaries (table parents return one
  child equal to the parent per ADR D5).
"""

from __future__ import annotations

import re

from ._tokens import (
    MAX_CHILD_TOKENS,
    MAX_PARENT_TOKENS,
    MAX_UNBREAKABLE_CHILD_TOKENS,
    MIN_CHILD_TOKENS,
    count_tokens,
)
from ._types import ChildChunk, ChunkMetadata, ChunkValidationError, ParentChunk
from .detect._common import _mask_comments

# Natural HTML break boundaries for narrative packing — closing tags
# that mark logical content units (paragraphs, divs, table rows, list
# items, headings). The regex finds the END of each closing tag so
# breakpoints fall at semantic seams, never mid-tag.
_HTML_BREAK = re.compile(r"</(?:p|div|td|tr|li|h[1-6])\s*>", re.IGNORECASE)


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
    scan from `_mask_comments(html)` so it agrees with the section
    detector (`detect_sections_periodic`) about what counts as
    document structure.

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
        # INV-2-adjacent contracts. Two paths:
        #   (a) candidate_tokens ≤ MAX_UNBREAKABLE_CHILD_TOKENS: emit
        #       the oversized child as-is. Real iXBRL filings (BE,
        #       others) have legitimate paragraphs slightly above MAX
        #       with no `</p>`/sentence boundaries inside; failing
        #       these loses the whole ChunkSet for a chunk that's
        #       only modestly over budget.
        #   (b) candidate_tokens > MAX_UNBREAKABLE_CHILD_TOKENS:
        #       raise — a runaway span this large indicates upstream
        #       parser breakage rather than legitimate filing
        #       structure. The caller routes to quarantine.
        if candidate_tokens <= MAX_UNBREAKABLE_CHILD_TOKENS:
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
            bp_index += 1
            continue
        raise ChunkValidationError(
            f"child subdivision failed for parent {parent.section_name!r} "
            f"at parent-relative offsets ({cursor}, {cut}): single span of "
            f"{candidate_tokens} tokens exceeds "
            f"MAX_UNBREAKABLE_CHILD_TOKENS={MAX_UNBREAKABLE_CHILD_TOKENS} "
            "with no available boundary to split on. Caller routes to "
            "quarantine_chunkset rather than emitting a pathological child."
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
