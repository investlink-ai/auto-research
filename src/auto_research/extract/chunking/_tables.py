"""ADR D5 table policy — emit `<table>...</table>` regions as standalone
ParentChunks with `table_html` populated.

Section-crossing tables (a `<table>` opens within the section but
`</table>` closes after) are NOT emitted as table chunks. Clamping their
span to the section end would produce malformed `table_html` (open
`<table>` with no close), violating ADR D5's well-formed-HTML contract.
Cross-boundary tables fall through into the narrative pass.
"""

from __future__ import annotations

import re

from ._packing import _pack_narrative_html
from ._tokens import count_tokens
from ._types import ChunkMetadata, ParentChunk, _DetectedSection

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
