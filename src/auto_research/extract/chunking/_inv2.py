"""INV-2 (citation-grounding) defenses for the chunking layer.

Three callables form the contract:

- `validate_char_spans` — runtime guard asserting `chunk.text ==
  source[char_span]` for every parent and child.
- `quarantine_chunkset` — write a JSON audit record under
  `<root>/chunking/<doc_id>.json` so humans can review the failure.
- `validate_or_quarantine_chunkset` — single-call routing helper that
  callers wrap their `parse_filing` result with.

Mirrors `extract.guardrails`'s discipline for the extraction half of
INV-2. INV-2 holds at write-time via the chunker's `char_span`
construction; these helpers are defense-in-depth, not the load-bearing
guarantee.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from auto_research._io import atomic_write_text

from ._types import ChildChunk, ChunkSet, ChunkValidationError, ParentChunk

DEFAULT_QUARANTINE_ROOT: Final[Path] = Path("data/quarantine")


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
