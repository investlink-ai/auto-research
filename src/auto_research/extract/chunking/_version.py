"""Chunker pure-function-contract version.

`CHUNKER_VERSION` identifies the chunk-boundary contract of `parse_filing`.
Bump it whenever a change could shift `ParentChunk.char_span` /
`ChildChunk.char_span` boundaries on the same input HTML — section
detector adjustments, packing heuristic changes, token-budget tweaks
that move splits, table-emission policy changes, etc.

Why this exists (INV-6 generalized one layer up):

The contextual-chunking cache keys per-child outputs by
`(raw_chunk_text, parent_text, CONTEXTUAL_CHUNK_PROMPT_VERSION, ...)`.
A chunker change that shifts boundaries produces *new* child texts and
thus naturally misses the contextual cache for those re-bounded chunks
— but a chunker change that keeps a given child text identical while
re-bounding its neighbors silently reuses the old contextual context,
which was written against the OLD parent / sibling layout. Folding
`CHUNKER_VERSION` into the contextual cache key removes this subtle
silent-reuse path: any chunker contract change invalidates all
downstream contextual cache entries.

At 90x2 backfill scope this is theoretical; at 1000x10 scope it's a
correctness blocker.

This constant is the chunker analogue of `*_PROMPT_VERSION` and is
covered by the same `bump-prompt-version` skill workflow. The cache
key it feeds is in `extract/chunking_contextual._cache_payload_key`.
"""

from __future__ import annotations

from typing import Final

CHUNKER_VERSION: Final[str] = "v1"
"""Bump when this artifact's pure-function contract changes.

Bump triggers:
- Section-detection regex / classifier changes (affects parent boundaries).
- Token-budget constants in `_tokens.py` that move child splits.
- `_packing` / `_tables` policy changes that re-assign text to parents.

Non-triggers (do NOT bump):
- Pure refactors that produce byte-identical `ChunkSet` outputs.
- Comment / docstring edits.
- Adding fields to `ChunkMetadata` that are not read by chunking logic.
"""


__all__ = ["CHUNKER_VERSION"]
