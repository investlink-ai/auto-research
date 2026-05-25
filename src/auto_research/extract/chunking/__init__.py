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

- Section detection is dispatched via `chunking.detect.get_detector(doc_type)`.
  10-K (and forthcoming 10-Q, per issue #19) share a periodic-form
  detector that uses an entity-aware regex on the raw HTML — tolerating
  `&#160;` / `&nbsp;` between `Item` and the number, since iXBRL
  filings commonly emit non-breaking-space entities at section headers.
  `unstructured.partition.html` is imported transitively for downstream
  callers; this package does NOT invoke it.

- 10-K Item 8 tables emit as standalone `ParentChunk`s with the raw
  `<table>...</table>` HTML attached on `table_html`. Downstream
  structured extraction reads `table_html` via a typed Pydantic
  schema, bypassing dense retrieval. Nested `<table>` depth is tracked
  so `table_html` always covers the full outer table.

- `chunk.text == html[char_span]` by construction. INV-2 holds via
  this identity; `validate_char_spans` is a defense-in-depth runtime
  check, not the load-bearing guarantee.

Children carry `parent_id = "{doc_id}::{parent.char_span[0]}-{parent
.char_span[1]}"`, `section_name` (copied from the parent so LanceDB
can filter at index time without a parent JOIN), and `from_table`
(True iff the child was emitted under a table parent — table parents
are atomic, so the child equals the parent in that case).

Module layout (split per issue #57 from the prior monolithic
`chunking.py`):

  - `_types.py`      — ChunkMetadata / ParentChunk / ChildChunk / ChunkSet
                       / ChunkValidationError / _DetectedSection
  - `_tokens.py`     — count_tokens + MAX_*_TOKENS constants
  - `_inv2.py`       — validate_char_spans / quarantine_chunkset / etc.
  - `_nlp_warmup.py` — _ensure_nlp_warmup (spaCy)
  - `_packing.py`    — _pack_narrative_html + subdivide_to_children
  - `_tables.py`     — _emit_section_chunks (ADR D5 table policy)
  - `_entrypoint.py` — parse_filing dispatcher
  - `detect/`        — doc-type → detector registry + per-form modules

Names re-exported with `as X` aliases are deliberate public-from-package
re-exports (ruff's F401 treats this idiom as intentional); scripts /
tests import them via this module and never via the underscore-
prefixed submodules.
"""

from __future__ import annotations

# Import order is alphabetical by module name (ruff isort), which puts
# `_entrypoint` first; this is safe because `_entrypoint.parse_filing`
# resolves `validate_char_spans` at CALL time via the package namespace
# (`import auto_research.extract.chunking as _pkg` in `_entrypoint.py`).
# By the time `parse_filing` is invoked, the rest of this file has
# finished and the symbol is bound. The call-time indirection also
# preserves the test monkeypatch contract: a test that patches
# `chunking.validate_char_spans` sees the override flow through
# `parse_filing` because the entrypoint never captures a local
# reference. The only failure mode is a future module imported by
# `_entrypoint` calling `parse_filing` at import time — currently no
# such call exists.
from ._entrypoint import parse_filing as parse_filing

# Foundational public surface — types, constants, INV-2 helpers.
from ._inv2 import DEFAULT_QUARANTINE_ROOT as DEFAULT_QUARANTINE_ROOT
from ._inv2 import quarantine_chunkset as quarantine_chunkset
from ._inv2 import validate_char_spans as validate_char_spans
from ._inv2 import validate_or_quarantine_chunkset as validate_or_quarantine_chunkset

# Internal helpers re-exported because the prior monolithic
# `chunking.py` exposed them on the package surface; the
# `_ensure_nlp_warmup` symbol is read directly by `tests/unit/conftest
# .py`, and `_detect_sections` (below) by `scripts/build_chunking_
# fixture.py`. The other underscore-prefixed re-exports are kept so the
# refactor preserves the prior `dir(chunking)` surface for any caller
# that probed it.
from ._nlp_warmup import _ensure_nlp_warmup as _ensure_nlp_warmup
from ._packing import _pack_narrative_html as _pack_narrative_html
from ._packing import _parent_id as _parent_id
from ._packing import _single_child_from_parent as _single_child_from_parent
from ._packing import subdivide_to_children as subdivide_to_children
from ._tables import _emit_section_chunks as _emit_section_chunks
from ._tables import _find_matching_table_close as _find_matching_table_close
from ._tokens import MAX_CHILD_TOKENS as MAX_CHILD_TOKENS
from ._tokens import MAX_PARENT_TOKENS as MAX_PARENT_TOKENS
from ._tokens import MAX_UNBREAKABLE_CHILD_TOKENS as MAX_UNBREAKABLE_CHILD_TOKENS
from ._tokens import MIN_CHILD_TOKENS as MIN_CHILD_TOKENS
from ._tokens import SINGLE_SHOT_TOKEN_CUTOFF as SINGLE_SHOT_TOKEN_CUTOFF
from ._tokens import count_tokens as count_tokens
from ._types import ChildChunk as ChildChunk
from ._types import ChunkMetadata as ChunkMetadata
from ._types import ChunkSet as ChunkSet
from ._types import ChunkValidationError as ChunkValidationError
from ._types import ParentChunk as ParentChunk
from ._types import UnsupportedDocTypeError as UnsupportedDocTypeError

# Section-detection internals re-exported for scripts that validate the
# regex against external sources: `scripts/validate_10k_items.py`
# pulls the Item whitelist + header classifiers; `scripts/build_
# chunking_fixture.py` calls `_detect_sections` to summarize trim
# results.
from .detect._common import _ITEM_HEADER as _ITEM_HEADER
from .detect._common import _is_real_section_header as _is_real_section_header
from .detect._common import _looks_like_block_header as _looks_like_block_header
from .detect._common import _mask_comments as _mask_comments
from .detect._periodic import _VALID_10K_ITEMS as _VALID_10K_ITEMS
from .detect._periodic import (
    detect_sections_periodic as _detect_sections,  # noqa: F401  — back-compat alias for scripts/build_chunking_fixture.py
)

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
    "UnsupportedDocTypeError",
    "count_tokens",
    "parse_filing",
    "quarantine_chunkset",
    "subdivide_to_children",
    "validate_char_spans",
    "validate_or_quarantine_chunkset",
]
