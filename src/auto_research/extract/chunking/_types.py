"""Frozen dataclasses for the chunking package.

Public types (`ChunkMetadata`, `ParentChunk`, `ChildChunk`, `ChunkSet`,
`ChunkValidationError`) are re-exported from `auto_research.extract.chunking`
and form the package's narrow contract with downstream consumers. The
internal `_DetectedSection` describes the (name, char_span) tuple that
section detectors return.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class ChunkValidationError(ValueError):  # typed contract name
    """Raised when a chunk's char_span doesn't slice back to its text.

    Mirrors `extract.guardrails.CitationMismatch` for the chunking
    layer's half of INV-2. Subclasses `ValueError` so generic callers
    can match on the base type; typed so chunking-aware callers can
    route to the chunking quarantine path without catching unrelated
    `ValueError`s.
    """


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
