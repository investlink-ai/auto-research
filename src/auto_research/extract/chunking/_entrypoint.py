"""`parse_filing` — the top-level chunking entrypoint.

Pure dispatcher: warm NLP, resolve a section detector from the doc-type
registry, build parent + child chunks, validate INV-2 char_span
fidelity, and wrap the whole thing in an OTel span carrying the
chunker-specific outcome enum. The OTel discipline mirrors the
extraction workers' span layout (one manual span at the orchestration
boundary so dashboards can filter chunker failures separately from
extraction failures).

Detector lookup is via `chunking.detect.get_detector(doc_type)`; the
registry maps `"10-K" → detect_sections_periodic` today and grows
one-line entries per form as issue #19 adds 10-Q/8-K support.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

import auto_research.extract.chunking as _pkg  # late-bound for test monkeypatching
from auto_research.telemetry import truncate_status_description as _truncate

from ._nlp_warmup import _ensure_nlp_warmup
from ._packing import subdivide_to_children
from ._tables import _emit_section_chunks
from ._types import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ChunkValidationError,
    ParentChunk,
    UnsupportedDocTypeError,
    _DetectedSection,
)
from .detect import get_detector

# Manual span at the chunking orchestration boundary. Auto-instrumentation
# covers SDK calls; chunking is a sub-step inside extraction workers, so
# this span nests under the worker's outer span (e.g. `extract.s_filings`)
# and reports the chunker's own outcome enum: dashboards can filter
# "chunking failures" separately from "extraction failures" without
# walking the full trace tree.
#
# Tracer scope is pinned to the package name (not `__name__`) so the
# refactor from monolithic chunking.py to chunking/_entrypoint.py does
# not silently shift OTel `instrumentation_scope.name` from
# `auto_research.extract.chunking` to `auto_research.extract.chunking
# ._entrypoint`. Dashboards / Langfuse queries that key on scope name
# continue to match pre-refactor traces.
_tracer = trace.get_tracer("auto_research.extract.chunking")


def parse_filing(*, html: str, metadata: ChunkMetadata) -> ChunkSet:
    """Parse SEC-filing HTML into section-aware parent + child chunks.

    Pure function (modulo the one-time spaCy warmup): same `(html,
    metadata)` → same `ChunkSet`. No network, no LLM. Raises
    `ChunkValidationError` if any chunk fails the char_span identity
    test; callers route to `quarantine_chunkset` rather than persisting
    a corrupted result.

    Implementation: section boundaries come from a doc-type-specific
    detector resolved via `chunking.detect.get_detector(metadata
    .doc_type)`. Unknown / unregistered `doc_type` raises
    `UnsupportedDocTypeError` (a `ValueError` subclass) with a
    remediation message — silently emitting one Body chunk would
    corrupt downstream LanceDB section filters.

    Tables (`<table>...</table>` spans) emit as standalone ParentChunks
    with `table_html` populated. Narrative between tables packs into
    chunks ≤ MAX_PARENT_TOKENS along HTML boundary tags. Children
    subdivide each parent into 200-800-token retrieval units; table
    parents emit a single child equal to the parent (ADR D5).

    Emits a manual OTel span `chunk.parse_filing` with attributes:
    `chunk.doc_id`, `chunk.doc_type`, `chunk.ticker`, `chunk.n_sections`,
    `chunk.n_parents`, `chunk.n_children`, `chunk.n_table_parents`,
    `chunk.outcome` (`ok` / `no_sections_detected` /
    `unsupported_doc_type` / `validation_failed` / `subdivision_failed`
    / `error`). See `docs/ARCHITECTURE.md` §5.3.
    """
    with _tracer.start_as_current_span("chunk.parse_filing") as span:
        span.set_attribute("chunk.doc_id", metadata.doc_id)
        span.set_attribute("chunk.doc_type", metadata.doc_type)
        span.set_attribute("chunk.ticker", metadata.ticker)
        try:
            _ensure_nlp_warmup()

            try:
                detector = get_detector(metadata.doc_type)
            except UnsupportedDocTypeError as exc:
                # Contract failure (caller passed an unregistered doc_type)
                # — distinct from INV-2 / infra failures so dashboards can
                # route foreign-filer / unsupported-form ingest separately
                # from spaCy-missing or programmer-error pages.
                span.set_attribute("chunk.outcome", "unsupported_doc_type")
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise
            sections = detector(html)
            if not sections:
                # Whole document as one synthetic "Body" section.
                sections = [_DetectedSection(name="Body", char_span=(0, len(html)))]
                outcome = "no_sections_detected"
            else:
                outcome = "ok"
            span.set_attribute("chunk.n_sections", len(sections))

            parents_list: list[ParentChunk] = []
            for s in sections:
                parents_list.extend(_emit_section_chunks(html, s, metadata))

            children_list: list[ChildChunk] = []
            try:
                for p in parents_list:
                    children_list.extend(subdivide_to_children(p))
            except ChunkValidationError as exc:
                # `subdivide_to_children` raises only for pathological
                # spans (>MAX_UNBREAKABLE_CHILD_TOKENS with no boundary)
                # — distinct from the char_span check below. Tag the
                # outcome before letting the caller route to quarantine.
                span.set_attribute("chunk.outcome", "subdivision_failed")
                span.set_attribute("chunk.n_parents", len(parents_list))
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

            try:
                # Late-bound through the package namespace so tests can
                # monkeypatch `chunking.validate_char_spans` to inject
                # synthetic failures into this span's error path.
                _pkg.validate_char_spans(html, parents_list, children_list)
            except ChunkValidationError as exc:
                span.set_attribute("chunk.outcome", "validation_failed")
                span.set_attribute("chunk.n_parents", len(parents_list))
                span.set_attribute("chunk.n_children", len(children_list))
                span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
                raise

            span.set_attribute("chunk.n_parents", len(parents_list))
            span.set_attribute("chunk.n_children", len(children_list))
            span.set_attribute(
                "chunk.n_table_parents",
                sum(1 for p in parents_list if p.table_html is not None),
            )
            span.set_attribute("chunk.outcome", outcome)
            return ChunkSet(
                parents=tuple(parents_list),
                children=tuple(children_list),
            )
        except (ChunkValidationError, UnsupportedDocTypeError):
            # Outcome + status already set on the typed-error branches above.
            raise
        except Exception as exc:
            # Anything else (NLP-warmup failure, programmer error) is an
            # infra failure — tag distinctly so dashboards don't count it
            # as a chunker contract violation.
            span.set_attribute("chunk.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
            raise
