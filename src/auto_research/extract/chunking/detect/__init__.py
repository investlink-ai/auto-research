"""Doc-type section-detection registry.

`parse_filing` calls `get_detector(metadata.doc_type)` rather than
hardcoding a single detector. Adding a new form (10-Q, 8-K, S-1) is a
one-line registry entry plus a detector module; no change to
`parse_filing`'s body or any other call site.

Foreign filers (20-F / 40-F) are intentionally not registered in v1 —
see `docs/decisions/2026-05-25-foreign-filers-deferred.md`. The error
message from `get_detector` names that ADR so the remediation path is
obvious.
"""

from __future__ import annotations

from collections.abc import Callable

from .._types import UnsupportedDocTypeError, _DetectedSection
from ._periodic import detect_sections_periodic

DetectorFn = Callable[[str], list[_DetectedSection]]

_DETECTOR_REGISTRY: dict[str, DetectorFn] = {
    "10-K": detect_sections_periodic,
    # Issue #19 will add:
    # "10-Q": detect_sections_periodic,  # same detector, different item whitelist
    # "8-K":  detect_sections_eight_k,
    # "S-1":  detect_sections_periodic,
    # "S-3":  detect_sections_periodic,
}


def get_detector(doc_type: str) -> DetectorFn:
    """Return the section detector registered for `doc_type`.

    Raises `UnsupportedDocTypeError` (a `ValueError` subclass) with a
    remediation message naming the supported set and the foreign-
    filers ADR when `doc_type` is not registered. The typed exception
    lets `parse_filing` tag a distinct OTel outcome
    (`unsupported_doc_type`) so dashboards can split contract-routing
    failures from infra failures. The error must be loud — a silent
    fallback to a generic detector would produce ChunkSets whose
    section_name is meaningless, which silently corrupts INV-2
    downstream (LanceDB section filters return wrong rows).
    """
    try:
        return _DETECTOR_REGISTRY[doc_type]
    except KeyError as exc:
        raise UnsupportedDocTypeError(
            f"No chunker detector registered for doc_type={doc_type!r}. "
            f"Supported: {sorted(_DETECTOR_REGISTRY)}. If this is a new "
            "form, add a detector under chunking/detect/ and register it "
            "here. Foreign filers (20-F / 40-F) are intentionally not "
            "supported in v1 — see docs/decisions/2026-05-25-foreign-"
            "filers-deferred.md."
        ) from exc
