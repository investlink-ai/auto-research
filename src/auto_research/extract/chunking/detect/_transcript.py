"""Earnings-transcript Q&A boundary detection — stub.

Transcripts split into prepared-remarks vs. Q&A halves, with each
analyst's question + management's answer forming a logical unit.
Detection is structurally different from SEC Item-based forms (no
`Item N` regex applies); the implementation lives with the transcripts
ingest workstream rather than the SEC-filing one.

Keeping the stub here means the registry-entry surface from issue #57
already names the right module so future work is a fill-in rather than
a restructure.
"""

from __future__ import annotations

from .._types import _DetectedSection


def detect_sections_transcript(html: str) -> list[_DetectedSection]:
    """Detect Q&A boundaries within an earnings transcript.

    Stub. Implementation is deferred to the transcripts workstream;
    today this raises so an accidental registry-entry mistake fails
    loudly rather than silently emitting a single Body chunk.
    """
    raise NotImplementedError(
        "Transcript section detection is not yet implemented. Transcripts "
        "split into prepared-remarks vs. Q&A halves; the regex-based "
        "periodic-form detector does not apply."
    )
