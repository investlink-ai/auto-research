"""8-K item-code section detection — stub.

Form 8-K uses dotted Item codes (Item 1.01, 2.02, …) drawn from the
SEC's Form 8-K instructions; the periodic-form detector's whitespace
regex does not match the dotted pattern. Implementation lives with
issue #19, which adds the regex + whitelist and registers this
detector against `doc_type="8-K"`.

Keeping the stub here means the registry-entry surface from issue #57
already names the right module, so #19 is a fill-in rather than a
restructure.
"""

from __future__ import annotations

from .._types import _DetectedSection


def detect_sections_eight_k(html: str) -> list[_DetectedSection]:
    """Detect 8-K Item codes (Item 1.01, 2.02, …).

    Stub. Issue #19 will implement the dotted-Item regex against
    Form 8-K's whitelist; today this raises so an accidental
    registry-entry mistake fails loudly rather than silently emitting
    a single Body chunk.
    """
    raise NotImplementedError(
        "8-K section detection is not yet implemented — see issue #19. "
        "Form 8-K uses dotted Item codes (Item 1.01, 2.02, …) that the "
        "periodic-form detector does not match."
    )
