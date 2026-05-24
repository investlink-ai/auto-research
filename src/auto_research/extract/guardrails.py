"""Citation-grounding post-validator + quarantine router (INV-2).

The validator walks the output's model tree and asserts that every
`Citation`'s `source_quote` is the verbatim slice
`source_text[source_span[0]:source_span[1]]`. The walker is generic over
schema shape: it finds `Citation`s nested via `Claim` (e.g.,
`TenKOutput.guidance_tone.citation`) *and* `Citation`s nested directly
(e.g., `SupplierMention.citation`). Adding a new schema field is
non-breaking — the walker discovers it on the next call without code
changes here.

A `CitationMismatch` failure is **terminal for the output**: the routing
helper `validate_or_quarantine` writes a `QuarantineRecord` to
`data/quarantine/<worker>/<doc_id>.json` and returns `None`. Callers
that get `None` must NOT persist any part of the output to
`data/extracted/` or to Feast — that's how silent degradation gets in.
The skill `citation-check` enforces this contract grep-style; the
`test_no_disabling_flags_on_public_api` meta-test enforces it
in-process.

Why no `permissive` / `soft_mode` / `skip_validation` flag exists:

- The contract's value comes from being absolute. A single `permissive=True`
  test fixture that leaks into production wires INV-2 to a boolean nobody
  reviews. The acceptance criteria for Issue #9 explicitly forbid such
  a flag.
- If a future use case needs to inspect a quarantined output (e.g., a
  human reviewer), read the `QuarantineRecord` JSON directly. Don't
  short-circuit the validator.

`CitationMismatch` subclasses `ValueError` so callers can pattern-match
on it without depending on this module's specific exception type, but it
remains typed so production code (workers, eval suites) can distinguish
"INV-2 violation; quarantine" from "generic data validation issue".
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from auto_research.extract.schemas import Citation

# Default quarantine root. Callers in production omit `quarantine_root` and
# get this; tests pass `tmp_path` to keep the suite hermetic.
DEFAULT_QUARANTINE_ROOT: Final[Path] = Path("data/quarantine")


class CitationMismatch(ValueError):  # noqa: N818  # typed contract name per citation-check skill
    """Raised when a `Citation`'s `source_quote` doesn't match
    `source_text[source_span[0]:source_span[1]]`.

    Subclasses `ValueError` (it IS a data-validity failure) but is typed
    so workers can `except CitationMismatch:` for the quarantine route
    without catching unrelated `ValueError`s from elsewhere.
    """


def _walk_citations(
    value: Any,
    path: str = "",
) -> Iterator[tuple[str, Citation]]:
    """Yield `(field_path, citation)` for every `Citation` reachable in the
    output tree. Walks nested `BaseModel`, `list`, and `tuple`. The
    `field_path` is included so a `CitationMismatch` message can pinpoint
    which field hallucinated — without it, debugging a 10-K with dozens
    of claims is hopeless.
    """
    if isinstance(value, Citation):
        yield path, value
        return
    if isinstance(value, BaseModel):
        for name, field_value in value:
            sub = f"{path}.{name}" if path else name
            yield from _walk_citations(field_value, sub)
        return
    if isinstance(value, list | tuple):
        for i, item in enumerate(value):
            yield from _walk_citations(item, f"{path}[{i}]")
        return
    # Primitive (str / int / float / bool / None / date / datetime). No
    # citation can be reached from here; stop.


def validate_citation_grounding(output: BaseModel, source_text: str) -> None:
    """Assert every `Citation` in `output` aligns with `source_text`.

    Raises `CitationMismatch` on the first failure. The error message
    carries the field path, the requested span, the expected quote, and
    the actual slice — enough for a human reviewer to triage from the
    `QuarantineRecord` alone.

    "First failure wins" rather than "collect all failures" because:
    - One mismatch already means the output is quarantine-bound; further
      validation can't change that verdict.
    - Bulk-collecting risks masking the *cause* (often the model misaligned
      one span and the rest cascaded).
    """
    for path, citation in _walk_citations(output):
        start, end = citation.source_span
        actual = source_text[start:end]
        if actual != citation.source_quote:
            raise CitationMismatch(
                f"citation grounding failed at {path}: "
                f"span=({start},{end}) "
                f"expected={citation.source_quote!r} "
                f"actual={actual!r}"
            )


class QuarantineRecord(BaseModel):
    """Audit trail entry for an output that failed citation grounding.

    Written to `data/quarantine/<worker>/<doc_id>.json` on mismatch.
    Frozen because it's an audit record — mutation post-write would
    corrupt the trail.

    `output` carries the *failed* dump verbatim, not a "cleaned" or
    "partial" version, so a human reviewer can see exactly what the LLM
    returned. `error` is the `CitationMismatch.__str__` which already
    encodes the field path + expected/actual.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    worker: str
    prompt_version: str
    output: dict[str, Any]
    error: str
    quarantined_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _atomic_write_text(path: Path, content: str) -> None:
    """tmp → fsync → rename, so a torn write can't leave a truncated audit
    record. Quarantine entries are rare (only on validator failure) so
    the durability cost is negligible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(content)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def validate_or_quarantine(
    output: BaseModel,
    source_text: str,
    *,
    doc_id: str,
    worker: str,
    prompt_version: str,
    quarantine_root: Path | None = None,
) -> BaseModel | None:
    """Production routing helper: workers call this after their LLM step.

    On success: returns `output` unchanged.
    On `CitationMismatch`: writes a `QuarantineRecord` to
    `<quarantine_root>/<worker>/<doc_id>.json` and returns `None`. The
    caller MUST treat `None` as "do not persist any part of this output."

    Note the signature does NOT carry a `permissive` / `soft_mode` /
    `skip_validation` flag. INV-2 has no escape hatch (see
    `tests/unit/test_extract_guardrails.py::test_no_disabling_flags_on_public_api`).

    `quarantine_root` defaults to `data/quarantine/` (gitignored, see
    `.claude/settings.json` deny rules); tests pass `tmp_path` to keep
    the suite hermetic.
    """
    try:
        validate_citation_grounding(output, source_text)
    except CitationMismatch as exc:
        root = quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT
        record = QuarantineRecord(
            doc_id=doc_id,
            worker=worker,
            prompt_version=prompt_version,
            output=output.model_dump(mode="json"),
            error=str(exc),
        )
        target = root / worker / f"{doc_id}.json"
        _atomic_write_text(target, record.model_dump_json(indent=2))
        return None
    return output


__all__ = [
    "DEFAULT_QUARANTINE_ROOT",
    "CitationMismatch",
    "QuarantineRecord",
    "validate_citation_grounding",
    "validate_or_quarantine",
]
