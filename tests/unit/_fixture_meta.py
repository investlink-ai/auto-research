"""Pydantic model for chunking fixture `meta.json` files.

Each `tests/fixtures/chunking/sample_10k_*.htm` is paired with a
`sample_10k_*.meta.json` describing the filing (ticker, doc_id, fiscal
period, expected sections, etc.) plus CI tiering. The schema has grown
organically; this model freezes it and rejects unknown fields so a
typo in a meta.json fails at fixture-load time with a clear stem-tagged
error rather than a downstream `KeyError` deep in a parametrized test.

The model is intentionally test-internal (`tests/unit/_fixture_meta.py`)
rather than a public `auto_research` schema — it describes the test
harness's fixture format, not a runtime contract.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class FixtureMetadata(BaseModel):
    """Frozen schema for `tests/fixtures/chunking/*.meta.json`.

    `extra="forbid"` catches typo'd keys at validation time. Adding a
    new field anywhere in the fixture set requires updating this model
    first, which gives the type checker something to anchor against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    filing_date: date
    fiscal_period: str
    doc_type: Literal["10-K"]
    doc_id: str
    # `tuple[str, ...]` rather than `list[str]` — `frozen=True` blocks
    # `__setattr__` only; list contents stay mutable, which combined
    # with the module-scope `_FIXTURE_CACHE` in `test_chunking.py` is
    # a footgun (an `.append` on a cached fixture's expected_sections
    # would silently corrupt every subsequent parametrized case).
    expected_sections: tuple[str, ...]
    source_url: str
    filename_convention: str
    note: str
    # `tier` is required but defaulted in older fixtures; pre-#13 entries
    # carry `tier="broad"`. Keeping it required (no Pydantic default)
    # forces explicit intent when adding a fixture; the build script
    # always writes it.
    tier: Literal["core", "broad"]
    # Only the 5 `core` fixtures carry a rationale today; broad-tier
    # entries omit it. Optional to preserve that.
    tier_rationale: str | None = None


def load_fixture_meta(path: Path) -> FixtureMetadata:
    """Parse a meta.json into FixtureMetadata, re-raising with the
    fixture filename on failure.

    Pydantic's own ValidationError lists field errors but not which
    fixture file produced them — we wrap it so test failures cite the
    offending fixture filename in the first line.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        return FixtureMetadata.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Malformed fixture meta.json at {path.name}: {exc}"
        ) from exc


__all__ = ["FixtureMetadata", "load_fixture_meta"]
