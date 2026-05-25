"""Universe loader.

Reads `config/universe/universe_v1.json` (~90 names from spec §5) and returns
a tuple of frozen `TickerEntry` models. The universe is versioned by
filename; v1 covers AI infrastructure (~70) + frontier tech (~20).

Two orthogonal flags carve the universe into operational roles:

  - `tradeable`: do we take direct positions in this name? Narrative-
    source names (AAPL, MSFT, GOOGL, NVDA, …) are `tradeable=False` —
    we *read* their filings to populate forward-demand signals on the
    tradeable book, but never trade them.

  - `feature_source`: does the chunker/extractor pipeline process this
    name's own filings? `False` for foreign filers (ASML, TSM, ARM,
    NVMI, SIMO, GFS, CCJ) whose annual reports are 20-F / 40-F, not
    10-K — the v1 chunker only handles 10-K. They remain in the
    universe so entity resolution can map mentions in *other* names'
    filings to a ticker (e.g., NVDA's 10-K naming "TSMC" → TSM). See
    `docs/decisions/2026-05-25-foreign-filers-deferred.md`.

Use `load_universe(tradeable_only=True)` for the tradeable book and
`load_universe(feature_source_only=True)` for names the extraction
pipeline should ingest. Filters compose (both can be set).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SubUniverse = Literal["ai_infra", "frontier_tech"]
MarketCapTier = Literal["mega", "large", "mid", "small", "micro"]
FilingForm = Literal["10-K", "20-F", "40-F"]


class TickerEntry(BaseModel):
    """One universe entry. Frozen — mutation raises `ValidationError`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=8, pattern=r"^[A-Z][A-Z0-9.\-]*$")
    sub_universe: SubUniverse
    sector: str = Field(min_length=1)
    market_cap_tier: MarketCapTier
    tradeable: bool
    # Below default to "U.S. 10-K filer that we ingest" so all
    # existing entries validate without per-row changes. Only the
    # 7 foreign filers carry non-default values explicitly.
    filing_form: FilingForm = "10-K"
    feature_source: bool = True


def _default_path() -> Path:
    """Anchor `config/universe/universe_v1.json` on the project root.

    Walks up from this module looking for `pyproject.toml`. Editable
    installs (`uv sync`) keep the package inside the repo so this resolves
    deterministically. Non-editable installs (wheel in site-packages) won't
    find a parent `pyproject.toml` — and the wheel build doesn't bundle
    `config/`, so there's no implicit path to fall back to. Raise loudly
    instead of silently returning a CWD path that may or may not exist;
    callers can pass `load_universe(path=...)` to override.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "config" / "universe" / "universe_v1.json"
    raise FileNotFoundError(
        f"Could not locate the auto-research project root above {here} "
        "(no pyproject.toml in any parent directory). If this is a "
        "non-editable install, pass `load_universe(path=...)` with an "
        "explicit path to universe_v1.json."
    )


def load_universe(
    path: Path | None = None,
    *,
    tradeable_only: bool = False,
    feature_source_only: bool = False,
) -> tuple[TickerEntry, ...]:
    """Load the universe from JSON and return a tuple of frozen entries.

    Raises `ValueError` for empty files or duplicate tickers, and Pydantic
    `ValidationError` for unknown sub-universe / market-cap tier values
    or malformed rows.

    Filters compose: `load_universe(tradeable_only=True,
    feature_source_only=True)` returns names we both trade AND extract
    features from (the standard "v1 pilot" set).
    """
    target = path if path is not None else _default_path()
    raw = json.loads(target.read_text())

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Universe at {target} is empty.")

    entries = tuple(TickerEntry.model_validate(row) for row in raw)

    seen: set[str] = set()
    for entry in entries:
        if entry.ticker in seen:
            raise ValueError(f"Universe at {target} contains duplicate ticker: {entry.ticker}")
        seen.add(entry.ticker)

    if tradeable_only:
        entries = tuple(e for e in entries if e.tradeable)
    if feature_source_only:
        entries = tuple(e for e in entries if e.feature_source)
    return entries


__all__ = ["FilingForm", "MarketCapTier", "SubUniverse", "TickerEntry", "load_universe"]
