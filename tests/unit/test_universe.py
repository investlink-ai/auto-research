"""Unit tests for the universe loader (Issue #4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_research.universe import TickerEntry, load_universe


def _entry(
    ticker: str = "NVDA",
    sub_universe: str = "ai_infra",
    sector: str = "semiconductors",
    market_cap_tier: str = "mega",
    tradeable: bool = True,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "sub_universe": sub_universe,
        "sector": sector,
        "market_cap_tier": market_cap_tier,
        "tradeable": tradeable,
    }


# ---------- Happy path against the checked-in universe_v1.json ----------


def test_load_universe_returns_non_empty_tuple() -> None:
    entries = load_universe()
    assert isinstance(entries, tuple)
    assert len(entries) >= 80  # spec §5 says ~90 names; ~80 is the floor


def test_load_universe_entries_are_frozen_models() -> None:
    entries = load_universe()
    assert isinstance(entries[0], TickerEntry)
    with pytest.raises(ValidationError):
        entries[0].ticker = "MUTATED"


def test_universe_v1_covers_spec_section_5_anchors() -> None:
    """Spot-check that headline names from spec §5 are checked in."""
    by_ticker = {e.ticker: e for e in load_universe()}
    # Narrative sources (Sub-universe A)
    for ticker in ("AAPL", "MSFT", "GOOGL", "NVDA", "DLR", "EQIX"):
        assert ticker in by_ticker, ticker
        assert by_ticker[ticker].sub_universe == "ai_infra"
        assert by_ticker[ticker].tradeable is False
    # Tradeable compute & networking
    for ticker in ("CRDO", "ASML", "TSM", "SMCI"):
        assert by_ticker[ticker].sub_universe == "ai_infra"
        assert by_ticker[ticker].tradeable is True
    # Tradeable power & infra
    for ticker in ("OKLO", "CEG", "VRT"):
        assert by_ticker[ticker].sub_universe == "ai_infra"
        assert by_ticker[ticker].tradeable is True
    # Frontier tech — quantum + space
    for ticker in ("IONQ", "RGTI"):
        assert by_ticker[ticker].sub_universe == "frontier_tech"
        assert by_ticker[ticker].tradeable is True
    for ticker in ("RKLB", "ASTS"):
        assert by_ticker[ticker].sub_universe == "frontier_tech"
        assert by_ticker[ticker].tradeable is True


def test_load_universe_tradeable_only_filters_narrative_sources() -> None:
    all_entries = load_universe()
    tradeable = load_universe(tradeable_only=True)
    assert 0 < len(tradeable) < len(all_entries)
    assert all(e.tradeable for e in tradeable)
    tradeable_tickers = {e.ticker for e in tradeable}
    # Narrative-source names from spec §5 are excluded
    for ticker in ("AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "DLR", "EQIX"):
        assert ticker not in tradeable_tickers, ticker


# ---------- Negative cases (fixture files in tmp_path) ----------


def test_load_universe_empty_raises(tmp_path: Path) -> None:
    bad = tmp_path / "empty.json"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        load_universe(path=bad)


def test_load_universe_duplicate_ticker_raises(tmp_path: Path) -> None:
    bad = tmp_path / "dup.json"
    bad.write_text(json.dumps([_entry("NVDA"), _entry("NVDA")]))
    with pytest.raises(ValueError, match="duplicate"):
        load_universe(path=bad)


def test_load_universe_unknown_sub_universe_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad-su.json"
    bad.write_text(json.dumps([_entry(sub_universe="crypto")]))
    with pytest.raises(ValidationError):
        load_universe(path=bad)


def test_load_universe_unknown_market_cap_tier_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad-mc.json"
    bad.write_text(json.dumps([_entry(market_cap_tier="gigantic")]))
    with pytest.raises(ValidationError):
        load_universe(path=bad)


def test_default_path_raises_when_project_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-editable installs (wheel) won't find pyproject.toml — fail loudly.

    Codex P2 review on #35 flagged the previous silent `Path.cwd()` fallback.
    The replacement raises FileNotFoundError pointing at the `path=` override.
    """
    import auto_research.universe as universe_mod

    # Pretend the module lives somewhere with no pyproject.toml above it.
    fake_module = tmp_path / "site-packages" / "auto_research" / "universe" / "__init__.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("")
    monkeypatch.setattr(universe_mod, "__file__", str(fake_module))

    with pytest.raises(FileNotFoundError, match="project root"):
        load_universe()
