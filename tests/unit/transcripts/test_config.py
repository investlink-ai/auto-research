"""Unit tests for the transcripts TOML config loader.

The loader reads `config/transcripts/sources.toml` and validates each
row via Pydantic. Tests cover (a) the happy path against the
checked-in file, (b) per-source invariants enforced at load time,
(c) override path injection for test isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_research.ingest.transcripts._config import (
    TickerSourceConfig,
    load_sources_config,
)

# ---------- happy path against checked-in file ----------


def test_default_load_returns_at_least_one_ticker() -> None:
    """The checked-in config has every universe ticker mapped."""
    cfg = load_sources_config()
    assert len(cfg.tickers) > 0
    assert all(isinstance(v, TickerSourceConfig) for v in cfg.tickers.values())


def test_default_load_all_sources_are_known() -> None:
    """Every source name in the TOML must be in KNOWN_SOURCES — caught
    by the orchestrator's import-time `validate()` too, but worth
    pinning at the config level."""
    from auto_research.ingest.transcripts import registry

    cfg = load_sources_config()
    seen_sources = {entry.source for entry in cfg.tickers.values()}
    assert seen_sources <= registry.KNOWN_SOURCES, (
        f"Config references unknown sources: "
        f"{sorted(seen_sources - registry.KNOWN_SOURCES)}"
    )


# ---------- Pydantic validation ----------


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        TickerSourceConfig.model_validate(
            {"source": "youtube", "made_up_field": "x"}
        )


def test_missing_source_rejected() -> None:
    with pytest.raises(ValidationError, match="source"):
        TickerSourceConfig.model_validate({"query": "Foo"})


def test_direct_mp3_without_url_rejected() -> None:
    """direct_mp3 source REQUIRES a url template — otherwise fetch
    has no URL to dispatch on. Caught at config-load."""
    with pytest.raises(ValidationError, match="direct_mp3"):
        TickerSourceConfig.model_validate({"source": "direct_mp3"})


def test_direct_mp3_with_url_accepted() -> None:
    cfg = TickerSourceConfig.model_validate(
        {"source": "direct_mp3", "url": "https://x/{year}Q{quarter}.mp3"}
    )
    assert cfg.url is not None


def test_youtube_without_query_accepted() -> None:
    """Most youtube rows omit the query (bare-ticker search works);
    no override is the common case."""
    cfg = TickerSourceConfig.model_validate({"source": "youtube"})
    assert cfg.query is None


def test_youtube_with_query_accepted() -> None:
    cfg = TickerSourceConfig.model_validate(
        {"source": "youtube", "query": "NVIDIA"}
    )
    assert cfg.query == "NVIDIA"


def test_frozen() -> None:
    """`TickerSourceConfig` is frozen — mutation raises ValidationError."""
    cfg = TickerSourceConfig.model_validate({"source": "youtube"})
    with pytest.raises(ValidationError):
        cfg.source = "direct_mp3"


# ---------- load() with injected path ----------


def test_load_with_explicit_path(tmp_path: Path) -> None:
    """Callers can inject a path — used by tests + future tooling."""
    f = tmp_path / "sources.toml"
    f.write_text(
        '[tickers]\n'
        'AAPL = { source = "youtube" }\n'
        'NVDA = { source = "youtube", query = "NVIDIA" }\n'
    )
    cfg = load_sources_config(path=f)
    assert set(cfg.tickers.keys()) == {"AAPL", "NVDA"}
    assert cfg.tickers["NVDA"].query == "NVIDIA"
    assert cfg.tickers["AAPL"].query is None


def test_load_raises_on_empty_tickers(tmp_path: Path) -> None:
    """An empty `[tickers]` table fails loud — the orchestrator would
    otherwise silently treat every fetch as 'unregistered'."""
    f = tmp_path / "empty.toml"
    f.write_text("[tickers]\n")
    with pytest.raises(ValueError, match="empty"):
        load_sources_config(path=f)


def test_load_raises_on_missing_tickers_table(tmp_path: Path) -> None:
    f = tmp_path / "missing.toml"
    f.write_text("# no tickers table at all\n")
    with pytest.raises(ValueError, match="empty"):
        load_sources_config(path=f)


def test_load_propagates_validation_errors(tmp_path: Path) -> None:
    """A row with unknown fields or missing required fields surfaces as
    Pydantic ValidationError, not a silent swallow."""
    f = tmp_path / "bad.toml"
    f.write_text(
        '[tickers]\n'
        'X = { source = "direct_mp3" }\n'  # missing required `url`
    )
    with pytest.raises(ValidationError, match="direct_mp3"):
        load_sources_config(path=f)


# ---------- shape against TranscriptsSourcesConfig ----------


def test_root_model_rejects_extra_top_level_keys(tmp_path: Path) -> None:
    f = tmp_path / "extra.toml"
    f.write_text(
        '[tickers]\n'
        'AAPL = { source = "youtube" }\n'
        '\n'
        '[unexpected]\n'
        'foo = "bar"\n'
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_sources_config(path=f)
