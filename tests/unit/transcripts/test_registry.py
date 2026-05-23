"""Unit tests for `auto_research.ingest.transcripts.registry`."""

from __future__ import annotations

import pytest

from auto_research.ingest.transcripts import registry


def test_registry_starts_empty() -> None:
    """v1 ships with no tickers wired. PR #6f populates after a
    Playwright coverage survey. If this changes, the test is the
    place to update with the rationale."""
    assert registry.REGISTRY == {}


def test_known_sources_contains_direct_mp3() -> None:
    assert "direct_mp3" in registry.KNOWN_SOURCES


def test_lookup_returns_none_for_unregistered() -> None:
    assert registry.lookup("NVDA") is None


def test_lookup_returns_source_when_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(registry.REGISTRY, "ACME", "direct_mp3")
    assert registry.lookup("ACME") == "direct_mp3"
    # Case-insensitive ticker lookup.
    assert registry.lookup("acme") == "direct_mp3"


def test_validate_passes_on_known_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry.REGISTRY, "ACME", "direct_mp3")
    # Should not raise.
    registry.validate()


def test_validate_rejects_unknown_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry.REGISTRY, "ACME", "nonexistent_source")
    with pytest.raises(ValueError, match="unknown source names"):
        registry.validate()
