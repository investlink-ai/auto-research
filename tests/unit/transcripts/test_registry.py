"""Unit tests for `auto_research.ingest.transcripts.registry`."""

from __future__ import annotations

import pytest

from auto_research.ingest.transcripts import registry


def test_registry_seeds_only_nvda_canary() -> None:
    """NVDA is the youtube canary; broader coverage is populated by
    the coverage-survey worker. If this changes, update this test
    with the rationale (and likely the parent issue's scope)."""
    assert registry.REGISTRY == {"NVDA": "youtube"}


def test_known_sources_contains_implemented_platforms() -> None:
    assert {"direct_mp3", "youtube"} <= registry.KNOWN_SOURCES


def test_lookup_returns_none_for_unregistered() -> None:
    assert registry.lookup("NEVER_REGISTERED") is None


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
