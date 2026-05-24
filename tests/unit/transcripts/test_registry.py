"""Unit tests for `auto_research.ingest.transcripts.registry`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_research.ingest.transcripts import registry

_UNIVERSE_PATH = Path("config/universe/universe_v1.json")


def _universe_tickers() -> set[str]:
    with _UNIVERSE_PATH.open() as f:
        return {entry["ticker"] for entry in json.load(f)}


def test_registry_covers_full_universe() -> None:
    """Every universe ticker has a source registered. If a new ticker
    is added to `config/universe/universe_v1.json`, this test fails
    until the registry maps it to a source — the right place to fix
    coverage gaps, since unmapped tickers fall through to retryable
    `status='error'` rows in the orchestrator's manifest."""
    universe = _universe_tickers()
    missing = universe - registry.REGISTRY.keys()
    assert not missing, (
        f"Universe tickers missing from REGISTRY: {sorted(missing)}. "
        "Either map them to a source or remove them from the universe."
    )


def test_registry_has_no_orphan_tickers() -> None:
    """Conversely, REGISTRY shouldn't carry tickers that aren't in the
    universe — orphans drift over time and confuse coverage audits."""
    universe = _universe_tickers()
    orphans = registry.REGISTRY.keys() - universe
    assert not orphans, (
        f"REGISTRY contains tickers not in universe: {sorted(orphans)}. "
        "Either add them to the universe or drop the registry rows."
    )


def test_registry_values_are_all_known_sources() -> None:
    """Defense-in-depth for the runtime `validate()` — catches typos
    at test time, not at first fetch."""
    unknown = set(registry.REGISTRY.values()) - registry.KNOWN_SOURCES
    assert not unknown, f"REGISTRY references unknown sources: {sorted(unknown)}"


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
