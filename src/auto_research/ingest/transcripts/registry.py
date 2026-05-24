"""Per-ticker mapping to its `AudioSource` implementation.

Loaded from `data/transcripts/sources.toml` at import time — that
file is the single source of truth for which transcript source
handles each ticker, plus per-source overrides. See
`_config.py` for the schema and `data/transcripts/sources.toml`
for the data.

Tickers absent from the registry are written to the manifest as a
retryable `status="error"` row with no fetch attempted, so the
operator can spot gaps via the manifest audit.

The orchestrator imports `REGISTRY` and `KNOWN_SOURCES` from this
module. `validate()` runs at orchestrator import to fail loud on
typos. Adding a new source = adding a name to `KNOWN_SOURCES` and
wiring a factory in `transcripts/__init__.py:_SOURCE_FACTORIES`.
"""

from __future__ import annotations

from auto_research.ingest.transcripts._config import load_sources_config

# Source-name set. Each name MUST correspond to a module under
# `transcripts.sources.<name>` exposing an `AudioSource`-compatible
# class. Adding a new source is a new module + an entry here.
KNOWN_SOURCES: frozenset[str] = frozenset({"direct_mp3", "youtube"})

# Loaded once at module import. Mutable for testing (`monkeypatch.
# setitem(registry.REGISTRY, ...)` is the established pattern), but
# the file is the authoritative source — restart re-reads the file.
_CONFIG = load_sources_config()
REGISTRY: dict[str, str] = {
    ticker: cfg.source for ticker, cfg in _CONFIG.tickers.items()
}


def lookup(ticker: str) -> str | None:
    """Return the source-name for `ticker`, or None if not registered."""
    return REGISTRY.get(ticker.upper())


def validate() -> None:
    """Sanity-check the registry: every source-name is implemented.

    Called from `__init__.py` at module import time so a typo in
    `sources.toml` fails loudly at startup rather than at fetch time.
    """
    unknown = {src for src in REGISTRY.values() if src not in KNOWN_SOURCES}
    if unknown:
        raise ValueError(
            f"REGISTRY references unknown source names: {sorted(unknown)}. "
            f"Known: {sorted(KNOWN_SOURCES)}."
        )


__all__ = ["KNOWN_SOURCES", "REGISTRY", "lookup", "validate"]
