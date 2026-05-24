"""Per-ticker mapping to its `AudioSource` implementation.

Data, not code: each ticker resolves to a short source-name string,
and the orchestrator wires the matching `sources/*.py` module. This
keeps platform additions from touching the orchestrator — a new
platform is a new module + new rows here.

V1 ships empty. Tickers absent from the registry are written to the
manifest as a retryable `status="error"` row with no fetch attempted;
the coverage-survey worker populates this map for real.

The data lives here (a constant dict) rather than as YAML/JSON
because (a) we want strict types — every entry validated against
known source names — and (b) the registry is small enough (~90
tickers max) that a code-loaded dict is cheaper than a runtime YAML
parse. Promote to a config file if the universe ever 10x's.
"""

from __future__ import annotations

# Source-name set. Each name MUST correspond to a module under
# `transcripts.sources.<name>` exposing an `AudioSource`-compatible
# class. Adding a new source is a new module + an entry here.
KNOWN_SOURCES: frozenset[str] = frozenset({"direct_mp3", "youtube"})

# ticker → source-name. NVDA seeded as the canary for the youtube
# discovery path; the full universe gets populated by the
# coverage-survey worker (which probes each ticker's reachability
# and picks the best source per ticker).
REGISTRY: dict[str, str] = {
    "NVDA": "youtube",
}


def lookup(ticker: str) -> str | None:
    """Return the source-name for `ticker`, or None if not registered."""
    return REGISTRY.get(ticker.upper())


def validate() -> None:
    """Sanity-check the registry: every source-name is implemented.

    Called from `__init__.py` at module import time so a typo in
    `REGISTRY` fails loudly at startup rather than at fetch time.
    """
    unknown = {src for src in REGISTRY.values() if src not in KNOWN_SOURCES}
    if unknown:
        raise ValueError(
            f"REGISTRY references unknown source names: {sorted(unknown)}. "
            f"Known: {sorted(KNOWN_SOURCES)}."
        )


__all__ = ["KNOWN_SOURCES", "REGISTRY", "lookup", "validate"]
