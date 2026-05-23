"""Per-ticker mapping to its `AudioSource` implementation.

Data, not code: each ticker resolves to a short source-name string,
and the orchestrator wires the matching `sources/*.py` module. This
keeps platform additions (q4inc, youtube, …) from touching the
orchestrator — a new platform is a new module + new rows here.

V1 ships empty. Tickers absent from the registry are written to the
manifest as `status="no_coverage"` with no fetch attempted. PR #6f's
Playwright-driven coverage survey populates this map for real.

The data lives here (a constant dict) rather than as YAML/JSON
because (a) we want strict types — every entry validated against
known source names — and (b) the registry is small enough (~90
tickers max) that a code-loaded dict is cheaper than a runtime YAML
parse. Promote to a config file if the universe ever 10x's.
"""

from __future__ import annotations

# Source-name set. Each name MUST correspond to a module under
# `transcripts.sources.<name>` exposing an `AudioSource`-compatible
# class. Sources added in follow-up PRs extend this set:
#   - "direct_mp3" — generic raw-MP3 URL on the IR page (PR #6 / this)
#   - "q4inc"     — Q4 Inc webcast via Playwright + ffmpeg (PR #6d)
#   - "youtube"   — YouTube replay via yt-dlp (PR #6e)
KNOWN_SOURCES: frozenset[str] = frozenset({"direct_mp3"})

# ticker → source-name. Empty in v1; populated by PR #6f after a
# Playwright-driven coverage survey of the universe.
REGISTRY: dict[str, str] = {}


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
