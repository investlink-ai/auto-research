"""Per-ticker mapping to its `AudioSource` implementation.

Data, not code: each ticker resolves to a short source-name string,
and the orchestrator wires the matching `sources/*.py` module. This
keeps platform additions from touching the orchestrator — a new
platform is a new module + new rows here.

Tickers absent from the registry are written to the manifest as a
retryable `status="error"` row with no fetch attempted, so the
operator can spot gaps via the manifest audit.

The data lives here (a constant dict) rather than as YAML/JSON
because (a) we want strict types — every entry validated against
known source names — and (b) the registry is small enough (~90
tickers max) that a code-loaded dict is cheaper than a runtime YAML
parse. Promote to a config file if the universe ever 10x's.

Coverage as of 2026-05-24: all 80 v1 universe tickers route to
`youtube`. The mapping was empirically validated against SEC's
canonical ticker→company-name index (`company_tickers.json`):
each ticker's matched URL's video title contains a meaningful word
from the SEC name. Per-ticker query overrides (where the bare
ticker symbol fails) live in `sources/youtube.py:TICKER_QUERIES`.

If a future probe shows a ticker is better served by `direct_mp3`
(issuer-self-hosted MP3 on the IR page) than YouTube aggregator
mirrors, edit the entry here; both sources satisfy the same
`AudioSource` Protocol so the orchestrator is unchanged.
"""

from __future__ import annotations

# Source-name set. Each name MUST correspond to a module under
# `transcripts.sources.<name>` exposing an `AudioSource`-compatible
# class. Adding a new source is a new module + an entry here.
KNOWN_SOURCES: frozenset[str] = frozenset({"direct_mp3", "youtube"})

# ticker → source-name. All 80 universe tickers route to `youtube`
# after the universe-wide validation against SEC ground truth (see
# module docstring). Tickers can migrate to `direct_mp3` (or future
# sources) by editing this map; the orchestrator is unchanged.
REGISTRY: dict[str, str] = {
    "AAOI": "youtube", "AAPL": "youtube", "ACMR": "youtube", "AEHR": "youtube", "ALAB": "youtube",
    "ALGM": "youtube", "AMAT": "youtube", "AMBA": "youtube", "AMD": "youtube", "AMZN": "youtube",
    "ANET": "youtube", "ARM": "youtube", "ASML": "youtube", "ASTS": "youtube", "AVGO": "youtube",
    "BE": "youtube", "BKSY": "youtube", "CCJ": "youtube", "CEG": "youtube", "CIEN": "youtube",
    "COHR": "youtube", "CRDO": "youtube", "CRM": "youtube", "DELL": "youtube", "DLR": "youtube",
    "EQIX": "youtube", "ETN": "youtube", "FORM": "youtube", "GEV": "youtube", "GFS": "youtube",
    "GOOGL": "youtube", "HON": "youtube", "IBM": "youtube", "IONQ": "youtube", "KLAC": "youtube",
    "LEU": "youtube", "LITE": "youtube", "LRCX": "youtube", "LSCC": "youtube", "LUNR": "youtube",
    "META": "youtube", "MIR": "youtube", "MPWR": "youtube", "MRVL": "youtube", "MSFT": "youtube",
    "MTSI": "youtube", "MU": "youtube", "NNE": "youtube", "NOW": "youtube", "NRG": "youtube",
    "NTNX": "youtube", "NVDA": "youtube", "NVMI": "youtube", "OKLO": "youtube", "ON": "youtube",
    "ORCL": "youtube", "PL": "youtube", "PLTR": "youtube", "PWR": "youtube", "QBTS": "youtube",
    "QRVO": "youtube", "QUBT": "youtube", "RDW": "youtube", "RGTI": "youtube", "RKLB": "youtube",
    "RMBS": "youtube", "SATS": "youtube", "SIMO": "youtube", "SMCI": "youtube", "SMR": "youtube",
    "SNDK": "youtube", "STRL": "youtube", "SWKS": "youtube", "TER": "youtube", "TLN": "youtube",
    "TSLA": "youtube", "TSM": "youtube", "VRT": "youtube", "VST": "youtube", "WDC": "youtube",
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
