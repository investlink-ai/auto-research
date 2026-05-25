"""Batch-build a chunking test fixture for every ticker in the universe.

Reads `config/universe/universe_v1.json`, resolves each ticker's CIK
via SEC's `company_tickers.json`, finds the most recent 10-K filing
via the submissions API, and runs the build_chunking_fixture pipeline
against it. Failures (no 10-K filed, foreign filer using 20-F, chunker
fails to detect Items in the trim) are logged and skipped — they do
not abort the batch.

Usage:
    uv run python scripts/build_universe_fixtures.py
    uv run python scripts/build_universe_fixtures.py --limit 10
    uv run python scripts/build_universe_fixtures.py --tickers NVDA,AMD

All new fixtures are tagged `tier="broad"` — they extend regression
coverage for `make test-broad` without inflating default CI time.
Promote individual fixtures to `tier="core"` by hand if they cover a
template variant worth catching in every PR.

Respects SEC's fair-access policy: minimum 0.12s between requests
(approximate 8 req/s, under the 10 req/s ceiling). User-Agent
identifies the project.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# `src/` is on the path because the project is editable-installed via uv.
from auto_research.universe import load_universe

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chunking"
BUILD_SCRIPT = Path(__file__).parent / "build_chunking_fixture.py"

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
USER_AGENT = "auto-research/0.1 research@example.com"
TICKER_MAP_URL = f"{SEC_BASE}/files/company_tickers.json"
MIN_REQ_INTERVAL_SEC = 0.12  # ≈ 8 req/s, under SEC's 10 req/s cap

_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.time() - _last_request_at
    if elapsed < MIN_REQ_INTERVAL_SEC:
        time.sleep(MIN_REQ_INTERVAL_SEC - elapsed)
    _last_request_at = time.time()


def _http_get(url: str) -> bytes:
    _throttle()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _load_ticker_to_cik_map() -> dict[str, int]:
    """Fetch SEC's master ticker→CIK map and invert by ticker."""
    raw = json.loads(_http_get(TICKER_MAP_URL))
    out: dict[str, int] = {}
    for entry in raw.values():
        ticker = entry["ticker"].upper()
        if ticker not in out:
            out[ticker] = int(entry["cik_str"])
    return out


def _load_universe(*, include_non_feature_source: bool = False) -> list[str]:
    """Return the universe tickers the chunker fixture pipeline should attempt.

    Defaults to `feature_source=True` names only — foreign filers
    (20-F / 40-F) are pre-filtered so the batch run does not log noisy
    `no_10k` failures for ASML/TSM/ARM/NVMI/SIMO/GFS/CCJ. Pass
    `include_non_feature_source=True` to verify the foreign filers'
    filing-form classifications still hold against SEC's submissions
    API.
    """
    entries = load_universe(feature_source_only=not include_non_feature_source)
    return [entry.ticker.upper() for entry in entries]


def _find_most_recent_10k(cik: int) -> tuple[str, str, str] | None:
    """Return (accession_dashed, filing_date, fiscal_period) or None."""
    cik_padded = f"{cik:010d}"
    url = f"{SEC_DATA_BASE}/submissions/CIK{cik_padded}.json"
    try:
        raw = json.loads(_http_get(url))
    except urllib.error.HTTPError:
        return None

    recent = raw.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    for form, accession, fdate, rdate in zip(
        forms, accessions, filing_dates, report_dates, strict=False
    ):
        if form != "10-K":
            continue
        # Derive fiscal period from the report date's year.
        fy_match = re.match(r"(\d{4})", rdate or fdate)
        fy = f"FY{fy_match.group(1)}" if fy_match else "FY"
        return accession, fdate, fy
    return None


def _build_one(
    ticker: str,
    cik: int,
    accession: str,
    filing_date: str,
    fiscal_period: str,
    out_stem: str,
) -> tuple[bool, str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(BUILD_SCRIPT),
        "--ticker",
        ticker,
        "--cik",
        str(cik),
        "--accession",
        accession,
        "--filing-date",
        filing_date,
        "--fiscal-period",
        fiscal_period,
        "--tier",
        "broad",
        "--out-stem",
        out_stem,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode == 0:
        return True, "ok"
    # Surface the build script's stderr tail for diagnosis.
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
    return False, " | ".join(tail) if tail else f"exit {result.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated subset")
    parser.add_argument("--limit", type=int, help="Build at most N tickers")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip tickers whose fixture already exists (default: True)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild fixtures even if they already exist",
    )
    parser.add_argument(
        "--include-non-feature-source",
        action="store_true",
        help=(
            "Include foreign filers (feature_source=False) — useful "
            "for verifying their filing-form classifications still hold. "
            "Default skips them so the batch run reports zero no_10k "
            "failures for ASML / TSM / ARM / NVMI / SIMO / GFS / CCJ."
        ),
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _load_universe(
            include_non_feature_source=args.include_non_feature_source
        )
    if args.limit:
        tickers = tickers[: args.limit]

    print(f"Universe: {len(tickers)} tickers", flush=True)
    print("Fetching SEC ticker→CIK map …", flush=True)
    try:
        ticker_to_cik = _load_ticker_to_cik_map()
    except Exception as exc:
        print(f"  failed: {exc}", file=sys.stderr)
        return 2
    print(f"  loaded {len(ticker_to_cik)} ticker entries", flush=True)

    built: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for ticker in tickers:
        out_stem = f"sample_10k_{ticker.lower()}"
        htm_path = FIXTURE_DIR / f"{out_stem}.htm"
        if htm_path.exists() and not args.rebuild:
            print(f"[{ticker}] already exists — skipping (use --rebuild to refresh)")
            skipped.append((ticker, "exists"))
            continue

        cik = ticker_to_cik.get(ticker)
        if cik is None:
            print(f"[{ticker}] ✗ no CIK in SEC ticker map")
            failed.append((ticker, "no_cik"))
            continue

        print(f"[{ticker}] CIK={cik}, finding most recent 10-K …")
        found = _find_most_recent_10k(cik)
        if found is None:
            print(f"[{ticker}] ✗ no 10-K filing found (foreign filer? recent IPO?)")
            failed.append((ticker, "no_10k"))
            continue
        accession, filing_date, fiscal_period = found
        print(
            f"[{ticker}]   most recent 10-K: {accession} filed {filing_date} ({fiscal_period})"
        )

        ok, reason = _build_one(
            ticker=ticker,
            cik=cik,
            accession=accession,
            filing_date=filing_date,
            fiscal_period=fiscal_period,
            out_stem=out_stem,
        )
        if ok:
            print(f"[{ticker}] ✓ built")
            built.append(ticker)
        else:
            print(f"[{ticker}] ✗ build failed: {reason}")
            failed.append((ticker, reason))

    print()
    print("=" * 60)
    print(f"Built:    {len(built):>3d}  {sorted(built)}")
    print(f"Skipped:  {len(skipped):>3d}  (already-built fixtures)")
    print(f"Failed:   {len(failed):>3d}")
    for ticker, reason in failed:
        print(f"    {ticker}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
