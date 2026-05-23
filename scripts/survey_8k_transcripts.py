"""Survey SEC 8-K filings for transcript-shaped content across the universe.

Question: can we get earnings call transcripts from SEC 8-K exhibits
(Item 2.02 Results of Operations, Item 7.01 Regulation FD) instead of
paying for FMP Ultimate ($149/mo)?

Method:
1. Resolve each universe ticker to a CIK via SEC's company_tickers.json.
2. Fetch the company's submissions JSON.
3. Find the most recent 8-K whose `items` field references 2.02 or 7.01.
4. Download the primary document.
5. Apply heuristics to classify:
   - `transcript`: ≥3 "Operator:" markers AND document >= 30KB
   - `press_release`: short document, no transcript markers
   - `mixed`: has some markers but small, or missing markers but large
   - `no_earnings_8k`: company has 8-Ks but none with Item 2.02 / 7.01
   - `no_filings`: no submissions data (private / newly-listed / wrong CIK)

Output: a markdown table to stdout, plus a CSV next to it for analysis.

Usage:
    SEC_USER_AGENT="Your Name your@email.com" \\
        uv run python scripts/survey_8k_transcripts.py

Cost: ~165 HTTP requests (one ticker-to-CIK lookup + 81 submissions + ~80
earnings-8K primary docs). At the 8 r/s rate limit, ~25 seconds wall-clock.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Use our EDGAR client for the rate-limited, retrying HTTP wrapper.
from auto_research.ingest.edgar import EdgarClient

_OPERATOR_RE = re.compile(r"\bOperator\s*:", re.IGNORECASE)
_QA_RE = re.compile(r"(question.{0,5}and.{0,5}answer|q\s*[&-]\s*a)", re.IGNORECASE)
_TRANSCRIPT_MIN_SIZE = 30_000  # bytes
_TRANSCRIPT_MIN_OPERATORS = 3


@dataclass
class TickerResult:
    ticker: str
    cik: str | None
    classification: str
    earnings_8k_count: int
    latest_8k_url: str | None
    latest_8k_size: int | None
    operator_count: int
    has_qa_marker: bool
    notes: str


def load_universe(path: Path) -> list[str]:
    entries = json.loads(path.read_text())
    return [e["ticker"] for e in entries]


def fetch_ticker_to_cik(client: EdgarClient) -> dict[str, str]:
    """SEC publishes a global ticker→CIK map. Returns {ticker: zero-padded CIK}."""
    # company_tickers.json lives on www.sec.gov, not data.sec.gov.
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = client._get(url)  # use our rate-limited wrapper
    resp.raise_for_status()
    raw = resp.json()
    # Shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    out: dict[str, str] = {}
    for entry in raw.values():
        out[entry["ticker"].upper()] = f"{int(entry['cik_str']):010d}"
    return out


def fetch_submissions(client: EdgarClient, cik_padded: str) -> dict[str, Any] | None:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        resp = client._get(url)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # `client._get` only raises HTTPStatusError on 429/5xx (which it
        # has already retried). 404s pass through and trigger here via
        # `raise_for_status`. Treat the 404 case as "no submissions"
        # cleanly; let other failures (auth, 5xx-after-retries) escape.
        if exc.response.status_code == 404:
            return None
        raise
    return resp.json()


def find_latest_earnings_8k(recent: dict[str, list[Any]]) -> tuple[int, int] | None:
    """Return (index, count) where index points to the most-recent earnings 8-K.

    count is the total number of earnings-flagged 8-Ks in the recent window.
    Returns None if no such filing.
    """
    forms = recent.get("form", [])
    items_list = recent.get("items", [])
    if not forms:
        return None
    latest_idx: int | None = None
    count = 0
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = items_list[i] if i < len(items_list) else ""
        if not items:
            continue
        # `items` is a comma-separated string like "2.02,9.01"
        item_set = {s.strip() for s in items.split(",")}
        if "2.02" in item_set or "7.01" in item_set:
            count += 1
            if latest_idx is None:
                latest_idx = i
    if latest_idx is None:
        return None
    return latest_idx, count


def classify(content: bytes) -> tuple[str, int, bool]:
    """Return (classification, operator_count, has_qa)."""
    # Strip HTML tags crudely for marker detection — SEC exhibits are usually HTML.
    text = re.sub(rb"<[^>]+>", b" ", content).decode("utf-8", errors="ignore")
    operator_count = len(_OPERATOR_RE.findall(text))
    has_qa = bool(_QA_RE.search(text))
    size = len(content)
    if operator_count >= _TRANSCRIPT_MIN_OPERATORS and size >= _TRANSCRIPT_MIN_SIZE:
        return "transcript", operator_count, has_qa
    if operator_count > 0 or has_qa:
        return "mixed", operator_count, has_qa
    return "press_release", operator_count, has_qa


def survey_ticker(
    client: EdgarClient,
    ticker: str,
    cik_padded: str | None,
) -> TickerResult:
    if not cik_padded:
        return TickerResult(
            ticker, None, "no_cik", 0, None, None, 0, False, "no CIK in SEC ticker map"
        )
    body = fetch_submissions(client, cik_padded)
    if body is None:
        return TickerResult(
            ticker, cik_padded, "no_filings", 0, None, None, 0, False, "404 on submissions"
        )
    recent = body.get("filings", {}).get("recent", {})
    hit = find_latest_earnings_8k(recent)
    if hit is None:
        return TickerResult(
            ticker, cik_padded, "no_earnings_8k", 0, None, None, 0, False, ""
        )
    idx, count = hit
    accession = recent["accessionNumber"][idx]
    primary = recent["primaryDocument"][idx]
    accession_nodash = accession.replace("-", "")
    cik_unpadded = str(int(cik_padded))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_nodash}/{primary}"
    try:
        resp = client._get(url)
    except httpx.HTTPStatusError as exc:
        return TickerResult(
            ticker,
            cik_padded,
            "fetch_error",
            count,
            url,
            None,
            0,
            False,
            f"HTTP {exc.response.status_code}",
        )
    resp.raise_for_status()
    content = resp.content
    classification, ops, has_qa = classify(content)
    return TickerResult(
        ticker=ticker,
        cik=cik_padded,
        classification=classification,
        earnings_8k_count=count,
        latest_8k_url=url,
        latest_8k_size=len(content),
        operator_count=ops,
        has_qa_marker=has_qa,
        notes="",
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    universe_path = repo_root / "data" / "universe" / "universe_v1.json"
    out_csv = repo_root / "scripts" / "survey_8k_transcripts.csv"
    out_md = repo_root / "scripts" / "survey_8k_transcripts.md"

    tickers = load_universe(universe_path)
    print(f"Surveying {len(tickers)} tickers …", file=sys.stderr)

    with EdgarClient() as client:
        ticker_to_cik = fetch_ticker_to_cik(client)
        results: list[TickerResult] = []
        for i, ticker in enumerate(tickers, 1):
            cik = ticker_to_cik.get(ticker.upper())
            try:
                r = survey_ticker(client, ticker, cik)
            except Exception as exc:
                r = TickerResult(
                    ticker, cik, "error", 0, None, None, 0, False, f"{type(exc).__name__}: {exc}"
                )
            results.append(r)
            print(
                f"  [{i:>2}/{len(tickers)}] {ticker:6} → {r.classification:14} "
                f"(ops={r.operator_count}, size={r.latest_8k_size or 0})",
                file=sys.stderr,
            )

    # CSV
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ticker",
                "cik",
                "classification",
                "earnings_8k_count",
                "latest_8k_size",
                "operator_count",
                "has_qa_marker",
                "latest_8k_url",
                "notes",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.ticker,
                    r.cik or "",
                    r.classification,
                    r.earnings_8k_count,
                    r.latest_8k_size or "",
                    r.operator_count,
                    r.has_qa_marker,
                    r.latest_8k_url or "",
                    r.notes,
                ]
            )

    # Markdown summary
    from collections import Counter

    counter: Counter[str] = Counter(r.classification for r in results)
    lines = [
        "# 8-K transcript survey",
        "",
        f"Surveyed {len(tickers)} tickers from `data/universe/universe_v1.json`.",
        "",
        "## Classification counts",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for cls, n in counter.most_common():
        lines.append(f"| {cls} | {n} |")
    lines.extend(
        [
            "",
            "## Per-ticker",
            "",
            "| Ticker | Class | Earnings 8-Ks | Latest 8-K size (KB) | `Operator:` count | Q&A marker |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for r in sorted(results, key=lambda x: (x.classification, x.ticker)):
        size_kb = f"{r.latest_8k_size / 1024:.0f}" if r.latest_8k_size else "-"
        qa = "✓" if r.has_qa_marker else ""
        lines.append(
            f"| {r.ticker} | {r.classification} | {r.earnings_8k_count} | "
            f"{size_kb} | {r.operator_count} | {qa} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {out_csv} and {out_md}", file=sys.stderr)
    # Aggregate guardrail: a survey with errors is degraded — surface
    # loudly so an ADR consumer doesn't read the counts as definitive.
    n_error = counter.get("error", 0) + counter.get("fetch_error", 0)
    if n_error:
        print(
            f"\nWARNING: {n_error}/{len(tickers)} tickers ended in 'error' / "
            f"'fetch_error' state. Survey is DEGRADED; re-run before citing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
