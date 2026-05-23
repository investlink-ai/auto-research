"""Follow-up: check ALL exhibits (not just the primary doc) of each
ticker's most-recent earnings 8-K. The primary doc was uniformly a
press release in the first survey; transcripts COULD theoretically be
attached as a separate EX-99.2 / 99.3 exhibit. This script tests that.

For each ticker's latest 8-K with Item 2.02 or 7.01:
  1. List all documents in the filing's index.json.
  2. Download every text/HTML exhibit (skip xml, exhibit indexes, graphics).
  3. Score each for transcript markers (Operator: count, Q&A phrase).
  4. Report the BEST candidate per ticker.
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

from auto_research.ingest.edgar import EdgarClient

_OPERATOR_RE = re.compile(r"\bOperator\s*:", re.IGNORECASE)
_QA_RE = re.compile(r"(question.{0,5}and.{0,5}answer|q\s*[&-]\s*a)", re.IGNORECASE)

# Heuristic thresholds. Real transcripts have many "Operator:" turns
# (each Q&A question is introduced by one) and are >50KB of text.
_TRANSCRIPT_MIN_SIZE = 30_000
_TRANSCRIPT_MIN_OPERATORS = 3

# File extensions we'll inspect. Skip indexes, graphics, XBRL.
_TEXT_EXTS = {".htm", ".html", ".txt"}


@dataclass
class ExhibitScore:
    name: str
    size: int
    operators: int
    has_qa: bool


@dataclass
class Result:
    ticker: str
    cik: str | None
    accession: str | None
    exhibits_checked: int
    best_operators: int
    best_qa: bool
    best_size: int
    best_name: str
    verdict: str  # "transcript_present" / "no_transcript" / "no_filing" / "error"


def fetch_ticker_to_cik(client: EdgarClient) -> dict[str, str]:
    resp = client._get("https://www.sec.gov/files/company_tickers.json")
    resp.raise_for_status()
    raw = resp.json()
    return {e["ticker"].upper(): f"{int(e['cik_str']):010d}" for e in raw.values()}


def find_latest_earnings_8k(recent: dict[str, list[Any]]) -> int | None:
    forms = recent.get("form", [])
    items_list = recent.get("items", [])
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = items_list[i] if i < len(items_list) else ""
        if not items:
            continue
        item_set = {s.strip() for s in items.split(",")}
        if "2.02" in item_set or "7.01" in item_set:
            return i
    return None


def list_exhibits(
    client: EdgarClient, cik_padded: str, accession: str
) -> list[dict[str, Any]] | None:
    """Fetch the filing's index.json to enumerate all documents."""
    accession_nodash = accession.replace("-", "")
    cik_unpadded = str(int(cik_padded))
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_nodash}/index.json"
    )
    try:
        resp = client._get(url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    resp.raise_for_status()
    body = resp.json()
    return body.get("directory", {}).get("item", [])


def score_exhibit(client: EdgarClient, url: str) -> ExhibitScore | None:
    try:
        resp = client._get(url)
    except httpx.HTTPStatusError:
        return None
    resp.raise_for_status()
    content = resp.content
    text = re.sub(rb"<[^>]+>", b" ", content).decode("utf-8", errors="ignore")
    ops = len(_OPERATOR_RE.findall(text))
    has_qa = bool(_QA_RE.search(text))
    return ExhibitScore(name=url.rsplit("/", 1)[-1], size=len(content), operators=ops, has_qa=has_qa)


def survey_ticker(client: EdgarClient, ticker: str, cik_padded: str | None) -> Result:
    if not cik_padded:
        return Result(ticker, None, None, 0, 0, False, 0, "", "no_filing")
    try:
        resp = client._get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Only 404 means "no submissions" — other HTTP errors (5xx after
        # retry exhaustion, 401/403) should propagate so the survey logs
        # a real `error` row rather than silently mislabel transient
        # SEC outages as "no filings", which would shrink the ADR's
        # denominator without warning.
        if exc.response.status_code == 404:
            return Result(ticker, cik_padded, None, 0, 0, False, 0, "", "no_filing")
        raise
    body = resp.json()
    recent = body.get("filings", {}).get("recent", {})
    idx = find_latest_earnings_8k(recent)
    if idx is None:
        return Result(ticker, cik_padded, None, 0, 0, False, 0, "", "no_filing")
    accession = recent["accessionNumber"][idx]
    items = list_exhibits(client, cik_padded, accession) or []
    accession_nodash = accession.replace("-", "")
    cik_unpadded = str(int(cik_padded))
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_nodash}"

    best_ops = 0
    best_qa = False
    best_size = 0
    best_name = ""
    checked = 0
    for item in items:
        name = item.get("name", "")
        if not name:
            continue
        ext = Path(name).suffix.lower()
        if ext not in _TEXT_EXTS:
            continue
        # Skip the cover sheet and the standalone index.
        if name.endswith("-index.htm") or name.endswith("-index.html"):
            continue
        url = f"{base}/{name}"
        score = score_exhibit(client, url)
        if score is None:
            continue
        checked += 1
        # Pick the exhibit with the highest operator count; ties broken by size.
        if (score.operators, score.size) > (best_ops, best_size):
            best_ops = score.operators
            best_qa = score.has_qa
            best_size = score.size
            best_name = score.name

    verdict = (
        "transcript_present"
        if best_ops >= _TRANSCRIPT_MIN_OPERATORS and best_size >= _TRANSCRIPT_MIN_SIZE
        else "no_transcript"
    )
    return Result(
        ticker, cik_padded, accession, checked, best_ops, best_qa, best_size, best_name, verdict
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    universe_path = repo_root / "data" / "universe" / "universe_v1.json"
    out_csv = repo_root / "scripts" / "survey_8k_exhibits.csv"
    out_md = repo_root / "scripts" / "survey_8k_exhibits.md"

    tickers = [e["ticker"] for e in json.loads(universe_path.read_text())]
    print(f"Surveying exhibits for {len(tickers)} tickers …", file=sys.stderr)

    with EdgarClient() as client:
        ticker_to_cik = fetch_ticker_to_cik(client)
        results: list[Result] = []
        for i, ticker in enumerate(tickers, 1):
            cik = ticker_to_cik.get(ticker.upper())
            try:
                r = survey_ticker(client, ticker, cik)
            except Exception as exc:
                r = Result(ticker, cik, None, 0, 0, False, 0, "", f"error: {exc}")
            results.append(r)
            print(
                f"  [{i:>2}/{len(tickers)}] {ticker:6} → {r.verdict:18} "
                f"(checked={r.exhibits_checked}, best_ops={r.best_operators}, "
                f"best_size={r.best_size})",
                file=sys.stderr,
            )

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ticker",
                "cik",
                "accession",
                "verdict",
                "exhibits_checked",
                "best_exhibit",
                "best_size",
                "best_operators",
                "best_has_qa",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.ticker,
                    r.cik or "",
                    r.accession or "",
                    r.verdict,
                    r.exhibits_checked,
                    r.best_name,
                    r.best_size,
                    r.best_operators,
                    r.best_qa,
                ]
            )

    from collections import Counter

    counter: Counter[str] = Counter(r.verdict for r in results)
    lines = [
        "# 8-K exhibits transcript survey",
        "",
        f"Surveyed all text/HTML exhibits of the most-recent earnings 8-K for {len(tickers)} tickers.",
        "",
        "## Verdict counts",
        "",
        "| Verdict | Count |",
        "|---|---:|",
    ]
    for v, n in counter.most_common():
        lines.append(f"| {v} | {n} |")
    lines.extend(
        [
            "",
            "## Per-ticker best-candidate exhibit",
            "",
            "| Ticker | Verdict | Exhibits checked | Best exhibit | Size (KB) | `Operator:` | Q&A |",
            "|---|---|---:|---|---:|---:|:---:|",
        ]
    )
    for r in sorted(results, key=lambda x: (x.verdict, -x.best_operators, x.ticker)):
        size_kb = f"{r.best_size / 1024:.0f}" if r.best_size else "-"
        qa = "✓" if r.best_qa else ""
        lines.append(
            f"| {r.ticker} | {r.verdict} | {r.exhibits_checked} | "
            f"{r.best_name or '-'} | {size_kb} | {r.best_operators} | {qa} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_csv} and {out_md}", file=sys.stderr)
    # Aggregate guardrail: a survey with errors is degraded — surface
    # loudly so the ADR's headline counts aren't read as definitive.
    n_error = sum(1 for v in counter if v.startswith("error"))
    if n_error or any(r.verdict.startswith("error") for r in results):
        bad = sum(1 for r in results if r.verdict.startswith("error"))
        print(
            f"\nWARNING: {bad}/{len(tickers)} tickers ended in an error state. "
            "Survey is DEGRADED; re-run before citing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
