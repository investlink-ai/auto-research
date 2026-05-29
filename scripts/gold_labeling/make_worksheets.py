"""Generate hand-labeling worksheets for the 8-K gold set from real EDGAR
filings. Labeling-support script (real SEC network I/O; needs SEC_USER_AGENT).

Per fetched 8-K it writes TWO files under
``eval/gold_sets/_worksheets/eight_k/<doc_id>/``:

- ``source.txt``     — the cleaned material-item text. READ-ONLY reference;
  this exact text becomes the gold ``raw_doc``, so every label quote must be
  a verbatim substring of it. Do not edit it.
- ``labels.json``    — what YOU fill in. Pick ``event_classification`` from
  the enum, and copy exact quote spans from ``source.txt`` into
  ``milestone_mentions`` / ``dilution_language_flags`` (empty lists are fine
  and expected for many filings). ``_candidates`` are keyword hints to
  verify, NOT answers — delete what doesn't belong.

Run from repo root:

    uv run python scripts/gold_labeling/make_worksheets.py --tickers NVDA AMD MU --per 1

Then hand the filled worksheets to `ingest_worksheets.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import httpx

from auto_research._io import project_root
from auto_research.extract.enums import EventClassification
from auto_research.ingest.edgar import EdgarClient

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def resolve_ciks_from_env() -> dict[str, str]:
    """Map ticker -> zero-padded 10-digit CIK via SEC's public registry."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise SystemExit("SEC_USER_AGENT not set (export it or source .env)")
    resp = httpx.get(_TICKER_MAP_URL, headers={"User-Agent": ua}, timeout=30.0)
    resp.raise_for_status()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in resp.json().values()}


def clean_filing_text(path: Path) -> str:
    """unstructured partition_html -> newline-joined element text."""
    from unstructured.partition.html import partition_html

    elements = partition_html(filename=str(path))
    return "\n".join(e.text for e in elements if getattr(e, "text", "").strip())

# Keyword hints for the two objective claim-list fields. These only SURFACE
# candidate lines for the labeler to verify — they are not the worker model
# and are not authoritative (keeps the gold set free of circular labels).
_MILESTONE_HINTS = re.compile(
    r"partnership|collaborat|agreement|contract|award|approval|clearance|"
    r"designation|milestone|launch|qualif|definitive",
    re.IGNORECASE,
)
_DILUTION_HINTS = re.compile(
    r"offering|shares of common stock|at-the-market|shelf|registration statement|"
    r"private placement|convertible|warrant|proceeds|underwrit|dilut",
    re.IGNORECASE,
)


def extract_item_section(text: str, *, max_chars: int = 4000) -> str:
    """Trim cover-page + signature/exhibit boilerplate to the material items."""
    upper = text.upper()
    start_m = re.search(r"ITEM\s+\d+\.\d+", upper)
    start = start_m.start() if start_m else 0
    tail = upper[start:]
    end_m = re.search(r"\n\s*SIGNATURE", tail)
    end = start + end_m.start() if end_m else len(text)
    return text[start:end].strip()[:max_chars]


def _candidate_lines(section: str, pattern: re.Pattern[str]) -> list[str]:
    """Verbatim lines of `section` matching `pattern` (deduped, capped)."""
    seen: list[str] = []
    for line in section.splitlines():
        s = line.strip()
        if len(s) > 25 and pattern.search(s) and s not in seen:
            seen.append(s)
    return seen[:6]


def write_worksheet(out_dir: Path, doc_id: str, meta: dict[str, str], section: str) -> None:
    doc_dir = out_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "source.txt").write_text(section)
    labels = {
        "doc_id": doc_id,
        "_meta": meta,
        "_instructions": (
            "Fill event_classification (one of "
            f"{[e.value for e in EventClassification]}). Copy EXACT quote spans "
            "from source.txt into the two lists; empty lists are fine. Verify and "
            "prune _candidates — they are keyword hints, not answers."
        ),
        "event_classification": "",
        "milestone_mentions": [],
        "dilution_language_flags": [],
        "rationale": "",
        "_candidates": {
            "milestone_mentions": _candidate_lines(section, _MILESTONE_HINTS),
            "dilution_language_flags": _candidate_lines(section, _DILUTION_HINTS),
        },
    }
    (doc_dir / "labels.json").write_text(json.dumps(labels, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--per", type=int, default=1)
    args = ap.parse_args()

    out_dir = project_root() / "eval" / "gold_sets" / "_worksheets" / "eight_k"
    raw_root = project_root() / "data" / "raw"
    t2c = resolve_ciks_from_env()

    written = 0
    with EdgarClient() as client:
        for ticker in args.tickers:
            cik = t2c.get(ticker.upper())
            if cik is None:
                print(f"!! {ticker}: no CIK")
                continue
            filings = client.list_recent_filings(cik, form_types=["8-K"])
            for filing in filings[: args.per]:
                if not filing.primary_document.endswith((".htm", ".html")):
                    continue  # skip legacy .txt filings for now
                path, _sha, _ = client.fetch_filing(filing, raw_root=raw_root / "edgar")
                section = extract_item_section(clean_filing_text(path))
                if len(section) < 80:
                    print(f"-- {ticker} {filing.accession_number}: thin section, skipped")
                    continue
                acc = filing.accession_number.replace("-", "")
                doc_id = f"{ticker.upper()}-{acc}"
                write_worksheet(
                    out_dir,
                    doc_id,
                    {
                        "ticker": ticker.upper(),
                        "cik": cik,
                        "accession": filing.accession_number,
                        "accepted": filing.accepted_datetime.isoformat(),
                    },
                    section,
                )
                written += 1
                print(f"wrote worksheet {doc_id} ({len(section)} chars)")
    print(f"\n{written} worksheets under {out_dir}")


if __name__ == "__main__":
    main()
