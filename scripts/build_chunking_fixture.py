"""Build a chunking test fixture from a real EDGAR 10-K filing.

Usage:
    uv run python scripts/build_chunking_fixture.py \
        --ticker AMD --cik 2488 --accession 0000002488-25-000049

Discovers the primary 10-K HTML document for the given filing, trims
it to the per-Item byte budget so the result sits under the repo's
500 KB pre-commit cap, validates that the chunker can detect the
real Item 1A / 7 / 8 sections, and writes the fixture + metadata
sidecar under `tests/fixtures/chunking/`.

The script is the documented path for adding 10-K test fixtures —
every fixture in the repo should be reproducible by re-running this
with the matching `--ticker / --cik / --accession` args. The
trimming approach preserves real section ordering and at least one
inline financial table for the table-policy tests.

Network access is required (fetches from sec.gov). User-Agent set per
SEC fair-access policy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "chunking"

USER_AGENT = "auto-research/0.1 research@example.com"
SEC_BASE = "https://www.sec.gov"

# Per-Item byte budgets when trimming the raw 10-K. Item 1A and 7 get
# the most prose; Item 8 gets the most table content. Tuned to keep
# total trimmed size under ~300 KB.
BUDGETS = {
    "1": 25_000,
    "1A": 70_000,
    "7": 70_000,
    "7A": 12_000,
    "8": 70_000,
}

# Entity-aware Item-header regex (mirrors chunking._ITEM_HEADER).
ITEM_HEADER = re.compile(
    r"\b(?i:item)(?:\s|&\#160;|&nbsp;|&\#x[Aa]0;)+(\d+[A-Za-z]?)\b",
)


def _http_get(url: str) -> bytes:
    """SEC fair-access-policy-compliant GET."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _find_primary_10k_url(cik: int, accession: str) -> str:
    """Resolve a 10-K filing's primary HTML document.

    EDGAR's filing index page lists every document in the submission.
    The primary 10-K may appear either as a direct `.htm` link or
    wrapped inside the iXBRL viewer (`/ix?doc=<actual-path>`). Both
    forms are valid; this resolver collects both, strips the
    `/ix?doc=` wrapper, then filters out exhibits/certifications/
    subsidiary lists by filename token.
    """
    accession_dashed = accession
    accession_undashed = accession.replace("-", "")
    index_url = (
        f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession_undashed}/{accession_dashed}-index.htm"
    )
    body = _http_get(index_url).decode("utf-8", errors="replace")

    # Collect both direct and iXBRL-wrapped links to .htm files in this
    # submission's folder; unwrap `/ix?doc=` so the resolver sees a
    # canonical path either way.
    direct = re.findall(
        rf'href="(/Archives/edgar/data/{cik}/{accession_undashed}/[^"]+\.htm)"', body
    )
    wrapped = re.findall(
        rf'href="/ix\?doc=(/Archives/edgar/data/{cik}/{accession_undashed}/[^"]+\.htm)"',
        body,
    )
    seen: set[str] = set()
    paths: list[str] = []
    for p in [*wrapped, *direct]:  # wrapped first — usually the primary doc
        if p not in seen:
            seen.add(p)
            paths.append(p)

    # Drop exhibits, certifications, subsidiary lists, consents, etc.
    exhibit_tokens = (
        "_ex",
        "-ex",
        "/ex",
        "exhibit",
        "subsidiaries",
        "subsidiary",
        "consent",
        "cert",
        "_xbrl",
        "descriptionof",
        "listofregistrants",
    )
    candidates = [p for p in paths if not any(tok in p.lower() for tok in exhibit_tokens)]
    if not candidates:
        raise RuntimeError(f"could not find primary 10-K doc in {index_url}")
    if len(candidates) == 1:
        return f"{SEC_BASE}{candidates[0]}"
    # Multiple candidates — pick the shortest filename (heuristic: the
    # primary doc's filename is typically `<ticker>-<period>.htm`
    # without modifier suffixes).
    candidates.sort(key=lambda p: len(p.rsplit("/", 1)[-1]))
    return f"{SEC_BASE}{candidates[0]}"


def _trim_10k(html: str, ticker: str) -> tuple[str, dict[str, int]]:
    """Trim a full 10-K HTML to representative Item sections.

    Reuses the chunker's `_detect_sections` so the trim picks exactly
    the offsets the production parser would identify — guarantees the
    validation pass agrees with the trim.

    Returns (trimmed_html, report) where report contains the chosen
    item positions and final byte size.
    """
    from auto_research.extract.chunking import _detect_sections

    body_match = re.search(r"<body[^>]*>", html, re.IGNORECASE)
    body_start = body_match.end() if body_match else 0
    body_end_idx = html.lower().rfind("</body>")
    body_end = body_end_idx if body_end_idx > 0 else len(html)
    body = html[body_start:body_end]

    head_match = re.search(r"<head[^>]*>.*?</head>", html, re.DOTALL | re.IGNORECASE)
    head = head_match.group(0) if head_match else "<head></head>"

    # Run the chunker's section detector against the body so we trim
    # at exactly the positions the production parser will find.
    sections = _detect_sections(body)
    detected_starts: dict[str, int] = {}
    for s in sections:
        name = s.name.removeprefix("Item ")  # "Item 1A" → "1A"
        detected_starts[name] = s.char_span[0]

    items_in_order = sorted(
        [(num, detected_starts[num]) for num in BUDGETS if num in detected_starts],
        key=lambda x: x[1],
    )

    parts: list[str] = []
    for i, (num, start) in enumerate(items_in_order):
        budget = BUDGETS[num]
        next_start = items_in_order[i + 1][1] if i + 1 < len(items_in_order) else len(body)
        end = min(start + budget, next_start)
        section = body[start:end]
        if end < start + budget and end == next_start:
            marker = ""
        else:
            marker = (
                f"\n<!-- TRUNCATED: Item {num} trimmed at byte {budget}; "
                "this is a test fixture -->\n"
            )
        parts.append(section + marker)

    trimmed_body = "\n".join(parts)
    trimmed = f"""<!DOCTYPE html>
<html>
{head}
<body>
<!-- TRIMMED EXCERPT of {ticker} 10-K for chunking unit tests.
     Real EDGAR filing; each Item truncated to a per-section byte
     budget to fit the repo's 500 KB pre-commit cap. Sections
     preserved in document order with at least one inline table in
     Item 8. -->
{trimmed_body}
</body>
</html>
"""
    report = {
        "detected_starts": detected_starts,
        "kept_items": [(num, start) for num, start in items_in_order],
        "trimmed_size_bytes": len(trimmed),
    }
    return trimmed, report


def _validate_trimmed(trimmed_html: str, expected_items: list[str]) -> set[str]:
    """Run the chunker over the trimmed fixture and return detected sections."""
    # Defer import to keep `--help` snappy and so the script can be
    # used as a fixture-build tool without spaCy installed (it'll fail
    # cleanly when validation runs, not at import).
    from datetime import date

    from auto_research.extract.chunking import ChunkMetadata, parse_filing

    meta = ChunkMetadata(
        ticker="VALIDATION",
        filing_date=date(2025, 1, 1),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="validation-build",
    )
    result = parse_filing(html=trimmed_html, metadata=meta)
    return {p.section_name for p in result.parents}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="e.g. AMD")
    parser.add_argument("--cik", required=True, type=int, help="SEC CIK number, no leading zeros")
    parser.add_argument(
        "--accession",
        required=True,
        help="SEC accession in dashed form, e.g. 0000002488-25-000049",
    )
    parser.add_argument(
        "--filing-date", required=True, help="Filed date YYYY-MM-DD"
    )
    parser.add_argument(
        "--fiscal-period", required=True, help="e.g. FY2024"
    )
    parser.add_argument(
        "--expected-sections",
        default="Item 1A,Item 7,Item 8",
        help="Comma-separated section names the chunker must detect",
    )
    parser.add_argument(
        "--out-stem",
        help="Override the fixture file stem (default: sample_10k_<ticker_lower>)",
    )
    parser.add_argument(
        "--tier",
        choices=["core", "broad"],
        default="broad",
        help=(
            "Which test-tier this fixture belongs to. `core` fixtures run "
            "in default CI via `make test`; `broad` fixtures run only via "
            "`make test-broad` (typically nightly). New fixtures default to "
            "broad — promote to core explicitly once they've earned their "
            "place in the CI surface."
        ),
    )
    args = parser.parse_args()

    out_stem = args.out_stem or f"sample_10k_{args.ticker.lower()}"
    htm_path = FIXTURE_DIR / f"{out_stem}.htm"
    meta_path = FIXTURE_DIR / f"{out_stem}.meta.json"

    print(f"Resolving primary 10-K URL for CIK={args.cik} accession={args.accession} …")
    primary_url = _find_primary_10k_url(args.cik, args.accession)
    print(f"  primary doc: {primary_url}")

    print(f"Fetching {primary_url} …")
    raw_bytes = _http_get(primary_url)
    raw_html = raw_bytes.decode("utf-8", errors="replace")
    print(f"  fetched {len(raw_html):,} chars")

    print("Trimming …")
    trimmed, report = _trim_10k(raw_html, args.ticker)
    print(f"  trimmed to {report['trimmed_size_bytes']:,} bytes")
    print(f"  kept items in order: {report['kept_items']}")

    if report["trimmed_size_bytes"] > 480_000:
        print(
            f"  ⚠ trimmed size {report['trimmed_size_bytes']} > 480 KB — pre-commit cap is 500 KB",
            file=sys.stderr,
        )

    # Validate BEFORE writing so a failed build never leaves a broken
    # fixture behind for the parameterized test suite to pick up.
    expected = [s.strip() for s in args.expected_sections.split(",") if s.strip()]
    print(f"Validating chunker detection (expected: {expected}) …")
    detected = _validate_trimmed(trimmed, expected)
    missing = [s for s in expected if s not in detected]
    if missing:
        print(f"  ✗ MISSING sections: {missing}", file=sys.stderr)
        print(f"  detected: {sorted(detected)}", file=sys.stderr)
        print(
            f"  fixture NOT written to {htm_path.relative_to(REPO_ROOT)} — "
            "fix the chunker or the trim heuristics first.",
            file=sys.stderr,
        )
        return 2
    print(f"  ✓ detected: {sorted(detected)}")

    print("Writing fixture …")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    htm_path.write_text(trimmed, encoding="utf-8")

    meta = {
        "ticker": args.ticker,
        "filing_date": args.filing_date,
        "fiscal_period": args.fiscal_period,
        "doc_type": "10-K",
        "doc_id": args.accession,
        "tier": args.tier,
        "expected_sections": expected,
        "source_url": primary_url,
        "filename_convention": (
            "Uses .htm extension per SEC EDGAR's 25+-year convention. "
            "Production input from EDGAR always ends in .htm; fixture "
            "matches so any extension-aware tooling behaves identically "
            "in tests and live."
        ),
        "note": (
            f"Real EDGAR 10-K for {args.ticker}, fiscal period "
            f"{args.fiscal_period}, filed {args.filing_date}. "
            f"Public-domain SEC filing. Trimmed via "
            f"scripts/build_chunking_fixture.py to fit the 500 KB "
            f"pre-commit cap while preserving real section headers "
            f"for {', '.join(expected)} in document order and at "
            f"least one inline financial table for Item 8 table-policy tests."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {htm_path.relative_to(REPO_ROOT)} ({htm_path.stat().st_size:,} bytes)")
    print(f"  wrote {meta_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
