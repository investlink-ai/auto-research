"""Validate `extract.chunking._VALID_10K_ITEMS` against the SEC's
canonical Form 10-K instructions and the checked-in NVDA fixture.

Run with `uv run python scripts/validate_10k_items.py`. Network access
is required to fetch the SEC PDF; pass `--offline` to skip the SEC
cross-check and run only the fixture audit.

Reports:

1. The set of Item identifiers extracted from the official Form 10-K
   PDF (https://www.sec.gov/files/form10-k.pdf), parsed via pdftotext.
2. The whitelist defined in `_VALID_10K_ITEMS`.
3. Every `Item N` candidate in the NVDA fixture and how each is
   classified by the detection chain (whitelist → block-header →
   prose-density heuristic).
4. Symmetric diff (whitelist vs PDF) so any future Form 10-K
   amendment shows up here loudly.

This script is the audit evidence cited in the PR description; it is
NOT part of the unit-test gate (it requires network + pdftotext) but
can be re-run any time the SEC amends Form 10-K.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from auto_research.extract.chunking import (
    _ITEM_HEADER,
    _VALID_10K_ITEMS,
    _is_real_section_header,
    _looks_like_block_header,
    _mask_comments,
)

FORM_10K_URL = "https://www.sec.gov/files/form10-k.pdf"
USER_AGENT = "auto-research/0.1 research@example.com"

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "chunking" / "sample_10k_nvda.htm"


def _item_sort_key(item: str) -> tuple[int, str]:
    digits = "".join(c for c in item if c.isdigit())
    return (int(digits) if digits else 0, item)


def _fetch_sec_form_10k_items() -> set[str]:
    """Download the SEC Form 10-K PDF and return the Item identifiers
    mentioned anywhere in its body. Requires `pdftotext` on PATH
    (poppler-utils)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        req = urllib.request.Request(FORM_10K_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tmp.write(resp.read())
        pdf_path = tmp.name
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pdftotext is required (install poppler-utils: `brew install poppler`)"
        ) from exc
    finally:
        Path(pdf_path).unlink(missing_ok=True)

    items = set(re.findall(r"Item\s+(\d+[A-Za-z]?)\.", result.stdout))
    return items


def audit_fixture(fixture_html: str) -> tuple[
    dict[str, list[int]], dict[str, list[int]], set[str]
]:
    """Return (all_candidates, real_sections, unknown_numbers)."""
    masked = _mask_comments(fixture_html)
    all_candidates: dict[str, list[int]] = defaultdict(list)
    real_sections: dict[str, list[int]] = defaultdict(list)
    unknown_numbers: set[str] = set()

    for m in _ITEM_HEADER.finditer(masked):
        num = m.group(1).upper()
        all_candidates[num].append(m.start())
        if num not in _VALID_10K_ITEMS:
            unknown_numbers.add(num)
            continue
        if not _looks_like_block_header(fixture_html, m.start()):
            continue
        if not _is_real_section_header(fixture_html, m.start()):
            continue
        real_sections[num].append(m.start())

    return all_candidates, real_sections, unknown_numbers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Skip the SEC PDF cross-check."
    )
    args = parser.parse_args()

    whitelist_sorted = sorted(_VALID_10K_ITEMS, key=_item_sort_key)
    print("=== chunking._VALID_10K_ITEMS ===")
    print(f"  {whitelist_sorted}\n")

    if not args.offline:
        print(f"=== SEC Form 10-K PDF (fetched from {FORM_10K_URL}) ===")
        try:
            sec_items = _fetch_sec_form_10k_items()
        except Exception as exc:
            print(f"  fetch/parse failed: {exc}")
            print("  Re-run with --offline to skip this check.\n")
            return 2

        sec_sorted = sorted(sec_items, key=_item_sort_key)
        print(f"  Items mentioned in PDF: {sec_sorted}\n")

        # Symmetric diff
        only_in_whitelist = set(_VALID_10K_ITEMS) - sec_items
        only_in_pdf = sec_items - set(_VALID_10K_ITEMS)
        print("=== Diff: whitelist vs SEC PDF ===")
        if only_in_whitelist:
            print(f"  In whitelist but not in PDF: {sorted(only_in_whitelist, key=_item_sort_key)}")
        if only_in_pdf:
            print(f"  In PDF but not in whitelist: {sorted(only_in_pdf, key=_item_sort_key)}")
        if not only_in_whitelist and not only_in_pdf:
            print("  ✓ MATCH — whitelist tracks SEC Form 10-K exactly.\n")

    print(f"=== Fixture audit: {FIXTURE.relative_to(REPO_ROOT)} ===")
    fixture_html = FIXTURE.read_text(encoding="utf-8", errors="replace")
    all_candidates, real_sections, unknown_numbers = audit_fixture(fixture_html)

    print("  Item candidates found in fixture (any context):")
    for num in sorted(all_candidates, key=_item_sort_key):
        in_whitelist = "✓" if num in _VALID_10K_ITEMS else "✗ NOT IN WHITELIST (filtered)"
        print(
            f"    Item {num:4s}  {len(all_candidates[num]):3d} occurrences  [{in_whitelist}]"
        )
    if unknown_numbers:
        print(
            f"\n  Numbers seen but filtered out: {sorted(unknown_numbers, key=_item_sort_key)}"
            "  (e.g. rule references like 'Item 408 of Regulation S-K')"
        )
    real_sorted = sorted(real_sections.keys(), key=_item_sort_key)
    print(f"\n  Final sections detected after all filters: {real_sorted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
