"""Live smoke for the SEC EDGAR client.

Hits real `data.sec.gov` and `www.sec.gov`. Confirms:

1. `fetch_filings_for_cik` orchestrates list → fetch → manifest → re-run.
2. Re-running the same call is a no-op (every result `cache_hit=True`).
3. The async path `afetch_filings_for_cik` exercises the same flow with
   bounded concurrency without breaking SEC's fair-access budget.

What this catches that VCR + units can't:

- SEC schema drift (a field rename in `filings.recent` would surface here
  before any user hits it).
- SEC actively blocking our UA / IP (UA policy enforcement changes).
- Redirect / TLS / DNS changes upstream.
- The rate limiter actually pacing real traffic at ≤ 8 r/s.

Cadence: nightly via `.github/workflows/live-smoke.yml`. Locally:
`SEC_USER_AGENT="your name your@email" make live-smoke`.

Scoped to NVDA + `("10-K",)` to keep the smoke under ~20 s wall-clock
on a cold manifest: NVDA has ~5 10-Ks in `recent`, well under SEC's
fair-access ceiling at 8 r/s + retries.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from auto_research.ingest import edgar, manifest

live_requires_env = ("SEC_USER_AGENT",)

NVDA_CIK = 1045810


def test_edgar_sync_roundtrip_against_real_sec(live_tmpdir: Path) -> None:
    """Cold-run fetches; warm re-run reports every filing as cache_hit."""
    raw_root = live_tmpdir / "raw"
    manifest_path = live_tmpdir / "manifest.parquet"

    first = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        raw_root=raw_root,
        manifest_path=manifest_path,
        form_types=("10-K",),
    )
    assert len(first) >= 1, "NVDA should always have at least one recent 10-K"
    assert all(r.cache_hit is False for r in first), "cold run should be all fresh fetches"
    for result in first:
        assert result.path is not None
        assert result.path.exists()
        assert result.content_sha256 is not None
        assert len(result.content_sha256) == 64  # sha256 hex

    # Manifest reflects the work.
    table = manifest.read(manifest_path)
    assert table.num_rows == len(first)
    assert set(table.column("doc_id").to_pylist()) == {r.accession_number for r in first}

    # Re-run is a full no-op.
    second = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        raw_root=raw_root,
        manifest_path=manifest_path,
        form_types=("10-K",),
    )
    assert {r.accession_number for r in second} == {r.accession_number for r in first}
    assert all(r.cache_hit is True for r in second), "warm re-run should be all manifest hits"
    # Manifest didn't grow.
    assert manifest.read(manifest_path).num_rows == len(first)


def test_edgar_async_smoke_against_real_sec(live_tmpdir: Path) -> None:
    """Async fan-out works end-to-end with bounded concurrency.

    Uses a fresh manifest path so it isn't a cache hit from the sync
    test above. Shares the SEC budget naturally — `sec_rate_limiter()`
    defaults to 8 r/s, and `concurrency=2` keeps the in-flight queue
    small.
    """
    raw_root = live_tmpdir / "raw_async"
    manifest_path = live_tmpdir / "manifest_async.parquet"

    async def runner() -> list[edgar.FetchResult]:
        return await edgar.afetch_filings_for_cik(
            NVDA_CIK,
            raw_root=raw_root,
            manifest_path=manifest_path,
            form_types=("10-K",),
            concurrency=2,
        )

    results = asyncio.run(runner())
    assert len(results) >= 1
    assert all(r.cache_hit is False for r in results)
    for result in results:
        assert result.path is not None
        assert result.path.exists()
    assert manifest.read(manifest_path).num_rows == len(results)


@pytest.mark.parametrize("form_type", ["10-K", "8-K"])
def test_edgar_form_filter_real_response(live_tmpdir: Path, form_type: str) -> None:
    """Every returned filing has the requested form (parser doesn't widen).

    Catches a class of bugs where the spec adds a new variant (e.g.,
    "10-K/A" amendments) and our filter accidentally includes it
    because of how SEC encodes the form name in `filings.recent`.
    """
    with edgar.EdgarClient() as client:
        filings = client.list_recent_filings(NVDA_CIK, form_types=(form_type,))
    assert len(filings) >= 1, f"NVDA should always have at least one recent {form_type}"
    assert all(f.form_type == form_type for f in filings)
