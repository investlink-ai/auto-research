"""Unit tests for the SEC EDGAR client (Issue #5).

HTTP layer is faked via `httpx.MockTransport`; only the unit logic
(idempotency, path layout, SHA-256, env validation, dedup, retries,
parser robustness) is exercised here. The full network shape is
covered by the VCR integration test under `tests/integration/`.

All tests construct the client with a high-rate `TokenBucket` so the
real SEC fair-access throttle (8 r/s) doesn't slow the suite. The
sec_rate_limiter() default is exercised via `test_rate_limit.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from auto_research.ingest import edgar, manifest
from auto_research.ingest.rate_limit import TokenBucket

NVDA_CIK = 1045810
NVDA_CIK_PADDED = f"{NVDA_CIK:010d}"


def _fast_limiter() -> TokenBucket:
    """1000 r/s — effectively no-op for tests; keeps the suite fast."""
    return TokenBucket(rate=1000.0, capacity=1000.0)


def _submissions_payload(
    *,
    accessions: list[str],
    forms: list[str],
    primary_docs: list[str],
    acceptance_dts: list[str],
    filing_dates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "cik": str(NVDA_CIK),
        "filings": {
            "recent": {
                "accessionNumber": accessions,
                "form": forms,
                "primaryDocument": primary_docs,
                "acceptanceDateTime": acceptance_dts,
                "filingDate": filing_dates or ["2024-02-21"] * len(accessions),
            },
            "files": [],
        },
    }


def _fake_transport(
    submissions: dict[str, Any],
    docs: dict[str, bytes],
) -> httpx.MockTransport:
    """Mock SEC endpoints. `docs` keys are primary_document filenames."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://data.sec.gov/submissions/"):
            assert request.headers.get("User-Agent"), "User-Agent required by SEC"
            return httpx.Response(200, json=submissions)
        if url.startswith("https://www.sec.gov/Archives/"):
            assert request.headers.get("User-Agent"), "User-Agent required by SEC"
            doc_name = url.rsplit("/", 1)[-1]
            if doc_name in docs:
                return httpx.Response(200, content=docs[doc_name])
        return httpx.Response(404, text=f"unhandled URL {url}")

    return httpx.MockTransport(handler)


def _scripted_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> edgar.EdgarClient:
    monkeypatch.setenv("SEC_USER_AGENT", "Test Suite test@example.com")
    submissions = _submissions_payload(
        accessions=[
            "0001045810-24-000316",
            "0001045810-24-000209",
            "0001045810-24-000100",
            "0001045810-24-000050",
        ],
        forms=["10-K", "8-K", "S-3", "DEF 14A"],
        primary_docs=[
            "nvda-20240128.htm",
            "nvda-8k.htm",
            "nvda-s3.htm",
            "nvda-proxy.htm",
        ],
        acceptance_dts=[
            "2024-02-21T16:31:00.000Z",
            "2024-08-28T16:30:00.000Z",
            "2024-09-15T08:00:00.000Z",
            "2024-04-10T16:30:00.000Z",
        ],
    )
    docs = {
        "nvda-20240128.htm": b"<html>10-K body</html>",
        "nvda-8k.htm": b"<html>8-K body</html>",
        "nvda-s3.htm": b"<html>S-3 body</html>",
        "nvda-proxy.htm": b"<html>proxy</html>",
    }
    return edgar.EdgarClient(
        transport=_fake_transport(submissions, docs),
        rate_limiter=_fast_limiter(),
    )


# ---------- env validation ----------


def test_missing_user_agent_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(edgar.EdgarConfigError, match="SEC_USER_AGENT"):
        edgar.EdgarClient()


def test_blank_user_agent_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    with pytest.raises(edgar.EdgarConfigError):
        edgar.EdgarClient()


# ---------- end-to-end fetch ----------


def test_fetch_writes_to_canonical_path(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=raw_root,
        manifest_path=manifest_path,
        form_types=("10-K",),
    )
    assert len(results) == 1
    r = results[0]
    assert r.cache_hit is False
    assert r.path == raw_root / "edgar" / NVDA_CIK_PADDED / "2024" / "0001045810-24-000316.htm"
    assert r.path.exists()
    assert r.path.read_bytes() == b"<html>10-K body</html>"


def test_form_filter_excludes_non_target_forms(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
        form_types=("10-K", "8-K", "S-3"),
    )
    forms = sorted(r.form_type for r in results)
    assert forms == ["10-K", "8-K", "S-3"]


def test_manifest_row_carries_sha256_and_event_datetime(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=tmp_path / "raw",
        manifest_path=manifest_path,
        form_types=("10-K",),
    )
    table = manifest.read(manifest_path)
    assert table.num_rows == 1
    row = {col: table.column(col)[0].as_py() for col in table.schema.names}
    expected_sha = hashlib.sha256(b"<html>10-K body</html>").hexdigest()
    assert row["content_sha256"] == expected_sha
    assert row["source"] == "edgar"
    assert row["doc_id"] == "0001045810-24-000316"
    assert row["form_type"] == "10-K"
    assert row["event_datetime"].year == 2024
    assert row["event_datetime"].month == 2
    assert row["event_datetime"].day == 21


def test_rerun_is_no_op_via_manifest_hit(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    first = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=raw_root,
        manifest_path=manifest_path,
        form_types=("10-K", "8-K", "S-3"),
    )
    assert all(r.cache_hit is False for r in first)
    second = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=raw_root,
        manifest_path=manifest_path,
        form_types=("10-K", "8-K", "S-3"),
    )
    assert all(r.cache_hit is True for r in second)
    assert manifest.read(manifest_path).num_rows == 3


def test_partial_rerun_only_fetches_new_filings(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    # Pre-seed manifest with the 10-K only.
    manifest.append(
        manifest_path,
        [
            {
                "source": "edgar",
                "entity_id": NVDA_CIK_PADDED,
                "doc_id": "0001045810-24-000316",
                "form_type": "10-K",
                "event_datetime": datetime(2024, 2, 21, 16, 31, tzinfo=UTC),
                "fetched_at": datetime(2026, 5, 22, tzinfo=UTC),
                "content_sha256": "f" * 64,
                "path": "stale-path",
                "status": "ok",
            }
        ],
    )
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=tmp_path / "raw",
        manifest_path=manifest_path,
        form_types=("10-K", "8-K", "S-3"),
    )
    by_doc = {r.accession_number: r for r in results}
    assert by_doc["0001045810-24-000316"].cache_hit is True
    assert by_doc["0001045810-24-000209"].cache_hit is False
    assert by_doc["0001045810-24-000100"].cache_hit is False
    assert manifest.read(manifest_path).num_rows == 3


# ---------- code-review fixes ----------


def test_duplicate_accession_in_response_is_deduped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If SEC returns the same accession twice, the second occurrence is skipped."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["0001045810-24-000316", "0001045810-24-000316"],
        forms=["10-K", "10-K/A"],
        primary_docs=["nvda-10k.htm", "nvda-10k.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z"] * 2,
    )
    docs = {"nvda-10k.htm": b"<html>10-K body</html>"}
    client = edgar.EdgarClient(
        transport=_fake_transport(submissions, docs),
        rate_limiter=_fast_limiter(),
    )
    manifest_path = tmp_path / "manifest.parquet"
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=client,
        raw_root=tmp_path / "raw",
        manifest_path=manifest_path,
        form_types=("10-K", "10-K/A"),
    )
    assert len(results) == 1
    assert manifest.read(manifest_path).num_rows == 1


def test_empty_primary_document_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty primaryDocument would request a directory listing — skip it instead."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["0001045810-24-000316", "0001045810-24-000400"],
        forms=["10-K", "8-K"],
        primary_docs=["", "valid.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z", "2024-03-01T10:00:00Z"],
    )
    docs = {"valid.htm": b"<html>8-K</html>"}
    client = edgar.EdgarClient(
        transport=_fake_transport(submissions, docs),
        rate_limiter=_fast_limiter(),
    )
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=client,
        raw_root=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
        form_types=("10-K", "8-K"),
    )
    assert [r.accession_number for r in results] == ["0001045810-24-000400"]


def test_misaligned_recent_arrays_raise_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """zip(strict=True) must surface a ValueError on ragged parallel arrays."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    bad_submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["A", "B"],
                "form": ["10-K"],  # ragged
                "primaryDocument": ["a.htm", "b.htm"],
                "acceptanceDateTime": ["2024-02-21T16:31:00Z", "2024-03-01T10:00:00Z"],
            }
        }
    }
    client = edgar.EdgarClient(
        transport=_fake_transport(bad_submissions, {}),
        rate_limiter=_fast_limiter(),
    )
    with pytest.raises(ValueError):
        client.list_recent_filings(NVDA_CIK, form_types=("10-K",))


def test_partial_failure_flushes_already_fetched_rows_to_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crash mid-loop must persist what already succeeded; no orphans."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["0001045810-24-000001", "0001045810-24-000002", "0001045810-24-000003"],
        forms=["10-K", "10-K", "10-K"],
        primary_docs=["a.htm", "b.htm", "BOOM.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z"] * 3,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(200, json=submissions)
        if url.endswith("a.htm"):
            return httpx.Response(200, content=b"A")
        if url.endswith("b.htm"):
            return httpx.Response(200, content=b"B")
        # Persistent 500 on filing 3 — retries exhaust, exception escapes.
        return httpx.Response(500, text="boom")

    client = edgar.EdgarClient(
        transport=_scripted_transport(handler),
        rate_limiter=_fast_limiter(),
        max_attempts=2,
    )
    manifest_path = tmp_path / "manifest.parquet"
    with pytest.raises(edgar.EdgarServerError):
        edgar.fetch_filings_for_cik(
            NVDA_CIK,
            client=client,
            raw_root=tmp_path / "raw",
            manifest_path=manifest_path,
            form_types=("10-K",),
        )
    # Filings 1 & 2 must be in the manifest even though 3 raised.
    recorded = manifest.read(manifest_path).column("doc_id").to_pylist()
    assert sorted(recorded) == ["0001045810-24-000001", "0001045810-24-000002"]


def test_retries_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient 5xx should be retried and eventually succeed."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="bad gateway")
        return httpx.Response(
            200,
            json=_submissions_payload(
                accessions=["A"],
                forms=["10-K"],
                primary_docs=["a.htm"],
                acceptance_dts=["2024-02-21T16:31:00Z"],
            ),
        )

    client = edgar.EdgarClient(
        transport=_scripted_transport(handler),
        rate_limiter=_fast_limiter(),
        max_attempts=5,
    )
    filings = client.list_recent_filings(NVDA_CIK, form_types=("10-K",))
    assert calls["n"] == 3
    assert len(filings) == 1


def test_429_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 with Retry-After is followed (and counted as a retry attempt)."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    monkeypatch.setattr("time.sleep", lambda _: None)  # don't actually sleep
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200,
            json=_submissions_payload(
                accessions=["A"],
                forms=["10-K"],
                primary_docs=["a.htm"],
                acceptance_dts=["2024-02-21T16:31:00Z"],
            ),
        )

    client = edgar.EdgarClient(
        transport=_scripted_transport(handler),
        rate_limiter=_fast_limiter(),
        max_attempts=3,
    )
    filings = client.list_recent_filings(NVDA_CIK, form_types=("10-K",))
    assert calls["n"] == 2
    assert len(filings) == 1


def test_follows_3xx_redirects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SEC occasionally redirects Archives URLs; 3xx must not raise."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/old.htm"):
            return httpx.Response(301, headers={"Location": str(request.url).replace("old", "new")})
        if url.endswith("/new.htm"):
            return httpx.Response(200, content=b"redirected body")
        return httpx.Response(404)

    client = edgar.EdgarClient(
        transport=_scripted_transport(handler),
        rate_limiter=_fast_limiter(),
    )
    filing = edgar.Filing(
        cik=NVDA_CIK_PADDED,
        accession_number="0001045810-24-000001",
        form_type="10-K",
        primary_document="old.htm",
        accepted_datetime=datetime(2024, 2, 21, tzinfo=UTC),
    )
    _path, sha, content = client.fetch_filing(filing, raw_root=tmp_path / "raw")
    assert content == b"redirected body"
    assert sha == hashlib.sha256(b"redirected body").hexdigest()


def test_parse_acceptance_naive_treated_as_eastern() -> None:
    """`2024-02-21T20:30:00` (ET, 8:30 PM) → 2024-02-22 01:30 UTC."""
    dt = edgar._parse_acceptance("2024-02-21T20:30:00")
    assert dt == datetime(2024, 2, 22, 1, 30, tzinfo=UTC)


def test_parse_acceptance_z_suffix_is_utc() -> None:
    dt = edgar._parse_acceptance("2024-02-21T16:31:00Z")
    assert dt == datetime(2024, 2, 21, 16, 31, tzinfo=UTC)


# ---------- async smoke ----------


def test_async_client_fetches_concurrently(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """afetch_filings_for_cik orchestrates list + fetch with bounded concurrency."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["A", "B", "C"],
        forms=["10-K", "8-K", "S-3"],
        primary_docs=["a.htm", "b.htm", "c.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z"] * 3,
    )
    docs = {"a.htm": b"A", "b.htm": b"B", "c.htm": b"C"}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(200, json=submissions)
        doc_name = url.rsplit("/", 1)[-1]
        return httpx.Response(200, content=docs.get(doc_name, b""))

    async def runner() -> list[edgar.FetchResult]:
        client = edgar.AsyncEdgarClient(
            transport=httpx.MockTransport(handler),
            rate_limiter=_fast_limiter(),
        )
        try:
            return await edgar.afetch_filings_for_cik(
                NVDA_CIK,
                client=client,
                raw_root=tmp_path / "raw",
                manifest_path=tmp_path / "manifest.parquet",
                form_types=("10-K", "8-K", "S-3"),
                concurrency=2,
            )
        finally:
            await client.aclose()

    results = asyncio.run(runner())
    assert sorted(r.accession_number for r in results) == ["A", "B", "C"]
    assert all(r.cache_hit is False for r in results)
    assert manifest.read(tmp_path / "manifest.parquet").num_rows == 3


# ---------- post-review fixes ----------


def test_filing_rejects_unpadded_cik() -> None:
    """Filing.__post_init__ enforces zero-padded 10-digit CIK form."""
    with pytest.raises(ValueError, match="zero-padded 10 digits"):
        edgar.Filing(
            cik="1045810",  # unpadded
            accession_number="0001045810-24-000001",
            form_type="10-K",
            primary_document="a.htm",
            accepted_datetime=datetime(2024, 2, 21, tzinfo=UTC),
        )


def test_filing_rejects_blank_accession_number() -> None:
    with pytest.raises(ValueError, match="accession_number must be non-empty"):
        edgar.Filing(
            cik=NVDA_CIK_PADDED,
            accession_number="   ",
            form_type="10-K",
            primary_document="a.htm",
            accepted_datetime=datetime(2024, 2, 21, tzinfo=UTC),
        )


def test_filing_rejects_blank_primary_document() -> None:
    with pytest.raises(ValueError, match="primary_document must be non-empty"):
        edgar.Filing(
            cik=NVDA_CIK_PADDED,
            accession_number="0001045810-24-000001",
            form_type="10-K",
            primary_document="",
            accepted_datetime=datetime(2024, 2, 21, tzinfo=UTC),
        )


def test_parse_recent_skips_empty_accession(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SEC payload with blank accession should be skipped, not silently admitted."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["", "0001045810-24-000200"],
        forms=["10-K", "8-K"],
        primary_docs=["valid-a.htm", "valid-b.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z", "2024-03-01T10:00:00Z"],
    )
    docs = {"valid-b.htm": b"<html>8-K</html>"}
    client = edgar.EdgarClient(
        transport=_fake_transport(submissions, docs),
        rate_limiter=_fast_limiter(),
    )
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=client,
        raw_root=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
        form_types=("10-K", "8-K"),
    )
    assert [r.accession_number for r in results] == ["0001045810-24-000200"]


def test_429_does_not_double_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry-After is applied via tenacity wait, not inside _do_request.

    Regression for the previous double-sleep bug: a 429 used to sleep
    Retry-After inside _do_request AND then sleep again under
    tenacity's wait_exponential_jitter. Now the inline sleep is gone
    and tenacity's wait callback honors EdgarRateLimited.retry_after.
    """
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda t: sleeps.append(t))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(
            200,
            json=_submissions_payload(
                accessions=["0001045810-24-000001"],
                forms=["10-K"],
                primary_docs=["a.htm"],
                acceptance_dts=["2024-02-21T16:31:00Z"],
            ),
        )

    client = edgar.EdgarClient(
        transport=_scripted_transport(handler),
        rate_limiter=_fast_limiter(),
        max_attempts=3,
    )
    filings = client.list_recent_filings(NVDA_CIK, form_types=("10-K",))
    assert len(filings) == 1
    # Should sleep exactly once (the tenacity wait), and that sleep must be
    # at least the Retry-After hint of 5s.
    assert len(sleeps) == 1
    assert sleeps[0] >= 5.0


def test_async_partial_failure_flushes_completed_siblings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """gather(return_exceptions=True) lets siblings finish before the flush.

    Also verifies that failures are surfaced as a BaseExceptionGroup
    so callers see every failure mode, not just the first one.
    """
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["A1", "A2", "A3"],
        forms=["10-K", "10-K", "10-K"],
        primary_docs=["good.htm", "good2.htm", "BOOM.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z"] * 3,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(200, json=submissions)
        if url.endswith("good.htm"):
            return httpx.Response(200, content=b"A")
        if url.endswith("good2.htm"):
            return httpx.Response(200, content=b"B")
        return httpx.Response(500, text="boom")

    captured: list[BaseException] = []

    async def runner() -> None:
        client = edgar.AsyncEdgarClient(
            transport=httpx.MockTransport(handler),
            rate_limiter=_fast_limiter(),
            max_attempts=2,
        )
        try:
            try:
                await edgar.afetch_filings_for_cik(
                    NVDA_CIK,
                    client=client,
                    raw_root=tmp_path / "raw",
                    manifest_path=tmp_path / "manifest.parquet",
                    form_types=("10-K",),
                    concurrency=3,
                )
            except BaseExceptionGroup as eg:
                captured.extend(eg.exceptions)
        finally:
            await client.aclose()

    asyncio.run(runner())
    assert captured, "afetch should have raised an ExceptionGroup"
    assert any(isinstance(e, edgar.EdgarServerError) for e in captured)
    # A1 and A2 completed; A3 raised. Both completed rows must be in the manifest.
    table = manifest.read(tmp_path / "manifest.parquet")
    recorded = set(table.column("doc_id").to_pylist())
    assert recorded == {"A1", "A2"}


def test_async_concurrency_zero_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """concurrency=0 would deadlock on asyncio.Semaphore(0); reject it eagerly."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")

    async def runner() -> None:
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await edgar.afetch_filings_for_cik(
                NVDA_CIK,
                raw_root=tmp_path / "raw",
                manifest_path=tmp_path / "manifest.parquet",
                form_types=("10-K",),
                concurrency=0,
            )

    asyncio.run(runner())


# ---------- second-round P1/P2 fixes ----------


def test_parse_recent_skips_empty_acceptance_datetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A blank acceptanceDateTime must skip just that row, not crash the whole parse."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["0001045810-24-000001", "0001045810-24-000002"],
        forms=["10-K", "10-K"],
        primary_docs=["bad.htm", "good.htm"],
        acceptance_dts=["", "2024-02-21T16:31:00Z"],
    )
    docs = {"good.htm": b"<html>good</html>"}
    client = edgar.EdgarClient(
        transport=_fake_transport(submissions, docs),
        rate_limiter=_fast_limiter(),
    )
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=client,
        raw_root=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
        form_types=("10-K",),
    )
    assert [r.accession_number for r in results] == ["0001045810-24-000002"]


def test_empty_response_body_raises_retryable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """200 OK + 0 bytes is treated as a transient SEC/CDN propagation issue.

    Without this guard, the manifest would record a sha256-of-empty
    row that becomes a permanent cache hit, blocking future retries.
    """
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(
                200,
                json=_submissions_payload(
                    accessions=["0001045810-24-000001"],
                    forms=["10-K"],
                    primary_docs=["a.htm"],
                    acceptance_dts=["2024-02-21T16:31:00Z"],
                ),
            )
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, content=b"")  # transient empty body
        return httpx.Response(200, content=b"<html>real</html>")

    client = edgar.EdgarClient(
        transport=_scripted_transport(handler),
        rate_limiter=_fast_limiter(),
        max_attempts=5,
    )
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=client,
        raw_root=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
        form_types=("10-K",),
    )
    assert len(results) == 1
    assert results[0].cache_hit is False
    # Subsequent runs must see the cached row (sha is of real content, not empty).
    assert results[0].content_sha256 == hashlib.sha256(b"<html>real</html>").hexdigest()
    # Retries actually fired before success.
    assert calls["n"] == 3


def test_async_io_dispatched_via_to_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sync I/O in the async path runs via asyncio.to_thread AND outcomes are correct.

    Asserts both:
    - `_http.atomic_write_bytes` and `manifest.append` were dispatched
      to `asyncio.to_thread` (not invoked synchronously on the loop).
    - The files landed on disk with the expected bytes and the
      manifest reflects every successful fetch. A regression that
      passes the right function with wrong args would fail outcome
      assertions even if the dispatch assertion passed.
    """
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["A", "B"],
        forms=["10-K", "10-K"],
        primary_docs=["a.htm", "b.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z"] * 2,
    )
    docs = {"a.htm": b"A-body", "b.htm": b"B-body"}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(200, json=submissions)
        return httpx.Response(200, content=docs.get(url.rsplit("/", 1)[-1], b""))

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    async def runner() -> None:
        client = edgar.AsyncEdgarClient(
            transport=httpx.MockTransport(handler),
            rate_limiter=_fast_limiter(),
        )
        try:
            await edgar.afetch_filings_for_cik(
                NVDA_CIK,
                client=client,
                raw_root=tmp_path / "raw",
                manifest_path=tmp_path / "manifest.parquet",
                form_types=("10-K",),
                concurrency=2,
            )
        finally:
            await client.aclose()

    asyncio.run(runner())

    # Dispatch assertions: both sync-IO entrypoints went through to_thread.
    from auto_research.ingest import _http
    from auto_research.ingest import manifest as manifest_mod

    assert _http.atomic_write_bytes in to_thread_calls
    assert manifest_mod.append in to_thread_calls

    # Outcome assertions: the threaded calls actually did the right thing.
    cik_dir = tmp_path / "raw" / "edgar" / NVDA_CIK_PADDED / "2024"
    assert (cik_dir / "A.htm").read_bytes() == b"A-body"
    assert (cik_dir / "B.htm").read_bytes() == b"B-body"
    table = manifest.read(tmp_path / "manifest.parquet")
    assert set(table.column("doc_id").to_pylist()) == {"A", "B"}


def test_edgar_empty_response_error_is_exported() -> None:
    """EdgarEmptyResponseError must be in __all__ alongside the other typed errors."""
    assert "EdgarEmptyResponseError" in edgar.__all__


def test_afetch_finally_flush_failure_doesnt_swallow_gather_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If manifest.append in the finally raises, the gather's per-filing
    exceptions must still surface — folded into the same BaseExceptionGroup.

    Regression: previously a flush error replaced the pending exceptions
    list and the caller never learned about the per-filing failures.
    """
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    submissions = _submissions_payload(
        accessions=["A1", "A2"],
        forms=["10-K", "10-K"],
        primary_docs=["good.htm", "BOOM.htm"],
        acceptance_dts=["2024-02-21T16:31:00Z"] * 2,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(200, json=submissions)
        if url.endswith("good.htm"):
            return httpx.Response(200, content=b"A")
        return httpx.Response(500, text="boom")

    # Make manifest.append (called via to_thread in the finally) raise.
    from auto_research.ingest import manifest as manifest_mod

    real_append = manifest_mod.append

    def flaky_append(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk-full at flush time")

    monkeypatch.setattr(manifest_mod, "append", flaky_append)
    captured: list[BaseException] = []

    async def runner() -> None:
        client = edgar.AsyncEdgarClient(
            transport=httpx.MockTransport(handler),
            rate_limiter=_fast_limiter(),
            max_attempts=2,
        )
        try:
            try:
                await edgar.afetch_filings_for_cik(
                    NVDA_CIK,
                    client=client,
                    raw_root=tmp_path / "raw",
                    manifest_path=tmp_path / "manifest.parquet",
                    form_types=("10-K",),
                    concurrency=2,
                )
            except BaseExceptionGroup as eg:
                captured.extend(eg.exceptions)
        finally:
            await client.aclose()
            monkeypatch.setattr(manifest_mod, "append", real_append)

    asyncio.run(runner())
    # Must surface BOTH the per-filing 5xx AND the flush OSError.
    assert any(isinstance(e, edgar.EdgarServerError) for e in captured)
    assert any(isinstance(e, OSError) and "simulated disk-full" in str(e) for e in captured)


# `test_atomic_write_bytes_cleans_tmp_on_failure` moved to test_http.py
# along with the helper itself when `_atomic_write_bytes` was extracted
# from edgar.py into the shared `_http.py` module.
