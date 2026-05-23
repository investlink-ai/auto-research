"""Unit tests for the SEC EDGAR client (Issue #5).

HTTP layer is faked via `httpx.MockTransport`; only the unit logic
(idempotency, path layout, SHA-256, env validation) is exercised here.
The full network shape is covered by the VCR integration test under
`tests/integration/`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from auto_research.ingest import edgar, manifest

NVDA_CIK = 1045810
NVDA_CIK_PADDED = f"{NVDA_CIK:010d}"


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
    return edgar.EdgarClient(transport=_fake_transport(submissions, docs))


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
    # AC: data/raw/edgar/{cik}/{year}/{accession}.{ext}
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


# ---------- idempotency / manifest ----------


def test_manifest_row_carries_sha256_and_event_datetime(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=raw_root,
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
    # AC: accepted_datetime is the canonical PIT stamp.
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
    # Manifest didn't double-write.
    assert manifest.read(manifest_path).num_rows == 3


def test_partial_rerun_only_fetches_new_filings(
    fake_client: edgar.EdgarClient, tmp_path: Path
) -> None:
    """A pre-existing manifest row for one filing should skip just that one."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    # Seed the manifest with the 10-K only.
    manifest.append(manifest_path, [{
        "source": "edgar",
        "entity_id": NVDA_CIK_PADDED,
        "doc_id": "0001045810-24-000316",
        "form_type": "10-K",
        "event_datetime": __import__("datetime").datetime(2024, 2, 21, 16, 31, tzinfo=__import__("datetime").UTC),
        "fetched_at": __import__("datetime").datetime(2026, 5, 22, tzinfo=__import__("datetime").UTC),
        "content_sha256": "f" * 64,
        "path": "stale-path",
        "status": "ok",
    }])
    results = edgar.fetch_filings_for_cik(
        NVDA_CIK,
        client=fake_client,
        raw_root=raw_root,
        manifest_path=manifest_path,
        form_types=("10-K", "8-K", "S-3"),
    )
    by_doc = {r.accession_number: r for r in results}
    assert by_doc["0001045810-24-000316"].cache_hit is True
    assert by_doc["0001045810-24-000209"].cache_hit is False
    assert by_doc["0001045810-24-000100"].cache_hit is False
    # Manifest grew by exactly 2.
    assert manifest.read(manifest_path).num_rows == 3
