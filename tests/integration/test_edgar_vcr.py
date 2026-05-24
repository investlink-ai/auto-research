"""VCR-recorded integration test for the EDGAR client.

Covers one 10-K, one 8-K, and one S-3 (or S-3 variant) for a known
public issuer. The cassette is committed; CI replays it offline. To
regenerate, delete the cassette file and re-run with `SEC_USER_AGENT`
set — vcrpy's `record_mode="once"` records on absence and replays
otherwise.

VCR scrubs the User-Agent header (no recorder email in the cassette)
and truncates response bodies on record (`_MAX_BODY_BYTES`). Truncation
keeps the cassette under the 500 KB pre-commit threshold while still
covering everything the test asserts on: HTTP shape, path layout, and
internal SHA-256 consistency (sha-of-fetched matches sha-of-written).
The test deliberately does not pin a SHA against a known value, so
truncating bodies on re-record stays sound.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import vcr

from auto_research.ingest import edgar

CIK = 1045810  # NVIDIA
CASSETTE_PATH = (
    Path(__file__).parent / "cassettes" / "test_edgar" / "nvda_one_per_form.yaml"
)

# NVDA files plain "S-3" plus shelf variants; accept any S-3* form as the AC's "S-3".
_S3_PREFIX = "S-3"
_REQUESTED_FORMS: tuple[str, ...] = (
    "10-K",
    "8-K",
    "S-3",
    "S-3ASR",
    "S-3/A",
)
_MAX_BODY_BYTES = 8 * 1024  # plenty for sha + non-empty assertions; keeps cassette small


def _truncate_body(response: dict[str, Any]) -> dict[str, Any]:
    """Cap binary doc bodies before vcrpy serializes them to the cassette.

    Submissions JSON must be valid JSON or the parser blows up on
    replay, so skip truncation when the Content-Type signals JSON.
    The Archives endpoint returns HTML/PDF/text where arbitrary
    truncation is fine — the test only asserts non-empty + sha
    consistency, not parseability.
    """
    headers = response.get("headers", {}) or {}
    ctype_values: list[str] = []
    for key, value in headers.items():
        if key.lower() == "content-type":
            ctype_values.extend(value if isinstance(value, list) else [value])
    if any("json" in v.lower() for v in ctype_values):
        return response
    body = response.get("body", {})
    raw = body.get("string")
    if isinstance(raw, bytes) and len(raw) > _MAX_BODY_BYTES:
        body["string"] = raw[:_MAX_BODY_BYTES]
    return response


def _build_vcr() -> vcr.VCR:
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_PATH.parent),
        record_mode="once",
        filter_headers=[("User-Agent", "auto-research-test contact@example.com")],
        match_on=["method", "scheme", "host", "port", "path", "query"],
        decode_compressed_response=True,
        before_record_response=_truncate_body,
    )


@pytest.fixture
def sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the EdgarClient constructor happy.

    On replay the header is never actually sent; vcrpy serves the
    cassette regardless. On record we honour whatever the user has in
    their env so SEC sees a real contact address.
    """
    if not os.environ.get("SEC_USER_AGENT"):
        monkeypatch.setenv("SEC_USER_AGENT", "auto-research-test contact@example.com")


def test_fetch_10k_8k_s3_against_recorded_cassette(
    tmp_path: Path, sec_user_agent: None
) -> None:
    """End-to-end: list submissions, pick one 10-K / 8-K / S-3*, fetch each.

    Uses `EdgarClient` directly rather than `fetch_filings_for_cik` so
    the cassette stays bounded — only the four HTTP calls we care about
    get recorded instead of the full ~hundreds of recent NVDA filings.
    Idempotency and manifest behaviour are covered by the unit tests.
    """
    cassette = _build_vcr()
    with (
        cassette.use_cassette(CASSETTE_PATH.name),
        edgar.EdgarClient() as client,
    ):
        filings = client.list_recent_filings(CIK, form_types=_REQUESTED_FORMS)
        picked: dict[str, edgar.Filing] = {}
        for filing in filings:
            key = "S-3" if filing.form_type.startswith(_S3_PREFIX) else filing.form_type
            picked.setdefault(key, filing)
            if {"10-K", "8-K", "S-3"} <= picked.keys():
                break
        assert {"10-K", "8-K", "S-3"} <= picked.keys(), (
            f"Recent NVDA filings missing a required form: have {sorted(picked)}"
        )

        for form_key, filing in picked.items():
            path, sha, content = client.fetch_filing(
                filing, raw_root=tmp_path / "raw" / "edgar"
            )
            # AC: data/raw/edgar/{cik}/{year}/{accession}.{ext}
            assert path.parent.parent.parent == tmp_path / "raw" / "edgar"
            assert path.parent.parent.name == filing.cik  # zero-padded
            assert path.parent.name == str(filing.accepted_datetime.year)
            assert path.stem == filing.accession_number
            assert path.exists()
            # SHA-256 is over the persisted bytes.
            assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
            assert path.read_bytes() == content
            # The recorded body should be non-empty.
            assert len(content) > 0, f"empty body for {form_key} {filing.accession_number}"
