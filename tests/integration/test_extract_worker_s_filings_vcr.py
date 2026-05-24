"""VCR-recorded end-to-end test for the S-1/S-3 worker (Issue #11 AC).

Pattern: same as `test_extract_client_vcr.py` — record once against the
live Anthropic endpoint with `ANTHROPIC_API_KEY` set, replay offline
thereafter. The cassette redacts the API key header before commit.

The fixture (`tests/fixtures/s_filings/sample_s3_excerpt.txt`) is a
hand-curated S-3 excerpt mimicking real SEC language (cover summary +
use-of-proceeds + dilution + plan of distribution sections). The
production path reads real EDGAR-ingested filings out of `data/raw/`;
the EDGAR ingest itself is covered by `test_edgar_vcr.py`. This test's
job is to prove the worker handles a realistic S-3 end-to-end through
prompt → Anthropic → schema parse → citation grounding → cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import vcr
from anthropic import Anthropic

from auto_research.extract.workers.s_filings import extract_s_filing

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "s_filings"
    / "sample_s3_excerpt.txt"
)
CASSETTE_PATH = (
    Path(__file__).parent
    / "cassettes"
    / "test_extract_worker_s_filings"
    / "real_s3.yaml"
)


def _build_vcr() -> vcr.VCR:
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_PATH.parent),
        record_mode="once",
        filter_headers=[
            ("x-api-key", "REDACTED"),
            ("authorization", "REDACTED"),
            ("anthropic-organization-id", "REDACTED"),
            ("User-Agent", "auto-research-test"),
        ],
        match_on=["method", "scheme", "host", "port", "path"],
        decode_compressed_response=True,
    )


@pytest.fixture
def anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy key only when the cassette already exists (replay mode).

    Recording — when the cassette is absent — needs the real
    `ANTHROPIC_API_KEY` from the environment to reach the live endpoint;
    overriding it here would force a 401 during the recording session.
    On replay the dummy is fine because vcrpy intercepts the request
    before it hits the network.
    """
    if CASSETTE_PATH.exists():
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


def test_real_s3_passes_citation_grounding(
    tmp_path: Path,
    anthropic_api_key: None,
) -> None:
    raw = FIXTURE.read_text()
    cassette = _build_vcr()
    with cassette.use_cassette(CASSETTE_PATH.name):
        out = extract_s_filing(
            raw_doc=raw,
            doc_id="vcr-real-s3",
            cache_root=tmp_path,
            quarantine_root=tmp_path / "quarantine",
            anthropic_client=Anthropic(),
        )
    assert out is not None, "real S-3 must pass citation grounding"
    assert out.form_type in ("S-1", "S-3")

    # INV-2 contract literal check: every Citation's source_span indexes
    # into `raw` (the on-disk file), and the resulting slice equals
    # source_quote. The earlier worker design (whitespace-normalized
    # text) made this assertion silently false; the rewrite restores
    # raw-coordinate spans so this works end-to-end on a multi-line filing.
    from auto_research.extract.guardrails import _walk_citations

    for path, citation in _walk_citations(out):
        start, end = citation.source_span
        assert raw[start:end] == citation.source_quote, (
            f"citation at {path} span {(start, end)} does not align with raw_doc"
        )
