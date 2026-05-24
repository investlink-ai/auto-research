"""VCR-recorded integration test for `make_batch_client` (Issue #48 AC).

Acceptance criterion: "VCR integration test covers: (a) submit batch,
(b) poll status, (c) fetch results — all responses carry
`service_tier=batch`."

The cassette captures four HTTP interactions against the same batch:

1. `POST /v1/messages/batches` → returns a `MessageBatch` with
   `processing_status="in_progress"`.
2. `GET /v1/messages/batches/{id}` → from the test's explicit `poll()`
   call; returns `processing_status="ended"` with a `results_url`.
3. `GET /v1/messages/batches/{id}` → internal retrieve inside the SDK's
   `messages.batches.results()` (it fetches the batch first to read
   `results_url`, *then* GETs that URL). Same response shape as #2.
4. `GET /v1/messages/batches/{id}/results` → JSONL stream of two
   `MessageBatchIndividualResponse` lines; both messages carry
   `usage.service_tier="batch"` so `_pricing.usd_for_message` applies
   the documented 50% discount.

Replay is offline (no API key needed). To regenerate against the live
endpoint, delete the cassette and re-run with `ANTHROPIC_API_KEY` set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import vcr
from anthropic import Anthropic
from anthropic.types import Message

from auto_research.extract.batch_client import BatchRequest, make_batch_client

CASSETTE_PATH = (
    Path(__file__).parent
    / "cassettes"
    / "test_batch_client"
    / "submit_poll_results.yaml"
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
    """Set a dummy key so the SDK constructor is happy on replay."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


def test_submit_poll_results_against_recorded_responses(
    anthropic_api_key: None,
) -> None:
    """End-to-end: submit a batch, poll until ended, fetch results.

    The cassette serves a single-cycle poll (one in-progress response
    skipped, one ended response). All resulting messages carry
    `service_tier="batch"` per the cassette; `usd_for_message` then
    applies the 50% discount automatically.
    """
    cassette = _build_vcr()
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=Anthropic(),
    )

    with cassette.use_cassette(CASSETTE_PATH.name):
        handle = client.submit(
            task="dilution_event",
            requests=[
                BatchRequest(
                    custom_id="acc-001",
                    system_prompt="extract dilution events",
                    user_content="filing 1",
                ),
                BatchRequest(
                    custom_id="acc-002",
                    system_prompt="extract dilution events",
                    user_content="filing 2",
                ),
            ],
        )
        # (a) submit returned a handle with the recorded batch id.
        assert handle.batch_id == "msgbatch_01TestBatchSubmit"
        # (b) poll picks up the "ended" response on the first retrieve.
        batch = client.poll(handle)
        assert batch.processing_status == "ended"
        # (c) results: succeeded entries are real Pydantic `Message`s with
        # batch-tier usage metadata, so the 50% discount applies.
        results = client.results(handle)

    assert set(results.succeeded.keys()) == {"acc-001", "acc-002"}
    assert not results.failed
    msg_a = results.succeeded["acc-001"]
    msg_b = results.succeeded["acc-002"]
    assert isinstance(msg_a, Message)
    assert isinstance(msg_b, Message)
    # Both messages carry the batch tier — the discount-pinning regression
    # guard the AC explicitly calls for.
    assert msg_a.usage.service_tier == "batch"
    assert msg_b.usage.service_tier == "batch"
    # Running USD reflects the discounted cost. Haiku 4.5 = $1 + $5 per MTok.
    # acc-001: (120 in + 40 out) → (0.00012 * 1 + 0.00004 * 5) * 0.5 = 0.00016
    # acc-002: (130 in + 50 out) → (0.00013 * 1 + 0.00005 * 5) * 0.5 = 0.00019
    # Total ≈ 0.00035
    assert client.running_usd() == pytest.approx(0.00035, rel=1e-3)
