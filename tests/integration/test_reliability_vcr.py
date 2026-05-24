"""VCR-recorded integration test for `@cost_cap`.

`cost_cap` reads token counts from real Anthropic response metadata
(never a mock), and the VCR cassette pins a real-shape response.

The committed cassette captures one real-shape `POST /v1/messages` response
with a populated `usage` block (`input_tokens=1.5M`, `output_tokens=0.9M`).
With Sonnet 4.6 list pricing ($3 / $15 per MTok) that single call totals
$18, far above the $5 cap — so the *second* call must raise
`CostCapExceeded` before the inner SDK call ever runs.

The cassette is replayed without an API key. To regenerate against a live
endpoint, delete the file and re-run with `ANTHROPIC_API_KEY` set; vcrpy's
`record_mode="once"` records on absence and replays otherwise. The API key
header is filtered out at record time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import vcr
from anthropic import Anthropic
from anthropic.types import Message

from auto_research.agents.reliability import CostCapExceeded, cost_cap

CASSETTE_PATH = (
    Path(__file__).parent
    / "cassettes"
    / "test_reliability"
    / "anthropic_messages_one_call.yaml"
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
    """Set a dummy key so the SDK constructor is happy.

    On replay no real auth header reaches the network; on record the user's
    real key is read from the environment (we filter it out of the cassette
    before writing).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


def test_cost_cap_reads_usage_from_real_anthropic_response(
    anthropic_api_key: None,
) -> None:
    """End-to-end: the wrapped function calls the SDK; `cost_cap` reads
    `response.usage.input_tokens` / `output_tokens` off the returned
    `Message` object. The second call short-circuits with `CostCapExceeded`
    because the recorded response (1.5M in, 0.9M out on Sonnet 4.6) blows
    the $5 cap on the first call alone.
    """
    cassette = _build_vcr()
    inner_calls = []

    @cost_cap(usd=5.00)
    def llm_call(client: Anthropic) -> Message:
        inner_calls.append(1)
        return client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )

    with cassette.use_cassette(CASSETTE_PATH.name):
        client = Anthropic()
        response = llm_call(client)
        # Real Pydantic Message — not a Mock — with populated usage:
        assert isinstance(response, Message)
        assert response.usage.input_tokens == 1_500_000
        assert response.usage.output_tokens == 900_000

        with pytest.raises(CostCapExceeded):
            llm_call(client)

    # Second call short-circuited before the SDK was invoked.
    assert len(inner_calls) == 1
