"""VCR-recorded integration test for `make_extraction_client`.

Acceptance criterion: "VCR test confirms `cache_creation_input_tokens` /
`cache_read_input_tokens` appear in response metadata after a second call
with the same cached prefix."

The cassette captures two `POST /v1/messages` responses against the same
system prompt (the cached prefix) but different user content (per-doc
variable):

1. First call returns `cache_creation_input_tokens=5000`, `cache_read=0`
   — the cache was just written.
2. Second call returns `cache_creation_input_tokens=0`, `cache_read=5000`
   — the cache was reused.

Replay is offline (no API key needed). To regenerate against the live
endpoint, delete the cassette and re-run with `ANTHROPIC_API_KEY` set;
vcrpy's `record_mode="once"` records on absence and replays otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import vcr
from anthropic import Anthropic

from auto_research.extract.client import make_extraction_client

CASSETTE_PATH = (
    Path(__file__).parent
    / "cassettes"
    / "test_extract_client"
    / "cache_create_then_read.yaml"
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
        # No body match: two requests with identical system prompts but
        # different user content. vcrpy's default "play each interaction
        # at most once" semantic gives us request 1 → cassette entry 1,
        # request 2 → cassette entry 2, in order.
        match_on=["method", "scheme", "host", "port", "path"],
        decode_compressed_response=True,
    )


@pytest.fixture
def anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy key so the SDK constructor is happy on replay."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")


def test_cache_create_then_read_against_recorded_response(
    anthropic_api_key: None,
) -> None:
    """End-to-end: two calls through the wrapper against the same cached
    prefix. The first writes the cache, the second reads it. The wrapper
    sets `cache_control: ephemeral` on the system block by default, so
    the caller doesn't need to opt in.
    """
    cassette = _build_vcr()
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=1000.00,  # not under test; just don't trip the cap
        anthropic_client=Anthropic(),
    )

    with cassette.use_cassette(CASSETTE_PATH.name):
        _first_text, first_usage = client(
            task="dilution_event",
            system_prompt="A long stable system prompt that should be cached.",
            user_content="document A: dilution language sample",
        )
        _second_text, second_usage = client(
            task="dilution_event",
            system_prompt="A long stable system prompt that should be cached.",
            user_content="document B: different dilution language",
        )

    # `UsageDict` lifts the cache counters from `Message.usage` —
    # provider-agnostic dict access on what used to be a typed pydantic
    # field. `cache_*` are `NotRequired` so `usage.get(...)` collapses
    # absent + zero into the same comparison.

    # AC: cache_creation_input_tokens appears on the first call (write).
    assert first_usage["cache_creation_input_tokens"] == 5000
    # First call did not read any cache (nothing was there yet).
    assert first_usage.get("cache_read_input_tokens", 0) == 0

    # AC: cache_read_input_tokens appears on the second call (hit).
    assert second_usage["cache_read_input_tokens"] == 5000
    # Second call did not write to the cache again.
    assert second_usage.get("cache_creation_input_tokens", 0) == 0
