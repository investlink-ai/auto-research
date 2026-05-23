"""End-to-end: Anthropic call → trace landed in Langfuse.

Integration smoke for `init_telemetry()`. The `integration` marker is
auto-applied by `tests/integration/conftest.py`; this whole folder is
gated by the Langfuse-reachable session fixture, so missing-Docker
runs skip cleanly instead of failing.

Requires (in addition to the conftest-level Langfuse check):
  - `.env` with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    OTEL_EXPORTER_OTLP_ENDPOINT, ANTHROPIC_API_KEY

Verifies:
  1. Anthropic SDK call returns with token counts populated
  2. OTel batch processor flushes within timeout
  3. At least one trace lands in Langfuse within the test window
"""

from __future__ import annotations

import datetime
import os
import time

import pytest

from auto_research.telemetry import init_telemetry


def test_anthropic_call_under_telemetry() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping integration smoke")
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        pytest.skip("LANGFUSE_PUBLIC_KEY not set; skipping integration smoke")

    init_telemetry()

    window_start = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=5)

    import anthropic
    from anthropic.types import TextBlock

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with just: ok"}],
    )

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    first_block = response.content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text.strip().lower().startswith("ok")

    # Force-flush spans so we can query Langfuse synchronously.
    from opentelemetry import trace as otel_trace

    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        flushed = provider.force_flush(timeout_millis=5000)
        assert flushed, "OTel span flush timed out (5s)"

    # Poll Langfuse for the trace landing (ingestion is async).
    # Cold-Docker ingestion can take 10-15s, so budget 30s with early-exit.
    from langfuse import Langfuse

    lf = Langfuse()
    found = False
    for _ in range(15):  # up to ~30s
        time.sleep(2)
        traces = lf.fetch_traces(from_timestamp=window_start, limit=20)
        if traces.data:
            found = True
            break

    assert found, "No traces landed in Langfuse within 30s of the Anthropic call"
