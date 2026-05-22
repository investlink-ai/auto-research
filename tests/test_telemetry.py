"""Telemetry smoke tests.

Unit tests verify the error path (no integration cost). The integration
test makes a real Anthropic call and pushes a trace to Langfuse; it is
marked `integration` and excluded from CI. Run locally via `make eval`
after starting Langfuse (`docker compose up -d`) and configuring `.env`.
"""

from __future__ import annotations

import os

import pytest

from auto_research.telemetry import (
    TelemetryNotConfiguredError,
    init_telemetry,
    is_initialized,
)


def test_init_telemetry_raises_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    # Reset module-level state in case a prior test (or import order) set it.
    from auto_research import telemetry as t

    monkeypatch.setattr(t, "_INITIALIZED", False)

    with pytest.raises(TelemetryNotConfiguredError) as exc_info:
        init_telemetry()

    msg = str(exc_info.value)
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in msg
    assert "LANGFUSE_PUBLIC_KEY" in msg
    assert "LANGFUSE_SECRET_KEY" in msg


def test_is_initialized_starts_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_research import telemetry as t

    monkeypatch.setattr(t, "_INITIALIZED", False)
    assert is_initialized() is False


@pytest.mark.integration
def test_anthropic_call_under_telemetry() -> None:
    """Real Anthropic call after init_telemetry; verify visually in Langfuse UI.

    Requires:
      - `docker compose up -d` (Langfuse + Postgres reachable at :3000)
      - `.env` with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
        OTEL_EXPORTER_OTLP_ENDPOINT, ANTHROPIC_API_KEY

    Asserts the SDK call succeeds and reports token counts. The trace
    landing in Langfuse is verified visually at http://localhost:3000.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping integration smoke")

    init_telemetry()

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with just: ok"}],
    )

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.content[0].text.strip().lower().startswith("ok")
