"""Unit tests for telemetry init — error paths + idempotency state.

End-to-end Anthropic → Langfuse verification lives in
`tests/integration/test_telemetry_export.py`.
"""

from __future__ import annotations

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


def test_try_init_telemetry_returns_false_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing env -> warn once to stderr, return False, don't raise."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from auto_research import telemetry as t

    monkeypatch.setattr(t, "_INITIALIZED", False)
    monkeypatch.setattr(t, "_TRY_INIT_WARNED", False)

    assert t.try_init_telemetry() is False
    captured = capsys.readouterr()
    assert "telemetry" in captured.err.lower()

    # Idempotent warning — second call does not double-print.
    assert t.try_init_telemetry() is False
    captured2 = capsys.readouterr()
    assert captured2.err == ""


def test_try_init_telemetry_returns_true_when_already_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_research import telemetry as t

    monkeypatch.setattr(t, "_INITIALIZED", True)
    assert t.try_init_telemetry() is True


def test_span_recorder_fixture_captures_spans(span_recorder) -> None:  # type: ignore[no-untyped-def]
    """The shared in-memory tracer fixture captures finished spans."""
    from opentelemetry import trace

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("smoke") as span:
        span.set_attribute("k", "v")

    one = span_recorder.one("smoke")
    assert one.attributes["k"] == "v"
