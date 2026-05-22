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
