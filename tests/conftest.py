"""Shared pytest fixtures for the auto-research test suite.

`span_recorder` makes every manual span recorded by production code
visible to a unit test via an in-memory exporter — no live OTLP
required. `tests/integration/test_telemetry_export.py` covers the
real-Langfuse path.

Design note: OTel's `ProxyTracer` caches the real tracer on first use
and never refreshes. If each test installed its own `TracerProvider`
and shut it down on teardown, module-level `_tracer = trace.get_tracer
(__name__)` references in production code would keep pointing at the
already-shut-down provider — subsequent tests would silently record
zero spans. We dodge this by installing a single `TracerProvider`
once at session scope and rotating an `InMemorySpanExporter` per
test. The provider stays stable; the recorder doesn't.

The `SpanRecorder` class lives in `tests/_otel_helpers.py` so test
files can import it for type annotations — see that module's
docstring for why this conftest is excluded from mypy.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tests._otel_helpers import SpanRecorder


@pytest.fixture(scope="session")
def _session_tracer_provider() -> Iterator[TracerProvider]:
    """One TracerProvider for the whole test session.

    Installs into the global `trace._TRACER_PROVIDER` slot via direct
    attribute assignment (the OTel public API refuses to replace an
    already-set provider and has no "unset" call). Tracers acquired
    by production modules at import time bind to this provider and
    stay valid for the entire session.
    """
    provider = TracerProvider()
    previous = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    try:
        yield provider
    finally:
        provider.shutdown()
        trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]


@pytest.fixture
def span_recorder(
    _session_tracer_provider: TracerProvider,
) -> Iterator[SpanRecorder]:
    """Per-test in-memory span exporter, registered as a SpanProcessor
    on the session-scoped TracerProvider. Removed on teardown so spans
    from a later test don't leak into an earlier recorder."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    _session_tracer_provider.add_span_processor(processor)
    try:
        yield SpanRecorder(exporter)
    finally:
        # Shut down THIS processor so its on_end hook stops firing.
        # `_active_span_processor` on TracerProvider is a multi-
        # processor whose internal list isn't a public API; shutting
        # down the leaf processor is the portable equivalent.
        processor.shutdown()
