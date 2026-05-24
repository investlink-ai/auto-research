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
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class SpanRecorder:
    """Convenience wrapper around an `InMemorySpanExporter`.

    `by_name(name)` filters; `one(name)` asserts exactly one match.
    Both walk the live exporter on each call so tests can assert span
    state mid-run if they need to.
    """

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter

    def finished_spans(self) -> tuple[ReadableSpan, ...]:
        return tuple(self._exporter.get_finished_spans())

    def by_name(self, name: str) -> tuple[ReadableSpan, ...]:
        return tuple(s for s in self.finished_spans() if s.name == name)

    def one(self, name: str) -> ReadableSpan:
        matches = self.by_name(name)
        assert len(matches) == 1, (
            f"expected exactly one span named {name!r}, "
            f"got {len(matches)}: {[s.name for s in self.finished_spans()]}"
        )
        return matches[0]


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
        # Force-flush any pending spans, then remove the processor so
        # subsequent tests start with a clean slate. `_active_span_processor`
        # on TracerProvider is a `SynchronousMultiSpanProcessor` /
        # `ConcurrentMultiSpanProcessor`; both expose internal lists we
        # would have to mutate, so the cleanest portable approach is to
        # shut down THIS processor and rely on its on_end hook stopping.
        processor.shutdown()
