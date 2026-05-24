"""Shared pytest fixtures for the auto-research test suite.

`span_recorder` installs an in-memory OTel tracer provider for the
duration of one test and exposes the recorded spans. Used by unit
tests that assert on manual instrumentation without needing a live
OTLP exporter — `tests/integration/test_telemetry_export.py` covers
the live-Langfuse path.

`init_telemetry()` is NOT called inside the fixture: the fixture
provides its own provider, and Traceloop's provider would race with
it if both were installed in the same process.
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
    Both walk the live exporter on each call so a test can assert
    span state mid-run if it needs to (no snapshotting).
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


@pytest.fixture
def span_recorder() -> Iterator[SpanRecorder]:
    """Install an in-memory tracer provider for one test; restore on teardown.

    Restores `trace._TRACER_PROVIDER` via direct attribute set because
    `trace.set_tracer_provider` refuses to replace an already-set
    provider and the public API has no "unset" call. Touching the
    internal is the lesser evil compared to leaking the fixture's
    provider into subsequent tests.
    """
    previous = trace.get_tracer_provider()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    try:
        yield SpanRecorder(exporter)
    finally:
        provider.shutdown()
        trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]
