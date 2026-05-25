"""Shared OTel test helpers.

`SpanRecorder` wraps an in-memory OTel exporter. The corresponding
fixture lives in `tests/conftest.py`; tests import this class only
for type annotations on the fixture parameter.

Separated from conftest.py because mypy treats `tests/conftest.py`
and `tests/integration/conftest.py` as duplicate top-level modules —
conftest.py is excluded from strict typecheck (see pyproject.toml),
so a typed helper has to live alongside it under a unique module
name.
"""

from __future__ import annotations

from collections.abc import Mapping

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.types import AttributeValue


class SpanRecorder:
    """Convenience wrapper around an `InMemorySpanExporter`.

    `by_name(name)` filters; `one(name)` asserts exactly one match;
    `attrs(name)` returns the (non-None) attribute mapping of the one
    span named `name` — saves callers from repeating
    `assert span.attributes is not None` at every attribute access
    (the OTel SDK types `attributes` as `Mapping | None`).
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

    def attrs(self, name: str) -> Mapping[str, AttributeValue]:
        """Return the attribute mapping of the one span named `name`.

        Asserts the mapping is non-None — OTel SDK types `attributes`
        as `Mapping | None`, but a span produced by `start_as_current_span`
        always has at least an empty dict in practice.
        """
        attributes = self.one(name).attributes
        assert attributes is not None, (
            f"span {name!r} has no attributes (OTel sdk returned None)"
        )
        return attributes


__all__ = ["SpanRecorder"]
