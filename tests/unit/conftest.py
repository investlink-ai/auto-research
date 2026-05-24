"""Unit-test conftest.

Session-scoped autouse fixture warms `unstructured`'s spaCy model
before any test runs. Without this, `test_chunking.py`'s no-network
test would monkey-patch `socket` before `parse_filing` warmed the NLP
cache and trigger a `RuntimeError` (spaCy model not loadable under
patched sockets). Warming once at session start matches production
behavior — the import-time cost is paid once per process.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _warm_chunking_nlp() -> None:
    """Warm `auto_research.extract.chunking._ensure_nlp_warmup` once per session.

    Safe no-op if the module isn't imported by any test in the session
    (the import itself is cheap; the spaCy load is deferred to the
    warmup function). If `en_core_web_sm` is missing the warmup raises
    `RuntimeError` with a clear remediation message — better than a
    cryptic socket error from inside `partition_html`.
    """
    from auto_research.extract.chunking import _ensure_nlp_warmup

    _ensure_nlp_warmup()
