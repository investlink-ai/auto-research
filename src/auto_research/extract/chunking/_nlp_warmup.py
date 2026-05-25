"""One-shot spaCy warmup so `unstructured.partition_html` doesn't lazy-load
its NLP model on first parse.

`unstructured.partition.html.partition_html` calls into
`unstructured.nlp.tokenize.sent_tokenize` for element classification.
On first use this lazily downloads `en_core_web_sm` from GitHub — a
network call that breaks hermetic tests and silently surprises fresh
deployments. We warm the cache on the first `parse_filing` call so
production behavior is deterministic.
"""

from __future__ import annotations

# Module-level cache: once a successful warmup completes, subsequent
# `_ensure_nlp_warmup()` calls are no-ops. The flag is process-local so
# pytest-xdist workers each warm once, but a single test run does not
# re-pay the spaCy-load cost on every parse_filing call.
_NLP_WARMED: bool = False


def _ensure_nlp_warmup() -> None:
    """Warm `unstructured`'s spaCy model on first use.

    `unstructured.partition.html.partition_html` calls into
    `unstructured.nlp.tokenize.sent_tokenize` for element classification.
    On first use this lazily downloads `en_core_web_sm` from GitHub —
    a network call that breaks hermetic tests and silently surprises
    fresh deployments.

    We warm the cache on the first `parse_filing` call so:
      1. Production behavior is deterministic — within a process, the
         first parse pays the warmup cost; subsequent parses don't.
      2. The hermetic `test_parse_filing_makes_no_network_calls` test
         passes — the warmup runs once via a conftest autouse fixture
         BEFORE the socket monkey-patch, so by the time `parse_filing`
         is called under the patch, the model is already in memory.
      3. Module import is fast — no eager spaCy load at import time, so
         tooling that imports the module (mypy via inference,
         pytest-xdist worker bootstrap, IDE plugins) does not require
         the spaCy model to be installed just to read the module.

    Idempotent via the `_NLP_WARMED` flag. A missing model raises
    `RuntimeError` with a clear remediation path; no silent network
    downloads.
    """
    global _NLP_WARMED
    if _NLP_WARMED:
        return
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required by "
            "unstructured.partition_html for element classification but "
            "is not installed. Install with:\n"
            "    uv run python -m spacy download en_core_web_sm\n"
            "Or run `make setup-nlp` from the repo root."
        ) from exc
    _NLP_WARMED = True
