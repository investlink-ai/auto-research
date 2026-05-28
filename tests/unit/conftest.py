"""Unit-test conftest.

Two session-scoped autouse fixtures warm the lazily-loaded NLP models
before any test runs:

- `_warm_chunking_nlp` — `unstructured`'s spaCy `en_core_web_sm`,
  needed by `parse_filing`'s element classification.
- `_warm_bge_embeddings` — BAAI/bge-small-en-v1.5, the in-process
  embedding fallback used by every unit test that exercises
  `EmbeddingAdapter` with `VOYAGE_API_KEY` absent.

Both follow the same pattern: warm the cache at session start (when
sockets are still real) so subsequent hermetic tests can monkey-patch
the network without triggering a lazy model download. If the
underlying model isn't installed/cached the warmup raises
`RuntimeError` with a clear remediation pointing at `make setup-nlp`.

A third function-scoped autouse fixture redirects
`DEFAULT_QUARANTINE_ROOT` (used by every extraction worker when the
caller omits `quarantine_root`) to a per-test temp dir so a regression
that quarantines on a "happy path" test does NOT leak a record into the
working-tree `data/quarantine/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, ToolUseBlock, Usage

_StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


def make_tool_use_message(
    *,
    tool_input: dict[str, Any],
    tool_name: str = "record_extraction",
    model: str = "claude-haiku-4-5-20251001",
    stop_reason: _StopReason = "tool_use",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> Message:
    """Build a Message whose content is a single ToolUseBlock.

    Mirrors the on-wire shape every extraction response carries: one
    tool_use block whose `.input` is the parsed dict — no text +
    json.loads round-trip. `stop_reason` defaults to `tool_use` to
    match the production shape; pass `max_tokens` to exercise the
    truncation guard, or `refusal` for the refusal-collapse path.
    """
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[
            ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name=tool_name,
                input=tool_input,
            )
        ],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


def make_fake_anthropic_client(
    tool_input: Any,
    *,
    tool_name: str = "record_extraction",
    stop_reason: _StopReason = "tool_use",
) -> anthropic.Anthropic:
    """Wrap `make_tool_use_message` in a MagicMock-backed SDK stub.

    Workers under test inject this via `anthropic_client=...`. Returns
    a fake whose `messages.create(...)` yields the configured
    tool_use response on every call. For sequence-of-responses use
    `make_fake_anthropic_client_sequence`.
    """
    fake = MagicMock()
    fake.messages.create.return_value = make_tool_use_message(
        tool_input=tool_input, tool_name=tool_name, stop_reason=stop_reason
    )
    return cast(anthropic.Anthropic, fake)


def make_fake_anthropic_client_sequence(
    tool_inputs: list[Any],
    *,
    tool_name: str = "record_extraction",
) -> anthropic.Anthropic:
    """Fake whose `messages.create(...)` returns a different tool_use
    response per call (used by multi-call workers like the 10-K RAG
    path and the post-PR-B transcript binary split).
    """
    fake = MagicMock()
    fake.messages.create.side_effect = [
        make_tool_use_message(tool_input=t, tool_name=tool_name) for t in tool_inputs
    ]
    return cast(anthropic.Anthropic, fake)


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


@pytest.fixture(scope="session", autouse=True)
def _warm_bge_embeddings() -> None:
    """Warm `auto_research.extract.embeddings._ensure_bge_warmup` once per session.

    Without this, the first hermetic unit test exercising
    `EmbeddingAdapter` in BGE mode would trigger a `SentenceTransformer`
    HuggingFace download under whatever socket monkey-patch the test
    installed. Loading at session start (before any socket patching)
    lands the model in HF cache during a real-network window; the
    in-process singleton then serves every adapter for the rest of the
    session at zero further network cost.
    """
    from auto_research.extract.embeddings import _ensure_bge_warmup

    _ensure_bge_warmup()


def _is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.fixture(scope="session", autouse=True)
def _warm_qwen3_mlx_embeddings() -> None:
    """Warm the Qwen3-Embedding-0.6B MLX model once per session on
    Apple Silicon.

    Same lazy-load-then-socket-monkey-patch concern as BGE: the first
    real-inference Mac unit test would otherwise pull ~600 MB of
    Qwen3-0.6B weights from HuggingFace under a socket-blocked
    environment. Pre-warming at session start lands the weights in HF
    cache via `make setup-mlx`; the in-process singleton in
    `_QWEN3_MODELS` then serves every Qwen3-MLX adapter for the
    remainder of the session at zero further network cost.

    No-op on non-Apple-Silicon hosts — the `mlx-embeddings` extra
    isn't installed there and `_ensure_qwen3_warmup` would raise with
    the platform-check remediation; the Mac-only tests using this
    backend are individually marked `skipif` on the same predicate.
    """
    if not _is_apple_silicon():
        return
    from auto_research.extract.embeddings import _ensure_qwen3_warmup

    try:
        _ensure_qwen3_warmup("Qwen3-Embedding-0.6B")
    except RuntimeError as exc:
        # Only swallow the "mlx-embeddings extra not installed" case —
        # that's a legitimate Mac-dev environment where unit tests
        # should still run (qwen3-mlx tests will skip individually
        # via the skipif gate). All other exceptions (HF cache miss,
        # platform-check polarity bug, repo-name typo, API drift)
        # must propagate so the session start fails loudly with the
        # actionable remediation, per the explicit-config-loud rule.
        if "uv sync --extra mlx" not in str(exc):
            raise


@pytest.fixture(scope="session", autouse=True)
def _warm_qwen3_reranker() -> None:
    """Warm the Qwen3-Reranker-0.6B model once per session on any host.

    Same lazy-load-then-socket-monkey-patch concern as the BGE and
    Qwen3-Embedding warmups: the first hermetic reranker unit test
    that triggers a real load would otherwise pull ~1.2 GB of
    Qwen3-Reranker-0.6B weights from HuggingFace under a socket-blocked
    environment. Pre-warming at session start lands the weights in HF
    cache via `make setup-reranker`.

    Cross-platform: the reranker's `ci-cpu` tier runs on Linux CI. On
    Apple Silicon, the same warmup populates the `(0.6B, cpu)` cache
    entry; the `dev` tier's `(0.6B, mps)` entry is loaded lazily by
    the tests that actually exercise MPS.

    Only swallows the specific "transformers / torch not importable"
    remediation error — all other failures (cache miss, repo rename,
    API drift) propagate so the session start fails loudly with the
    actionable remediation, per the explicit-config-loud rule.
    """
    from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup

    try:
        _ensure_qwen3_reranker_warmup("Qwen3-Reranker-0.6B", "cpu", "fp32")
    except RuntimeError as exc:
        # Specific phrase from `_ensure_qwen3_reranker_warmup`'s
        # ImportError branch. Substring matching on `uv sync` alone
        # would also swallow unrelated errors mentioning those words.
        if "`transformers` / `torch` could not be imported" not in str(exc):
            raise


@pytest.fixture(autouse=True)
def _hermetic_default_quarantine_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect every worker's `DEFAULT_QUARANTINE_ROOT` to tmp_path.

    The default is `Path("data/quarantine")` — a relative path that
    resolves to the repo's working tree. A happy-path test that omits
    `quarantine_root=` and then quarantines (because of a regression)
    would silently write into the live repo. Each worker imports the
    name into its own module namespace, so we monkeypatch each one
    individually rather than only `guardrails.DEFAULT_QUARANTINE_ROOT`.

    Tests that pass `quarantine_root` explicitly are unaffected — the
    override only changes what callers receive when they omit the arg.
    """
    target = tmp_path / "default_quarantine"
    from auto_research.extract import guardrails
    from auto_research.extract.workers import eight_k, s_filings, ten_k, transcript

    monkeypatch.setattr(guardrails, "DEFAULT_QUARANTINE_ROOT", target)
    for module in (eight_k, s_filings, ten_k, transcript):
        if hasattr(module, "DEFAULT_QUARANTINE_ROOT"):
            monkeypatch.setattr(module, "DEFAULT_QUARANTINE_ROOT", target)
