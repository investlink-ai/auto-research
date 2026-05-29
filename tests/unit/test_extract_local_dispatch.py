"""Unit tests for the local-vs-Anthropic client dispatch.

What's pinned here:

- `_get_or_build_client(worker, task, anthropic_client=None)` chooses
  the right provider wrapper based on the routed model id:
  `local/...` ⇒ `get_or_build_local_client`; everything else ⇒
  `make_extraction_client`.
- The `_LOCAL_QWEN_*` constants exist and carry the documented string
  shape (`local/<server-native-id>`).
- The local singleton table is keyed on `(worker, model_id)` so a
  routing flip from 9B to 27B builds a fresh client rather than reusing
  the stale 9B singleton.

Today no `_ROUTING` row resolves to a `local/*` model id; the
acceptance criteria on the issue are infra-only. These tests use a
monkeypatched `route_model` to simulate a flipped routing entry —
exercising the dispatch path without polluting the production
routing table.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import anthropic
import pytest

from auto_research import _models
from auto_research.extract import openai_compat_client as local_module
from auto_research.extract.workers import _common


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    """Reset both the Anthropic and the local singleton tables between
    tests so dispatch decisions are observable per-test rather than
    bleeding from a prior case's caching."""
    _common._CLIENTS.clear()
    local_module._reset_local_clients_for_testing()


# --- constants --------------------------------------------------------------


def test_local_qwen_constants_are_local_prefixed() -> None:
    """Every `_LOCAL_QWEN_*` carries the `local/` dispatch hint that
    `_get_or_build_client` reads. A regression that renames the
    prefix without updating the dispatch would silently send local
    routes to the Anthropic SDK."""
    assert _models._LOCAL_QWEN_4B.startswith("local/")
    assert _models._LOCAL_QWEN_27B_DENSE.startswith("local/")
    assert _models._LOCAL_QWEN_35B_MOE.startswith("local/")


def test_local_qwen_constants_have_documented_values() -> None:
    """The cost-model doc §10.5 "Locked stack" names the smoke-tested
    vllm-mlx checkpoint for the MoE constant and upstream Qwen HF
    paths for the placeholder dense / pocket constants. Pin them so
    a drift between the constants and the documented launch command
    surfaces here. Changing any of these values means revisiting the
    smoke-test artifacts in the doc."""
    assert _models._LOCAL_QWEN_4B == "local/Qwen/Qwen3.6-4B-Instruct"
    assert _models._LOCAL_QWEN_27B_DENSE == "local/Qwen/Qwen3.6-27B-Instruct"
    assert (
        _models._LOCAL_QWEN_35B_MOE == "local/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
    )


# --- dispatch --------------------------------------------------------------


def test_dispatch_picks_anthropic_for_non_local_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An Anthropic model id (no `local/` prefix) routes through
    `make_extraction_client` — the production case for every worker
    today. Pin by observing that the Anthropic factory was called
    and the local factory was NOT."""
    anthropic_sentinel = MagicMock(name="anthropic_client_sentinel")
    local_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "auto_research.extract.workers._common.make_extraction_client",
        lambda **kw: anthropic_sentinel,
    )
    monkeypatch.setattr(
        "auto_research.extract.workers._common.get_or_build_local_client",
        lambda worker, model_id, **kw: local_calls.append((worker, model_id)),
    )
    client = _common._get_or_build_client(
        "s_filings", "dilution_event", anthropic_client=None
    )
    assert client is anthropic_sentinel
    assert local_calls == []


def test_dispatch_picks_local_for_local_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `route_model` returns a `local/*` id, dispatch goes to
    `get_or_build_local_client` — the wrapper that talks to Ollama /
    vLLM / MLX-server."""
    # Patch the `route_model` symbol imported into _common — the
    # production routing table stays untouched.
    monkeypatch.setattr(
        "auto_research.extract.workers._common.route_model",
        lambda _w, _t: _models._LOCAL_QWEN_35B_MOE,
    )
    sentinel = MagicMock(name="local_client_sentinel")
    monkeypatch.setattr(
        "auto_research.extract.workers._common.get_or_build_local_client",
        lambda worker, model_id, **kw: sentinel,
    )
    client = _common._get_or_build_client(
        "contextual_chunking", "contextual_chunk", anthropic_client=None
    )
    assert client is sentinel


def test_dispatch_local_ignores_injected_anthropic_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`anthropic_client=` is the Anthropic path's test-injection
    escape hatch; on the local branch it must be silently ignored so
    a worker that always threads `anthropic_client` through doesn't
    accidentally short-circuit the local dispatch."""
    monkeypatch.setattr(
        "auto_research.extract.workers._common.route_model",
        lambda _w, _t: _models._LOCAL_QWEN_35B_MOE,
    )
    sentinel = MagicMock(name="local_client_sentinel")
    monkeypatch.setattr(
        "auto_research.extract.workers._common.get_or_build_local_client",
        lambda worker, model_id, **kw: sentinel,
    )
    fake_sdk = cast(anthropic.Anthropic, MagicMock())
    client = _common._get_or_build_client(
        "contextual_chunking", "contextual_chunk", anthropic_client=fake_sdk
    )
    assert client is sentinel


def test_local_singleton_keyed_by_worker_and_model_id() -> None:
    """`get_or_build_local_client` keys on `(worker, model_id)` so a
    routing-table flip from 9B to 27B inside the same worker builds
    a fresh client rather than reusing the stale 9B singleton — the
    routing change wouldn't otherwise take effect mid-process."""
    a = local_module.get_or_build_local_client(
        "contextual_chunking", _models._LOCAL_QWEN_35B_MOE
    )
    b = local_module.get_or_build_local_client(
        "contextual_chunking", _models._LOCAL_QWEN_35B_MOE
    )
    assert a is b, "same (worker, model_id) must return the same instance"

    c = local_module.get_or_build_local_client(
        "contextual_chunking", _models._LOCAL_QWEN_27B_DENSE
    )
    assert c is not a, (
        "different model_id under the same worker must build a fresh client"
    )

    d = local_module.get_or_build_local_client(
        "some_other_worker", _models._LOCAL_QWEN_35B_MOE
    )
    assert d is not a, "different worker under the same model_id must isolate"


_ALLOWED_LOCAL_ROWS: frozenset[tuple[str, str]] = frozenset(
    {
        # 10-K narrative-disclosure signals XBRL cannot give. Bounded
        # output shape (Claim | None or list[Claim]), per-field cache
        # keys, smoke-tested via tests/live/test_ten_k_local_qwen_smoke.py
        # against the locked Qwen 35B-MoE stack (cost-model doc §10.5).
        ("ten_k", "going_concern"),
        ("ten_k", "icfr_material_weaknesses"),
        ("ten_k", "critical_accounting_estimate_changes"),
    }
)


def test_only_allowed_routes_flipped_to_local() -> None:
    """Acceptance-criteria pin: the only `_ROUTING` rows allowed to
    resolve to a `local/*` model id are those enumerated in
    `_ALLOWED_LOCAL_ROWS`. New flips must extend the allowlist AND cite
    the eval / smoke-test evidence in the PR body — the assertion is
    the gate the author has to look at."""
    for (worker, task), model_id in _models._ROUTING.items():
        is_local = model_id.startswith("local/")
        is_allowed = (worker, task) in _ALLOWED_LOCAL_ROWS
        if is_local:
            assert is_allowed, (
                f"({worker!r}, {task!r}) routes to {model_id!r} but is not "
                f"in _ALLOWED_LOCAL_ROWS — local routes are eval-gated; "
                "extend the allowlist and cite the eval delta in the PR body"
            )
        else:
            assert not is_allowed, (
                f"({worker!r}, {task!r}) is in _ALLOWED_LOCAL_ROWS but routes "
                f"to {model_id!r}, not a local/* model — remove the row from "
                "_ALLOWED_LOCAL_ROWS or restore the local routing"
            )
