"""Live-smoke test infrastructure.

Tests under `tests/live/` exercise the **real** external service (SEC
EDGAR, FMP, Anthropic, …) over real network. They're the canary layer
of the testing pyramid: unit + VCR-cassette tests catch our logic;
live smokes catch upstream API drift, expired auth, broken redirects,
and other failure modes a recording can't.

Cadence: nightly cron via `.github/workflows/live-smoke.yml`, plus
`make live-smoke` for local on-demand runs. Excluded from per-PR CI
because they're slow, flaky in aggregate, and may consume rate-limit
or paid-API budgets we don't want to spend on every push.

This conftest:
- Auto-applies `@pytest.mark.live` to every test file in this folder.
- **Opt-in by default**: live tests are skipped unless the run
  explicitly selects them via `-m live` (the marker expression the
  Makefile's `live-smoke` target passes). Without this, a contributor
  who has `SEC_USER_AGENT` exported from unrelated work could fire
  real SEC traffic during a routine `pytest tests/` run.
- Skips collection when the test's required credential env vars
  aren't set (declared via `live_requires_env` on the test or module).
  This makes `make live-smoke` safe to run with partial credentials —
  e.g., a contributor with `SEC_USER_AGENT` set but no `FMP_API_KEY`
  will run the EDGAR smoke and skip the FMP smoke, instead of failing
  the whole suite.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest


def _is_live_item(item: pytest.Item) -> bool:
    return "tests/live" in str(item.path).replace("\\", "/")


def _required_env(item: pytest.Item) -> Iterable[str]:
    """Required env vars declared at the module level via `live_requires_env`.

    Convention: a module declares `live_requires_env = ("SEC_USER_AGENT",)`
    at top level; every test in that module skips cleanly when any of
    those vars are missing.
    """
    module = getattr(item, "module", None)
    if module is None:
        return ()
    raw = getattr(module, "live_requires_env", ())
    return tuple(raw)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    live_items = [item for item in items if _is_live_item(item)]
    if not live_items:
        return

    markexpr = config.getoption("markexpr", default="") or ""
    explicit_opt_in = "live" in markexpr

    for item in live_items:
        item.add_marker(pytest.mark.live)
        if not explicit_opt_in:
            item.add_marker(
                pytest.mark.skip(
                    reason="live tests require explicit `-m live` opt-in "
                    "(use `make live-smoke`); skipped to avoid firing real "
                    "external API traffic during a generic test run"
                )
            )
            continue
        missing = [v for v in _required_env(item) if not os.environ.get(v, "").strip()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"live smoke needs env: {', '.join(missing)} — set and re-run"
                )
            )


@pytest.fixture
def live_tmpdir(tmp_path: Path) -> Path:
    """Pytest's tmp_path with a clearer alias for live tests.

    No special behavior — just signals at the call site that this is a
    fresh, hermetic directory the smoke can write to without leaking
    state across runs.
    """
    return tmp_path
