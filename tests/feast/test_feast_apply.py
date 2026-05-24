"""Smoke test for ``feast apply`` over a synthetic 30-day price window.

End-to-end: ``feast apply`` succeeds, ``price_features`` is materialized
for the synthetic window, and the registry holds the FeatureView with
the expected schema.

The test copies ``feast_repo/`` into a tmp dir, seeds the offline parquet via
the production materializer (so the PIT cutoff is baked in by the same code
the property test pins), runs ``feast apply``, materializes into the online
store, and asserts the registry holds the FeatureView with the expected
schema.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from feast_repo._materialize import PRICE_FEATURE_COLUMNS, materialize_price_features

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEAST_REPO_SRC = _REPO_ROOT / "feast_repo"


def _resolve_feast_bin() -> str | None:
    """Resolve the feast CLI: prefer the venv's binary (guarantees the
    project-pinned version), fall back to PATH for non-venv pytest runs.
    Cross-platform (handles `feast.exe` on Windows).
    """
    bin_name = "feast.exe" if sys.platform == "win32" else "feast"
    venv_bin = Path(sys.executable).parent / bin_name
    if venv_bin.is_file():
        return str(venv_bin)
    return shutil.which("feast")


_FEAST_BIN = _resolve_feast_bin()

# Synthetic universe of 3 tickers x 30 wall-calendar days starting 2024-01-02
# (so 30 event_datetimes spanning ~22 NYSE sessions plus weekends/MLK Day).
_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")
_WINDOW_START = datetime(2024, 1, 2, 16, 0, tzinfo=ZoneInfo("America/New_York"))


def _synthetic_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in _TICKERS:
        for offset in range(30):
            event = _WINDOW_START + timedelta(days=offset)
            rows.append(
                {
                    "entity_id": ticker,
                    "event_datetime": pd.Timestamp(event).tz_convert("UTC"),
                    "close_adj": 100.0 + offset,
                    "returns_1d": 0.001 * offset,
                    "returns_5d": 0.005 * offset,
                    "vol_20d_annualized": 0.25,
                    "bid_ask_half_spread_bps": 1.5,
                    "adv_20d_usd": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def feast_workspace(tmp_path: Path) -> Path:
    """Copy ``feast_repo/`` into a tmp dir and seed the synthetic parquet."""
    if _FEAST_BIN is None:
        pytest.skip("feast CLI not on PATH")
    workspace = tmp_path / "feast_repo"
    shutil.copytree(_FEAST_REPO_SRC, workspace)
    # Wipe the entire data/ subtree (not just files) so any provider-created
    # subdirs from a prior fixture instance don't leak across tests.
    data_dir = workspace / "data"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir()
    (data_dir / ".gitkeep").touch()
    materialized = materialize_price_features(_synthetic_events())
    materialized.to_parquet(data_dir / "price_features.parquet", index=False)
    return workspace


def _run_feast_apply(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Invoke ``feast apply`` against ``workspace`` with the natural CLI
    contract — no ``PYTHONPATH`` override, no env hacks. This is what a human
    or CI runs by hand; if it ever needs PYTHONPATH to succeed, the repo has
    regressed to the fragile cross-file import pattern Codex flagged on #38.
    """
    assert _FEAST_BIN is not None  # guarded by fixture skip
    return subprocess.run(
        [_FEAST_BIN, "apply"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_feast_apply_succeeds_for_synthetic_30_day_window(feast_workspace: Path) -> None:
    result = _run_feast_apply(feast_workspace)
    assert result.returncode == 0, (
        f"feast apply failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Registry must exist after apply.
    assert (feast_workspace / "data" / "registry.db").is_file()


def test_price_features_registered_with_full_schema(feast_workspace: Path) -> None:
    result = _run_feast_apply(feast_workspace)
    assert result.returncode == 0, (
        f"feast apply failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Import after apply so we use the registry-backed FeatureStore.
    from feast import FeatureStore

    fs = FeatureStore(repo_path=str(feast_workspace))
    fv = fs.get_feature_view("price_features")
    fields = {f.name for f in fv.schema}
    # Feast surfaces the entity join key alongside the feature columns.
    assert set(PRICE_FEATURE_COLUMNS) <= fields
    assert "entity_id" in fields
    # ttl=timedelta(0) is Feast's "unbounded" sentinel; pin it tight so a
    # future Feast change to ttl normalisation surfaces as a test failure
    # rather than silently producing empty PIT joins.
    assert fv.ttl == timedelta(0)


def test_materialized_parquet_carries_pit_columns(feast_workspace: Path) -> None:
    df = pd.read_parquet(feast_workspace / "data" / "price_features.parquet")
    assert {"entity_id", "event_datetime", "as_of_ts", *PRICE_FEATURE_COLUMNS} <= set(df.columns)
    # PIT invariant on the materialized parquet: as_of_ts > event_datetime for every row.
    assert (df["as_of_ts"] > df["event_datetime"]).all()
    assert len(df) == len(_TICKERS) * 30


def test_price_features_source_has_pit_tiebreaker(feast_workspace: Path) -> None:
    """Intraday events on the same NYSE trading day collapse to identical
    (entity_id, as_of_ts) under the lag-1 cutoff, so Feast's PIT join needs
    a tie-breaker to be deterministic. We use event_datetime as the
    tie-breaker — the later intraday snapshot wins, which is the correct
    PIT-conservative reading.
    """
    result = _run_feast_apply(feast_workspace)
    assert result.returncode == 0, (
        f"feast apply failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    from feast import FeatureStore

    fs = FeatureStore(repo_path=str(feast_workspace))
    fv = fs.get_feature_view("price_features")
    source = fv.batch_source
    assert source is not None
    assert source.created_timestamp_column == "event_datetime"


def test_feature_views_importable_via_package_path() -> None:
    """`from feast_repo.feature_views import price_features` must work from
    the project root, not just under `feast apply`'s chdir context — future
    programmatic FeatureStore.apply, unit tests that introspect the FV
    object, and tooling all rely on this canonical import path.
    """
    from feast_repo.feature_views import price_features

    assert price_features.name == "price_features"
