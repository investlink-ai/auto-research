"""Feast FeatureViews. See ``docs/DATA_MODEL.md`` §3.

Every FeatureView's source parquet carries two timestamps (INV-1):
``event_datetime`` (when the underlying event happened) and ``as_of_ts``
(when the row becomes tradeable, baked at write-time via
``feast_repo._pit.next_trading_day_cutoff``). Feast's PIT join is configured
against ``as_of_ts`` so lookahead is structurally impossible.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import FeatureView, Field, FileSource
from feast.types import Float64

# Cross-file imports inside feast_repo/ need to work in TWO contexts:
#   * Project-root Python (mypy, pytest, programmatic FeatureStore.apply)
#     where `feast_repo` is a package and the absolute import resolves.
#   * `feast apply` CLI, which chdir's into feast_repo/ and puts it (not its
#     parent) on sys.path, leaving only the bare module name resolvable.
# Try the package path first so mypy can verify the symbol; fall back to the
# bare-name import at apply time. If `entities.py` is renamed/removed the
# package-path branch fails at typecheck (mypy strict) AND at runtime — the
# fallback only kicks in for the apply-time chdir context, where neither
# import works if the symbol is wrong.
try:
    from feast_repo.entities import entity_id
except ModuleNotFoundError:  # pragma: no cover - exercised by `feast apply` only
    from entities import entity_id  # type: ignore[no-redef,import-not-found]

_DATA_DIR = Path(__file__).resolve().parent / "data"

# created_timestamp_column wired to event_datetime: Feast's PIT join uses this
# as the tie-breaker when multiple rows share the same (entity_id, as_of_ts).
# Under the lag-1 cutoff, ALL intraday events for one ticker on one ET
# trading day collapse onto a single as_of_ts (the next session's close), so
# without a tie-breaker the served row is non-deterministic. Using
# event_datetime means Feast picks the LATEST intraday snapshot (ORDER BY
# created_timestamp DESC, keep first) — the correct PIT-conservative reading.
# Feast docs nominally describe this field as ingestion-time; we're using it
# as a same-day-recency tie-breaker. Documented in docs/DATA_MODEL.md §3.5.
price_features_source = FileSource(
    name="price_features_source",
    path=str(_DATA_DIR / "price_features.parquet"),
    timestamp_field="as_of_ts",
    created_timestamp_column="event_datetime",
)

# ttl=timedelta(0) is Feast's documented sentinel for "unbounded retention" —
# matches what Feast's registry round-trip produces from ttl=None, but spelled
# explicitly so consumers reading `fv.ttl` see the exact value the registry
# stores. Price history is forever-valid per docs/DATA_MODEL.md §3.5.
price_features = FeatureView(
    name="price_features",
    entities=[entity_id],
    ttl=timedelta(0),
    schema=[
        Field(name="close_adj", dtype=Float64),
        Field(name="returns_1d", dtype=Float64),
        Field(name="returns_5d", dtype=Float64),
        Field(name="vol_20d_annualized", dtype=Float64),
        Field(name="bid_ask_half_spread_bps", dtype=Float64),
        Field(name="adv_20d_usd", dtype=Float64),
    ],
    online=True,
    source=price_features_source,
    description="Daily price-derived features per ticker (PIT-disciplined).",
)
