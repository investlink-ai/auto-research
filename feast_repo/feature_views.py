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
from typing import TYPE_CHECKING

from feast import FeatureView, Field, FileSource
from feast.types import Float64

if TYPE_CHECKING:
    # Mypy resolves the symbol via the package path so a rename in
    # entities.py is caught at typecheck time, not as a runtime crash inside
    # the `feast apply` subprocess.
    from feast_repo.entities import entity_id
else:
    # `feast apply` chdir's into this directory and puts it (not its parent)
    # on sys.path, so cross-module imports inside feast_repo/ use the
    # no-prefix Feast tutorial convention.
    from entities import entity_id

_DATA_DIR = Path(__file__).resolve().parent / "data"

# created_timestamp_column is omitted: Feast treats it as the row's
# ingestion/write time used as a tie-breaker (conventionally >= timestamp_field),
# and event_datetime is the underlying real-world event time (<= as_of_ts) — the
# opposite semantic. The materializer emits one row per (entity_id, event_datetime),
# so there's no tie-breaker need today; if late-arrivals appear later, add a
# true ingest timestamp then.
price_features_source = FileSource(
    name="price_features_source",
    path=str(_DATA_DIR / "price_features.parquet"),
    timestamp_field="as_of_ts",
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
