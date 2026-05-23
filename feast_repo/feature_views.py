"""Feast FeatureViews. See ``docs/DATA_MODEL.md`` §3.

Every FeatureView's source parquet carries two timestamps (INV-1):
``event_datetime`` (when the underlying event happened) and ``as_of_ts``
(when the row becomes tradeable, baked at write-time via
``feast_repo._pit.next_trading_day_cutoff``). Feast's PIT join is configured
against ``as_of_ts`` so lookahead is structurally impossible.
"""

from __future__ import annotations

from pathlib import Path

from feast import FeatureView, Field, FileSource
from feast.types import Float64

from feast_repo.entities import entity_id

_DATA_DIR = Path(__file__).resolve().parent / "data"

price_features_source = FileSource(
    name="price_features_source",
    path=str(_DATA_DIR / "price_features.parquet"),
    timestamp_field="as_of_ts",
    created_timestamp_column="event_datetime",
)

# ttl=None -> unbounded retention, per docs/DATA_MODEL.md §3.5 (prices forever-valid).
price_features = FeatureView(
    name="price_features",
    entities=[entity_id],
    ttl=None,
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
