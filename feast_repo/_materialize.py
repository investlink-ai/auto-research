"""Materializer for the ``price_features`` FeatureView (AGENTS.md INV-1).

The lag-1 PIT cutoff is baked into ``as_of_ts`` here, at write-time. This is
the contract the ``pit-check`` skill defends: query code must never compute
``as_of_ts`` arithmetically, only read it.
"""

from __future__ import annotations

import pandas as pd

from feast_repo._pit import next_trading_day_cutoff

PRICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "close_adj",
    "returns_1d",
    "returns_5d",
    "vol_20d_annualized",
    "bid_ask_half_spread_bps",
    "adv_20d_usd",
)


def materialize_price_features(events: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with ``as_of_ts`` populated via the PIT cutoff.

    ``events`` must contain ``entity_id``, ``event_datetime``, and every column
    in :data:`PRICE_FEATURE_COLUMNS`. ``event_datetime`` must be a tz-aware
    datetime column with no nulls (validated once at the boundary so the
    per-row cutoff doesn't surface confusing errors from inside
    ``exchange_calendars``). The returned frame is the on-disk shape written
    to ``feast_repo/data/price_features.parquet``.
    """
    missing = {"entity_id", "event_datetime", *PRICE_FEATURE_COLUMNS} - set(events.columns)
    if missing:
        raise ValueError(f"events frame missing required columns: {sorted(missing)}")
    if not isinstance(events["event_datetime"].dtype, pd.DatetimeTZDtype):
        raise TypeError(
            "materialize_price_features: event_datetime must be a tz-aware "
            f"datetime column; got dtype={events['event_datetime'].dtype!r}"
        )
    if events["event_datetime"].isna().any():
        # TypeError mirrors the per-row cutoff's NaT rejection; docs/DATA_MODEL.md
        # §1.1 documents this contract — producer try/excepts catch one type.
        raise TypeError("materialize_price_features: event_datetime contains NaT")
    out = events.copy()
    out["as_of_ts"] = out["event_datetime"].map(next_trading_day_cutoff)
    return out
