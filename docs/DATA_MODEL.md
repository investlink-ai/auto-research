# Data Model

Feast feature store schema + point-in-time discipline. This is the cross-plane
contract: extraction writes here, signals read here. Nothing else.

For runtime architecture see `docs/ARCHITECTURE.md`. For schema details see
`docs/CONTRACTS.md`.

---

## 1. PIT contract (INV-1)

Every row in every FeatureView carries two timestamps:

- **`event_datetime`** — when the underlying event happened (filing
  acceptance time, transcript start time, market close for prices).
- **`as_of_ts`** — when the feature became *usable* for trading. Equal to
  `event_datetime + 1 trading day cutoff`. **Baked in at write-time.**

Feast `get_historical_features` joins on `as_of_ts ≤ entity_event_timestamp`.
This is the lookahead-protection contract. Callers never compute
`as_of_ts - timedelta(...)` at query time. If you find yourself doing that,
stop and re-read this section.

```python
# Correct (at write-time, inside the FeatureView materializer)
df["event_datetime"] = filing.accepted_datetime
df["as_of_ts"] = next_trading_day_cutoff(filing.accepted_datetime)
write_to_feast(df)

# Wrong (at query-time)
features = fs.get_historical_features(...)
features = features[features["as_of_ts"] <= entity_ts - pd.Timedelta(days=1)]  # NO
```

`next_trading_day_cutoff(t)` returns the next NYSE close ≥ `t + 1 trading day`,
respecting holidays. Implementation in `feast_repo/_pit.py`.

### 1.1 Producer-side contract

`event_datetime` columns written to `feast_repo/data/*.parquet` MUST be
tz-aware (UTC by convention). Naive timestamps are rejected at the
materializer boundary with `TypeError` — this is intentional and protects
INV-1 from silent off-by-one ET-date bucketing.

```python
# Correct: tz-aware (UTC) — survives pyarrow parquet round-trip.
df["event_datetime"] = pd.to_datetime(raw_timestamps, utc=True)

# Wrong: naive — would be re-interpreted as UTC by downstream code with no
# warning, producing one-trading-day-too-early cutoffs for ET-centric data.
df["event_datetime"] = pd.to_datetime(raw_timestamps)  # NO
```

The pyarrow parquet writer preserves tz metadata only if the column is
tz-aware before `to_parquet`. A naive column round-trips as
`datetime64[ns]` (still naive) and the materializer rejects it on the next
read. Tests pinning both paths:

- `tests/feast/test_pit_properties.py::test_next_trading_day_cutoff_rejects_tz_naive`
- `tests/feast/test_pit_properties.py::test_materialize_price_features_rejects_tz_naive_column`
- `tests/feast/test_pit_properties.py::test_naive_event_datetime_after_parquet_roundtrip_is_rejected`
- `tests/feast/test_pit_properties.py::test_tz_aware_event_datetime_survives_parquet_roundtrip`

`pd.NaT` is likewise rejected with `TypeError` at the boundary (rather than
surfacing as a cryptic `AttributeError` from inside `exchange_calendars`).

---

## 2. Entity

```python
# feast_repo/entities.py
from feast import Entity, ValueType

entity_id = Entity(
    name="entity_id",
    value_type=ValueType.STRING,
    description="Tradeable ticker symbol; universe-managed",
    join_keys=["entity_id"],
)
```

Single entity across all FeatureViews. Universe lives at
`data/universe/universe_v1.json` and is git-tracked. Changes are explicit and
versioned.

---

## 3. FeatureViews

Source location: `src/auto_research/feast_repo/feature_views.py`. All
FeatureViews use file-based Parquet offline storage under `feast_repo/data/`.

### 3.1 `ten_k_features`

| Feature | Type | Source | Notes |
|---|---|---|---|
| `guidance_tone_score` | Float | `TenKOutput.guidance_tone.confidence` * sign | -1..+1 |
| `accrual_flag_count` | Int | `len(TenKOutput.accrual_flags)` | |
| `supplier_mention_count` | Int | `len(TenKOutput.supplier_mentions)` | |
| `customer_mention_count` | Int | `len(TenKOutput.customer_mentions)` | |
| `language_novelty_score` | Float | `TenKOutput.language_novelty_score` | 0..1 |
| `risk_factor_delta_count` | Int | `len(TenKOutput.risk_factor_deltas)` | |

TTL: 365 days. Materialized after every nightly extraction batch.

### 3.2 `transcript_features`

| Feature | Type | Source |
|---|---|---|
| `prepared_remarks_tone_score` | Float | `TranscriptOutput.prepared_remarks_tone` |
| `q_and_a_evasiveness_score` | Float | `TranscriptOutput.q_and_a_evasiveness` |
| `forward_statement_count` | Int | `len(TranscriptOutput.forward_statements)` |
| `forward_horizon_days_mean` | Float | mean horizon across forward statements |

TTL: 90 days (faster decay; quarterly cadence).

### 3.3 `eight_k_features`

| Feature | Type | Source |
|---|---|---|
| `event_classification` | String | `EightKOutput.event_classification` |
| `milestone_mention_count` | Int | `len(EightKOutput.milestone_mentions)` |
| `dilution_flag_count` | Int | `len(EightKOutput.dilution_language_flags)` |

TTL: 30 days (event-driven; signal decays quickly).

### 3.4 `s_filing_features`

| Feature | Type | Source |
|---|---|---|
| `form_type` | String | `SFilingOutput.form_type` |
| `dilution_event_flag` | Bool | `SFilingOutput.dilution_event.confidence > 0.5` |
| `capital_raise_language_count` | Int | `len(SFilingOutput.capital_raise_language)` |

TTL: 60 days.

### 3.5 `price_features`

| Feature | Type | Source |
|---|---|---|
| `close_adj` | Float | adjusted close price |
| `returns_1d` | Float | log return |
| `returns_5d` | Float | log return |
| `vol_20d_annualized` | Float | rolling realized vol |
| `bid_ask_half_spread_bps` | Float | from FMP, daily snapshot |
| `adv_20d_usd` | Float | 20-day average dollar volume (for cost model) |

TTL: unbounded (price history is forever-valid).

### 3.6 `signal_features`

| Feature | Type | Source |
|---|---|---|
| `signal_a1_score` | Float | output of `signals/a1_supply_chain.py` |
| `signal_a2_score` | Float | output of `signals/a2_pead.py` |
| `signal_b1_score` | Float | output of `signals/b1_frontier.py` |
| `combined_alpha` | Float | output of `signals/combiner.py` |

TTL: 7 days. Materialized daily.

---

## 4. FeatureServices

Service-level grouping for clean signal-train / backtest reads.

```python
# feast_repo/feature_services.py

signal_a1 = FeatureService(
    name="signal_a1",
    features=[
        transcript_features[["forward_statement_count", "forward_horizon_days_mean"]],
        ten_k_features[["supplier_mention_count"]],
        price_features[["returns_5d", "vol_20d_annualized"]],
    ],
)

signal_a2 = FeatureService(
    name="signal_a2",
    features=[
        transcript_features[["q_and_a_evasiveness_score"]],
        ten_k_features[["guidance_tone_score", "language_novelty_score"]],
        price_features[["returns_5d", "vol_20d_annualized"]],
    ],
)

signal_b1 = FeatureService(
    name="signal_b1",
    features=[
        eight_k_features[["event_classification", "milestone_mention_count", "dilution_flag_count"]],
        s_filing_features[["dilution_event_flag"]],
        price_features[["close_adj", "vol_20d_annualized"]],
    ],
)
```

Adding a new signal = add a new FeatureService. Adding a feature to an
existing service is a non-breaking change.

---

## 5. Schema migration

Migrations are explicit and version-tagged. Steps:

1. Edit `feature_views.py` (add/rename/remove a feature).
2. If renaming/removing: bump the `prompt_version` of the source worker
   (INV-6) and update the materializer to populate both old and new during a
   transition window.
3. Run `feast apply` from the `feast_repo/` directory — Feast validates and
   updates the local SQLite registry.
4. Run `scripts/backfill_feast.py --feature-view ten_k_features --from 2024-01-01`
   to repopulate.
5. Commit the schema change + a migration entry in `docs/decisions/` if the
   change is breaking.

The PIT contract (§1) survives migrations — `as_of_ts` semantics are
schema-independent.

---

## 6. Data lineage

| Layer | Storage | Path |
|---|---|---|
| Raw documents | Filesystem | `data/raw/{source}/{entity_id}/{year}/{doc_id}.{ext}` |
| Extracted JSON | JSONL, partitioned | `data/extracted/{worker}/{year}/{month}.jsonl` |
| Quarantine | JSON, flat | `data/quarantine/{worker}/{doc_id}.json` |
| Feast offline | Parquet | `feast_repo/data/{feature_view}.parquet` |
| Feast registry | SQLite | `feast_repo/data/registry.db` |
| Backtest artifacts | MLflow file backend | `mlruns/` |
| Memos | Markdown + LanceDB | `data/memos/*.md` + `data/rag/memos.lance` |

Every layer is reproducible from `data/raw/` + git HEAD of the prompt
registry. `data/raw/` and `data/extracted/` are the audit trail — never
deleted, never rewritten (enforced by `.claude/settings.json` deny rule).

---

## 7. Common queries

### Read PIT-correct features for a signal training set

```python
from feast import FeatureStore

fs = FeatureStore(repo_path="feast_repo")
entity_df = pd.DataFrame({
    "entity_id": tickers,
    "event_timestamp": training_dates,
})
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=fs.get_feature_service("signal_a2"),
).to_df()
# training_df is guaranteed PIT-correct: as_of_ts ≤ event_timestamp
```

### Materialize incrementally

```bash
cd feast_repo
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

Triggered by `scripts/nightly_pipeline.sh` after extraction completes.
