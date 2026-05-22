# Backtest

Validation gauntlet semantics. Every signal PR cites this file.

For runtime architecture see `docs/ARCHITECTURE.md`. For data contracts see
`docs/DATA_MODEL.md`. For the original design rationale see
`docs/specs/2026-05-22-design.md` §10.

---

## 1. Tiered validation

```
T1 (info content)  ──pass──▶  T2 (portfolio backtest)  ──pass──▶  T3 (stress)
       │                              │                                  │
       ▼ fail                         ▼ fail                             ▼ fail
     kill                           iterate                          iterate/kill
```

Each tier produces a typed report (`InfoReport` or `BacktestReport`) persisted
as an MLflow artifact. Decisions (promote / iterate / kill) are made by the
`decide` node in `agents/research_graph.py` against the **code-checked**
constants in §2 — never by an LLM.

| Tier | Runtime | Pure info-content hypotheses | Portfolio hypotheses |
|---|---|---|---|
| **T1** | seconds–minutes | promote / kill | gate to T2 |
| **T2** | minutes–hours | n/a | gate to T3 |
| **T3** | hours | n/a | promote / iterate / kill |

---

## 2. Gate constants (INV-3: code-checked)

Lives at `src/auto_research/backtest/gates.py`. **Never read by an LLM. Never
overridden by an LLM critique.** The LLM critic produces qualitative addendum
text in the memo; the `decide` node never reads `state.critique`.

```python
T1_GATE: dict[str, float | int] = {
    "ic_t_stat_min": 2.0,
    "top_minus_bottom_t_stat_min": 1.8,
    "event_study_car_t_stat_min": 2.0,
    "n_observations_min": 30,
}

T2_GATE: dict[str, float] = {
    "deflated_sharpe_min": 1.0,
    "ic_mean_min": 0.02,
    "ic_half_life_min_days": 1.0,
    "capacity_usd_min": 1_000_000,
    "sharpe_net_min": 0.7,             # net of costs
    "sharpe_at_2x_costs_min": 0.3,     # stress sensitivity
    "max_beta_to_existing": 0.5,       # decorrelation vs alpha library
}

def check_t1_gate(report: InfoReport) -> Literal["promote", "kill"]: ...
def check_t2_gate(report: BacktestReport) -> Literal["promote", "iterate", "kill"]: ...
```

Adjusting these constants is a Tier 2 change — requires a Change Contract, an
ADR in `docs/decisions/`, and a regression-test that the alpha library's
existing signals still pass under the new gate.

---

## 3. T1 — information analysis

`src/auto_research/backtest/info_tests.py`. First-pass screening. Pure
info-content hypotheses stop here.

| Function | Returns | Use |
|---|---|---|
| `event_study(events_df, returns_df, window)` | CAR ± window, bootstrap 95% CI | Did the announcement move the stock? |
| `ic_analysis(features_df, fwd_returns_df, horizons)` | Spearman IC by horizon + t-stat + decay curve | Does the feature predict forward returns? |
| `quantile_sort(features_df, fwd_returns_df, n_quantiles)` | mean return per quantile + top-bottom + t-stat | Monotonic relationship? |
| `conditional_distribution(features_df, fwd_returns_df, condition)` | lift, hit-rate, posterior shift | Conditional on X, does the signal sharpen? |
| `mutual_information(features_df, fwd_returns_df)` | KSG estimator | Non-linear dependence test |
| `bootstrap_significance(stat_func, data, n_boot)` | bootstrap CI for any statistic | Generic CI |

Output: `InfoReport` dataclass (frozen) with all stats + `n_observations`.
`check_t1_gate(report)` returns `"promote"` iff every required threshold in
`T1_GATE` is satisfied; `"kill"` otherwise.

---

## 4. T2 — portfolio backtest

`src/auto_research/backtest/engine.py`. Built on `vectorbt.pro` for generic
mechanics + custom layers for López de Prado discipline.

### 4.1 Labels (`backtest/labels.py`)

**Triple-barrier with vol-adjusted bands:**

```python
def triple_barrier_label(
    entry_price: float,
    forward_path: pd.Series,           # close prices, t+1 .. t+H
    sigma: float,                      # realized vol, daily
    upper_barrier_pt: float,           # multiples of sigma
    lower_barrier_pt: float,
    time_barrier_days: int,
) -> Literal[-1, 0, 1]: ...
```

Upper barrier hit → `+1`. Lower → `-1`. Time barrier hit first → `0`.
`vol_adjusted_bands(sigma, mult)` returns `(upper, lower)` in price units.

### 4.2 Cross-validation (`backtest/cpcv.py`)

**Combinatorial purged CV with embargo.** Standard k-fold leaks labels in
financial time series because triple-barrier outcome dates overlap across
folds.

```python
def cpcv_splits(
    sample_times: pd.Series,           # entry datetime per sample
    label_end_times: pd.Series,        # outcome resolution datetime per sample
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
```

For ~2 years × weekly entries, `n_splits=6, n_test_splits=2` gives ~15
train/test combinations × embargoed purge. Hypothesis-tested invariants
(property tests in `tests/backtest/test_cpcv_properties.py`):

- No train sample has its label-end-time within an embargo distance of any
  test sample's entry time.
- Train and test index sets are disjoint.
- Every sample appears in test for exactly `comb(n_splits-1, n_test_splits-1)`
  combinations.

### 4.3 Deflated Sharpe (`backtest/deflated_sharpe.py`)

**Multiple-testing-adjusted Sharpe.** Naive Sharpe over-states significance
when many hypotheses were tried.

```python
def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,                     # how many signal variants were tested
    sample_size: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float: ...
# Returns deflated Sharpe per López de Prado (2014).
```

Every `BacktestReport.sharpe` field is a deflated Sharpe. Reporting raw
Sharpe in a report is a contract violation (see `docs/AI_CODE_STYLE.md` §4).

### 4.4 Cost model (`backtest/costs.py`)

Plumbed into `vbt.Portfolio.from_signals` via the `fees` and `slippage`
arguments. **All reported PnL is net of costs.**

```python
def realistic_costs(
    notional_usd: float,
    adv_usd: float,
    bid_ask_half_spread_bps: float,
    borrow_bps_annual: float,
    commission_bps: float = 0.5,
    impact_coefficient: float = 0.1,
) -> CostBreakdown:
    """
    Total cost (bps) =
        bid_ask_half_spread_bps
      + impact_coefficient * sqrt(notional_usd / adv_usd) * 10_000
      + commission_bps
      + (borrow_bps_annual * holding_days / 365) for short legs
    """
```

`bid_ask_half_spread_bps` and `adv_usd` are read from `price_features`
FeatureView. `borrow_bps_annual` defaults to 50bps (IBKR proxy) per ticker
unless overridden.

### 4.5 Engine (`backtest/engine.py`)

```python
def run_t2_backtest(
    signal: SignalDefinition,
    universe: list[str],
    start: date,
    end: date,
    params: BacktestParams,
) -> BacktestReport:
    """
    - daily long-short, sub-universe-neutralized
    - vol-scaled to 10% target portfolio vol
    - single-name cap 5%
    - turnover penalty in objective
    - ADV-based participation cap (5% of ADV)
    - costs plumbed via realistic_costs()
    - CV via cpcv_splits()
    - sharpe via deflated_sharpe(observed, n_trials=signal.n_variants_tested, ...)
    """
```

### 4.6 Reports (`backtest/report.py`)

```python
class InfoReport(BaseModel):
    signal_id: str
    n_observations: int
    ic_mean: float
    ic_t_stat: float
    ic_by_horizon: dict[int, float]
    top_minus_bottom_return: float
    top_minus_bottom_t_stat: float
    event_study_car: float
    event_study_car_t_stat: float
    mutual_information: float

class BacktestReport(BaseModel):
    signal_id: str
    sharpe_net: float                  # deflated, net of costs
    sharpe_at_2x_costs: float          # stress
    sharpe_gross_diagnostic_only: float  # never used in gates
    ic_mean: float
    ic_half_life_days: float
    deflated_sharpe: float             # explicit field
    capacity_usd: float
    max_drawdown: float
    turnover_annualized: float
    beta_to_existing_alpha: float
    cv_folds: int
    n_trials_for_deflation: int
```

`BacktestReport` is frozen. `sharpe_gross_diagnostic_only` exists for
notebook debugging — code that reads it for a gate decision fails a static
check in `tests/backtest/test_no_gross_sharpe_in_gates.py`.

---

## 5. T3 — stress tests

Run after T2 promote. Pass/fail recorded in `BacktestReport.stress_results`.

| Stress | What it does | Pass condition |
|---|---|---|
| 2x cost stress | Re-run backtest with `bid_ask_half_spread * 2` and `impact_coefficient * 2` | `sharpe_at_2x_costs ≥ T2_GATE["sharpe_at_2x_costs_min"]` |
| Regime breakouts | Slice CV folds by regime (high-vol / low-vol / trending / mean-reverting) | Sharpe positive in ≥ 3 of 4 regimes |
| Hyperparameter sensitivity | Sweep ±20% on each free parameter, count Sharpe sign flips | ≤ 1 sign flip across sweep |
| Decorrelation vs alpha library | `corr(signal_returns, existing_signal_returns)` for each promoted signal | `|corr| ≤ T2_GATE["max_beta_to_existing"]` |

---

## 6. Common mistakes

These are pre-empted by `docs/AI_CODE_STYLE.md` §4; repeated here in the
test-failure context:

- **Reading `sharpe_gross_diagnostic_only` in `check_t2_gate`.** Static check
  fails. Use `sharpe_net` and `deflated_sharpe`.
- **K-fold CV instead of CPCV.** Property test in
  `tests/backtest/test_cpcv_properties.py` catches embargo violation.
- **Triple-barrier with fixed (not vol-adjusted) bands.** Acceptable only when
  the signal hypothesis explicitly justifies it; default is vol-adjusted.
- **Not passing `n_trials` to `deflated_sharpe`.** Defaults to 1, which makes
  deflated_sharpe == observed_sharpe. Always pass the true count of variants
  tested across the research session.
- **Cost model parameters hardcoded in signal code.** They live in
  `price_features` FeatureView and `costs.py` defaults — never inlined.
- **LLM critique adjusting gate output.** The `decide` node reads
  `validation_report` only; `critique` lives in a separate state field used
  only for memo content.

---

## 7. Reproducibility

Every backtest run logs to MLflow:

- Signal definition (pickled)
- Full `BacktestReport` (artifact)
- Cost model parameters
- CV split indices (for replay)
- Random seed
- Code git SHA + prompt-registry SHA

`uv run auto-research backtest replay <mlflow_run_id>` re-runs the exact
configuration. Required for any signal promoted to the alpha library.
