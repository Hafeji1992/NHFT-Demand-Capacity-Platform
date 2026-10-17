# Testing & Validation (Module 7)

This folder contains the project’s `pytest` suite, the ETS forecast backtest, and short notes that map them back to the **Testing_&_Validation** assessment items.

| File | Type | Covers |
|---|---|---|
| `ets_backtest.py` | Standalone script (rolling-origin backtest) | 7.1 |
| `test_sql_connection.py` | `pytest` integration test | 7.3 |
| `test_patient_data_ingestion.py` | `pytest` unit tests | 7.3 |
| `test_staffing_data_ingestion.py` | `pytest` unit tests | 7.3 |

Run all tests from the repo root:

```bash
pytest -q
```

Run the backtest from the repo root (it is a script, not a `pytest` test, so `pytest` does not pick it up):

```bash
python src/tests/ets_backtest.py --end-date 2026-02-28 --sarima --jobs 6
```

> Note: One test (`test_sql_connection.py`) is an **integration** check against a real SQL Server connection and may require valid credentials/config on the machine it’s run on.

---

## 7.1 Compare Forecasts Against Actuals → AIC Model Selection + Rolling-Origin Backtest

### What I did
Forecast accuracy is validated in two layers:

1. **Model selection (in-sample) using AIC.** For each series, the ETS (Holt-Winters) specification is chosen from a small grid of candidate configurations by **lowest AIC**.
   - Model selection logic: `small_grid_search_aic_ets(...)` in `src/forecasting/ets.py`
   - Dashboard usage (per-metric, per-filter-state): ETS selection + caching in `src/dashboard/app.py`
2. **Out-of-sample validation using a rolling-origin backtest.** `ets_backtest.py` compares forecasts against what actually happened, using only data that would have been available at each forecast origin.

### Why AIC for selection
- **Works without an explicit holdout set**: in the dashboard the user changes filters (service line, date range) frequently, so a fixed train/test split inside each callback would be unstable and slow.
- **Balances fit vs complexity**: AIC penalises model complexity, which discourages overfitting when comparing ETS configurations.
- **Fast enough for interactive use**: AIC comes straight from the fitted model, so selection stays responsive inside callbacks.

AIC is an **in-sample** criterion and does not on its own guarantee good out-of-sample accuracy. The backtest checks that.

### How the backtest works
- Each service's monthly series (default metric: `referrals`) is built the same way the platform builds it (`build_monthly_series(...)`).
- An **expanding training window** is rolled forward one month at a time. At each origin the ETS specification is **re-selected by AIC on the training data only** (no leakage), the model is refitted, and the next **6 months** are forecast.
- Defaults: minimum 24 months of training, horizon 6, step 1. Services with too little history are excluded; folds with an all-zero training window are skipped.
- ETS is compared with three simple **baselines**: naive (last value), seasonal naive (same month last year), and the mean of the last 12 months.
- With `--sarima`, the **archived SARIMA forecaster** (`src/forecasting/archived_sarima`) is backtested on the same folds. Its order is selected once per service on the first training window (or at every origin with `--sarima-reselect`).

**Metrics reported:** pooled WAPE, MASE (scaled by in-sample seasonal-naive error; below 1 beats seasonal naive), per-service win rates, error by horizon, empirical coverage of the 95% prediction interval, and Wilcoxon signed-rank tests on per-service MAE differences.

**Outputs** (written to `data/`):
- `backtest_ets_results.csv`: every forecast point (origin, horizon, actual, each model's forecast, ETS interval, chosen spec, fit time)
- `backtest_ets_results_by_service.csv`: per-service MAE / MASE / coverage
- `backtest_ets_results_sarima_by_service.csv`: per-service ETS vs SARIMA (when `--sarima` is used)

### Results (referrals, data to Feb 2026)
105 services, 618 folds (origins Mar–Aug 2025), 3,708 forecast points, 6-month horizon. AIC chose a level-only model in 48% of fits, a seasonal model in 40% and a trend in 27%.

| Model | Pooled WAPE | Median MASE |
|---|---|---|
| **ETS (production)** | **13.4%** | 0.89 |
| Naive | 14.3% | 1.02 |
| 12-month mean | 19.6% | 0.83 |
| SARIMA (archived) | 19.5% | 1.04 |
| Seasonal naive | 22.7% | 0.99 |

| Comparison (per-service MAE) | ETS win rate | Wilcoxon p (n = 105) |
|---|---|---|
| ETS vs seasonal naive | 60.0% | 0.008 |
| ETS vs naive | 66.7% | not tested |
| ETS vs 12-month mean | 41.9% | 0.063 |
| ETS vs SARIMA | 73.3% | < 0.001 |

95% interval coverage: **ETS 90.4%**, SARIMA 72.6%. All SARIMA folds produced a forecast, so the ETS vs SARIMA comparison uses the same points.

### What the results show
- ETS has the **lowest pooled error** of every model tested, and beats seasonal naive by a statistically significant margin (p = 0.008).
- ETS clearly outperforms the **SARIMA model it replaced**: lower error on 73% of services (p < 0.001) and much better-calibrated intervals (90% vs 73%). This supports the decision to move from SARIMA to ETS.
- The **12-month mean** has a slightly lower median MASE and beats ETS on more services (58%), but the difference is not significant (p = 0.063). Pooled WAPE favours ETS because ETS does better on the larger services. On smaller, flatter series a simple average is just as good.
- ETS 95% intervals cover **90%** of actuals, so they are a little too narrow. Users should read them as a guide, not a strict 95% guarantee.

### Limitations
- The backtest covers one metric (`referrals`) and six origins per service. Other metrics and longer histories could behave differently.
- The backtest runs offline. It is not run inside the dashboard, which still selects models by AIC for speed.

---

## 7.2 Parameter Tuning & Fairness Adjustments

### What I did
I designed the dashboard so that key “tuning” parameters update **dynamically on every callback**, meaning the analysis remains consistent and fair across:
- different **service lines** selected in the dropdown
- different **time windows** selected in the date-range slider

The main “fairness adjustment” implemented is a **target framework based on a configurable percentile** (e.g., 50th/60th/70th/etc.). In practice:
- the user chooses a percentile via `demand-percentile-dropdown`
- each callback recomputes a **service-line-specific target** using the currently filtered dataset
- downstream metrics (e.g., Demand Ratio, Sustainable Caseload) are recalculated from those targets

### Where it happens
- Filters are applied and stored in `dcc.Store`: `filter_data(...)` writes `filtered-data-store` in `src/dashboard/app.py`
- Tab content is re-rendered when filters/percentile change: `update_tab_content(...)`
- Demand Analysis table recomputes when filters/percentile change: `update_patient_summary_table(...)`
- Target-based computations (examples): `_compute_demand_ratio_latest(...)`, `_compute_sustainable_caseload_latest(...)`

This design ensures the “target” (and therefore the derived ratios) is **recomputed per cohort** rather than applying a single global assumption to all services.

---

## 7.3 Stress-Test Direct-Query Pipeline

### What I did
I created a set of tests in this folder to validate that the **direct-query → ingestion → cleaned dataframe** pipeline behaves correctly and fails safely.

These tests stress the pipeline in two complementary ways:

1) **Integration connectivity check**
- `test_sql_connection.py` uses the reusable `SQLServerConnection` class and attempts a real connection.
- Purpose: quickly detect environment/config issues that would break “direct query” usage.

2) **Deterministic unit tests (no real database required)**
- `test_patient_data_ingestion.py` and `test_staffing_data_ingestion.py` use `monkeypatch` with `FakeSQLServerConnection`.
- Purpose: validate extraction + transformations reliably in CI/offline environments.

### What the ingestion tests validate
Across patient and staffing ingestion, the tests check:
- **column normalisation** (standardised snake_case naming)
- **type coercion / parsing** (e.g., `period_end` parsed to datetime)
- **canonical ID formatting** (provider codes zero-padded to three digits)
- **data quality guardrails** (schema validation, duplicate detection where applicable)
- **CSV loading behaviour** (successful reads + missing-file errors)

Together, these tests act as a stress-test harness: they confirm the pipeline continues to produce consistent, analysis-ready frames even when upstream shapes (column cases, provider code types) vary.

---

## 7.4 Resolve Model Errors & Outlier Handling

### What I tested / evaluated
During development I explored adding automated **outlier handling** and additional **model error checks** (e.g., detecting extreme spikes/drops and handling edge cases where time series are too short/flat to fit reliably).

The feature was assessed from a testing perspective in terms of:
- **Correctness**: whether suspected outliers would be flagged consistently and whether the downstream forecast would remain stable.
- **Robustness**: ensuring the pipeline fails safely on problematic series (missing months, very short histories, constant values).
- **Runtime impact**: measuring the additional processing time when these checks run inside the dashboard’s interactive callbacks.

### Why it was not built into the live dashboard pipeline
The main constraint was **performance**.

In this project the dashboard is designed for interactive exploration (users frequently change service line(s) and date range). Running additional outlier detection / correction on every filter change would:
- noticeably **slow data loading and callback execution**, and
- reduce the overall responsiveness of the dashboard experience.

### Stakeholder communication & decision
This trade-off (better automated outlier handling vs. slower dashboard performance) was communicated to stakeholders.

Stakeholders were initially reluctant to drop the feature, but agreed that maintaining a responsive dashboard was higher priority for this iteration. The outcome was to **leave outlier handling out of the runtime pipeline** and keep the implementation focused on stability and speed.

### Future option (if re-scoped)
If outlier handling is required later, the preferred approach would be to run it as an **offline / scheduled preprocessing step** (or as a cached, precomputed dataset), so the dashboard still loads quickly while benefiting from cleaner inputs.
