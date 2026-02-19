# Testing & Validation (Module 7)

This folder contains the project’s `pytest` suite and short notes that map the tests back to the **Testing_&_Validation** assessment items.

Run all tests from the repo root:

```bash
pytest -q
```

> Note: One test (`test_sql_connection.py`) is an **integration** check against a real SQL Server connection and may require valid credentials/config on the machine it’s run on.

---

## 7.1 Compare Forecasts Against Actuals (RMSE/MAE) → Implemented as AIC (Model Selection)

### What I did
Instead of implementing a full backtesting workflow with RMSE/MAE, the forecasting component performs **model selection using AIC (Akaike Information Criterion)**.

In this project, the ETS forecast specification is selected by searching a small grid of candidate configurations and choosing the one with the **lowest AIC**:

- Model selection logic: `small_grid_search_aic_ets(...)` in `src/forecasting/ets.py`
- Dashboard usage (per-metric, per-filter-state): ETS selection + caching in `src/dashboard/app.py`

### Why AIC (in this iteration)
- **Works without an explicit holdout set**: RMSE/MAE require an “actuals vs forecast” comparison on unseen periods (e.g., rolling-origin evaluation). In this dashboard-led workflow the user can change filters (service line, date range) frequently, making a fixed train/test split less stable and more complex to explain.
- **Balances fit vs complexity**: AIC explicitly penalises model complexity, so it discourages overfitting when comparing multiple ETS configurations.
- **Fast enough for interactive use**: AIC can be computed from the fitted model results and used for lightweight selection with a small search space, which is important for responsiveness inside callbacks.

### Limitation (and future extension)
AIC is primarily an **in-sample** criterion; it does not guarantee the best out-of-sample accuracy. A natural next step (if required) would be to add a rolling backtest that reports RMSE/MAE over the last *k* months for each service line and selected metric.

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
