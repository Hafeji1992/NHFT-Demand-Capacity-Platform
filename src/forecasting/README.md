# Forecasting with ETS (Error Trend and Seasonality, or Holt-Winters Exponential Smoothing)

This folder contains the time-series forecasting utilities for the NHFT Demand–Capacity Platform.

The goal is to generate short-to-medium horizon planning forecasts (typically **6-month** and **12-month**) from historical monthly aggregates and provide **uncertainty bounds** (95% confidence intervals) for the dashboard.

## Why ETS is used (dashboard default)

Intially, Seasonal ARIMA (Auto Regressive Integrated Moving Average) was implemented, though now the dashboard uses **ETS** as the forecasting method.

This change was made because the project’s interactive dashboard needs forecasting that is:

- **Fast enough for UI use**: forecasts are generated on-demand from user selections.
- **Stable on short histories**: the current dataset spans about **30 monthly points** per metric.
- **Robust to convergence issues**: SARIMA/SARIMAX can emit optimisation warnings (e.g., non-convergence) more often on short seasonal series.

ETS is a strong baseline for operational planning series because it captures:

- **Level + trend** (optionally damped)
- **Seasonality** (for monthly series, annual seasonality uses period $s=12$)

In practice, ETS usually fits quickly and reliably for ~24–36 months of monthly data.

## What we are doing

At a high level, the pipeline is:

1. **Aggregate to monthly time series** (one value per month)
   - Converts row-level data into a regularly spaced monthly series.
   - Fills missing months (if any) so the model sees a consistent calendar.

2. **Fit an ETS model**
   - Models the series using level/trend/seasonal components.
   - Can optionally use a **damped trend**, which often improves medium-horizon stability.

3. **Forecast forward and produce 95% confidence intervals**
   - Outputs are returned as a plot-ready DataFrame.
   - Intervals are **approximate** (see caveats below).

## Model selection (AIC)

For the dashboard, ETS specifications are selected using a **small AIC search**.

- The search evaluates a small set of combinations across:
  - trend: add / none
  - seasonal: add / none
  - damped trend: true / false
- The spec with the **lowest AIC** is selected.

To keep the UI responsive:

- The search space is intentionally small.
- Selection is cached **per metric and per filter state** (so it only runs once for the same filtered series).

### Dashboard caching (spec per metric per filter state)

The dashboard keeps two small **in-memory LRU caches** so forecasts feel responsive:

- **Best-spec cache**: the selected `EtsSpec` is cached using a key derived from:
  - the metric name (e.g., `referrals`)
  - a *series signature* for the filtered monthly series (length + endpoints + a fast content hash)

- **Forecast-frame cache**: the computed forecast frame (future rows only) is cached using a key derived from:
  - the metric name
  - the same series signature
  - the selected `EtsSpec`
  - forecast settings (horizon, confidence level, frequency)

This means:

- The first time you toggle forecast for a metric under a given filter selection, the model is fit and cached.
- Subsequent toggles with the same filters reuse the cached spec + forecast.
- Cache lives in-process (it resets when the dashboard restarts).

## Frequency handling (monthly)

- Series are coerced to a regular month-end frequency using pandas’ **ME** frequency alias.
- This ensures the modelling layer and the plotting layer agree about month timestamps.

## Outputs

The main helpers return a DataFrame with history and forecast in one table, including:

- `y` (observed)
- `yhat` (forecast mean)
- `yhat_lower`, `yhat_upper` (95% interval)

This plugs directly into Plotly traces (solid actuals, dashed forecast, and a shaded interval band).

## Key files

- `preprocessing.py`
  - `build_monthly_series(df, value_col, date_col="period_end")`
  - `coerce_monthly_series(y, freq="ME")`
- `ets.py`
  - `EtsForecaster` (fit + forecast)
  - `EtsSpec` (trend/seasonal/damped settings)
  - `small_grid_search_aic_ets(...)` (lightweight AIC search)
- `forecast.py`
  - `ForecastConfig` (forecast horizon, confidence level, frequency)
  - `make_ets_forecast_frame(y, config=..., spec=...)` (history + future)
- `ets_report.py`
  - Console report script for ETS selection + fit + forecast preview
- `archived_sarima/`
  - `sarima.py`, `sarima_report.py` (kept for reference; not used by dashboard)

## Quick example (ETS)

```python
from forecasting.preprocessing import build_monthly_series
from forecasting.forecast import make_ets_forecast_frame, ForecastConfig
from forecasting.ets import EtsSpec

# df must have: period_end (datetime-like), referrals (numeric)
y = build_monthly_series(df, "referrals", date_col="period_end")

# 6-month forecast with 95% interval (auto-select best ETS spec via AIC)
frame_6 = make_ets_forecast_frame(
    y,
    config=ForecastConfig(months_ahead=6, conf_level=0.95, freq="ME", auto_select=True),
)

# 12-month forecast with 95% interval (explicit fixed spec)
frame_12 = make_ets_forecast_frame(
    y,
    config=ForecastConfig(months_ahead=12, conf_level=0.95, freq="ME"),
    spec=EtsSpec(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True),
)
```

## SARIMA (archived)

The earlier SARIMA/SARIMAX implementation has been moved into `archived_sarima/` for reference.
It is not used by default in the dashboard because seasonal SARIMA fitting was slower and more
likely to emit optimiser non-convergence warnings on short monthly series.

## Important caveats

- Forecasts are only as good as the historical signal; sudden operational changes (policy, service redesign, data definition changes) may not be captured.
- ETS intervals used here are **approximate** (residual-based). They are useful for planning bands, but they are not strict statistical guarantees.
- If a series is extremely short or mostly flat/zero, any method will struggle; in those cases, the dashboard should display forecasts cautiously (and may show very wide or very tight bands depending on residual variance).