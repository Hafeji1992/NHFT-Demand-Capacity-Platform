# Forecasting (SARIMA)

This folder contains the forecasting utilities for the NHFT Demand-Capacity Platform.

## What it does

- Converts raw row-level data to a monthly time series
- Fits a Seasonal ARIMA (SARIMA / SARIMAX) model
- Produces multi-step monthly forecasts (e.g., 6-month and 12-month) with **95% confidence intervals**
- Returns outputs in a simple DataFrame format that is easy to plot in Plotly/Dash

## Key files

- `preprocessing.py`
  - `build_monthly_series(df, value_col, date_col="period_end")`
- `sarima.py`
  - `SarimaForecaster` (fit + forecast with intervals)
  - `small_grid_search_aic(...)` (optional lightweight order search)
- `forecast.py`
  - `make_forecast_frame(y, config=...)` (history + future in one DataFrame)

## Quick example

```python
import pandas as pd
from forecasting.preprocessing import build_monthly_series
from forecasting.forecast import make_forecast_frame, ForecastConfig

# df must have: period_end (datetime-like), referrals (numeric)
y = build_monthly_series(df, "referrals", date_col="period_end")

# 6-month forecast with 95% CI
frame_6 = make_forecast_frame(y, config=ForecastConfig(months_ahead=6, conf_level=0.95))

# 12-month forecast with 95% CI
frame_12 = make_forecast_frame(y, config=ForecastConfig(months_ahead=12, conf_level=0.95))
```

## Notes

- The default seasonal period is 12 (monthly seasonality).
- For interactive dashboards, keep `auto_select=False` (fast). You can run auto-selection offline.
