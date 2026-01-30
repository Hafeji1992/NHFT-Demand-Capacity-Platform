"""Forecasting package for the NHFT Demand-Capacity Platform.

This package focuses on *monthly* time series forecasting with 95% confidence
intervals, suitable for extending the dashboard time-series charts.

Primary entry points:
- `EtsForecaster` (ETS/Holt-Winters fit/forecast) — dashboard default
- `build_monthly_series` (aggregate raw rows to a monthly series)
- `make_ets_forecast_frame` (ETS history + forecast + intervals)

SARIMA is kept for reference under `forecasting.archived_sarima`.
"""

from .preprocessing import build_monthly_series
from .ets import EtsForecaster
from .forecast import make_forecast_frame, make_ets_forecast_frame
from .archived_sarima.sarima import SarimaForecaster

__all__ = [
    "SarimaForecaster",
    "EtsForecaster",
    "build_monthly_series",
    "make_forecast_frame",
    "make_ets_forecast_frame",
]
