"""Forecasting package (SARIMA) for the NHFT Demand-Capacity Platform.

This package focuses on *monthly* time series forecasting with 95% confidence
intervals, suitable for extending the dashboard time-series charts.

Primary entry points:
- `SarimaForecaster` (fit/forecast)
- `build_monthly_series` (aggregate raw rows to a monthly series)
- `make_forecast_frame` (history + forecast + intervals in one DataFrame)
"""

from .preprocessing import build_monthly_series
from .sarima import SarimaForecaster
from .forecast import make_forecast_frame

__all__ = [
    "SarimaForecaster",
    "build_monthly_series",
    "make_forecast_frame",
]
