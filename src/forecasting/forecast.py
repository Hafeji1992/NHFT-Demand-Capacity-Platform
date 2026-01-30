"""
NHFT Forecast Frame Builder
==========================
Combines observed history with SARIMA/SARIMAX forecasts and confidence
intervals into a single plot-ready DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .preprocessing import coerce_monthly_series
from .sarima import SarimaForecaster, SarimaSpec, small_grid_search_aic


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for generating a monthly SARIMA forecast."""

    months_ahead: int = 6
    conf_level: float = 0.95
    freq: str = "ME"
    auto_select: bool = False
    seasonal_period: int = 12


def make_forecast_frame(
    y: pd.Series,
    *,
    config: ForecastConfig = ForecastConfig(),
    spec: Optional[SarimaSpec] = None,
) -> pd.DataFrame:
    """Build a single DataFrame containing history + forecast + confidence intervals.

    Output is designed to be easy to plot in Plotly:
    - plot actuals: `y` where `is_forecast=False`
    - plot forecast line: `yhat` where `is_forecast=True`
    - plot interval band using `yhat_lower` and `yhat_upper`

    Args:
        y: Monthly series with DatetimeIndex.
        config: Forecast configuration (horizon, CI, etc.).
        spec: Optional fixed SARIMA spec. If None, uses defaults, or auto-select.

    Returns:
        DataFrame with columns:
        - period_end (datetime)
        - y (actual)
        - yhat, yhat_lower, yhat_upper (forecast components; NaN for history rows)
        - is_forecast (bool)
    """

    y2 = coerce_monthly_series(y, freq=config.freq, fill_missing="zero")

    if spec is None:
        if config.auto_select:
            spec = small_grid_search_aic(y2, seasonal_period=config.seasonal_period)
        else:
            spec = SarimaSpec(
                order=(1, 1, 1), seasonal_order=(1, 1, 1, config.seasonal_period)
            )

    model = SarimaForecaster(spec=spec).fit(y2)
    fc = model.forecast(config.months_ahead, conf_level=config.conf_level)

    history = pd.DataFrame(
        {
            "period_end": y2.index,
            "y": y2.values,
            "yhat": pd.NA,
            "yhat_lower": pd.NA,
            "yhat_upper": pd.NA,
            "is_forecast": False,
        }
    )

    future = pd.DataFrame(
        {
            "period_end": fc.index,
            "y": pd.NA,
            "yhat": fc["yhat"].values,
            "yhat_lower": fc["yhat_lower"].values,
            "yhat_upper": fc["yhat_upper"].values,
            "is_forecast": True,
        }
    )

    out = pd.concat([history, future], ignore_index=True)

    # Ensure types are Plotly-friendly
    out["period_end"] = pd.to_datetime(out["period_end"])
    for col in ["y", "yhat", "yhat_lower", "yhat_upper"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
