"""NHFT ETS (Exponential Smoothing) Forecasting
=============================================

A lightweight, dashboard-friendly forecasting backend based on Holt-Winters
Exponential Smoothing (ETS).

Compared with SARIMA, ETS often converges more reliably on short monthly series
(~24-36 points) and is faster to fit for interactive use.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass(frozen=True)
class EtsSpec:
    """Model specification for Holt-Winters ETS."""

    trend: Optional[str] = "add"  # "add" or None
    seasonal: Optional[str] = "add"  # "add" or None
    seasonal_periods: int = 12
    damped_trend: bool = True


class EtsForecaster:
    """Fit an ETS model and produce forecasts with approximate intervals."""

    def __init__(self, *, spec: EtsSpec = EtsSpec()):
        self.spec = spec
        self._model = None
        self._results = None
        self._y: Optional[pd.Series] = None

    @property
    def is_fitted(self) -> bool:
        return self._results is not None

    def fit(self, y: pd.Series) -> "EtsForecaster":
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("y must have a DatetimeIndex")

        y = pd.to_numeric(y, errors="coerce").astype(float)
        y = y.sort_index()
        if y.isna().any():
            raise ValueError(
                "y contains NaN values; fill missing months before fitting"
            )

        # Encourage a stable monthly index.
        if y.index.freq is None:
            y = y.asfreq("ME")

        self._y = y

        self._model = ExponentialSmoothing(
            y,
            trend=self.spec.trend,
            seasonal=self.spec.seasonal,
            seasonal_periods=(
                int(self.spec.seasonal_periods) if self.spec.seasonal else None
            ),
            damped_trend=bool(self.spec.damped_trend) if self.spec.trend else False,
            initialization_method="estimated",
        )

        # optimized=True triggers parameter estimation; this is generally fast for ETS.
        self._results = self._model.fit(optimized=True)
        return self

    def forecast(self, steps: int, *, conf_level: float = 0.95) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit(y) first.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if not (0.0 < conf_level < 1.0):
            raise ValueError("conf_level must be in (0, 1)")

        yhat = pd.Series(self._results.forecast(steps=steps)).astype(float)

        # Build a month-end index for outputs (statsmodels often preserves index but be safe).
        if isinstance(yhat.index, pd.DatetimeIndex) and yhat.index.freq is not None:
            idx = yhat.index
        else:
            last = self._y.index.max()
            idx = pd.date_range(
                last + pd.offsets.MonthEnd(1), periods=int(steps), freq="ME"
            )
            yhat.index = idx

        # Approximate prediction intervals:
        # use residual std dev and widen with sqrt(h).
        resid = pd.Series(getattr(self._results, "resid", pd.Series([], dtype=float)))
        resid = pd.to_numeric(resid, errors="coerce").dropna().astype(float)
        sigma = (
            float(np.nanstd(resid.values, ddof=1))
            if len(resid) >= 3
            else float(np.nanstd(self._y.values))
        )
        sigma = float(sigma) if np.isfinite(sigma) else 0.0

        z = float(NormalDist().inv_cdf(0.5 + (float(conf_level) / 2.0)))
        h = np.arange(1, int(steps) + 1, dtype=float)
        se = sigma * np.sqrt(h)

        lower = yhat.values - z * se
        upper = yhat.values + z * se

        # Counts should not be negative.
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)

        out = pd.DataFrame(
            {
                "yhat": yhat.values,
                "yhat_lower": lower,
                "yhat_upper": upper,
            },
            index=idx,
        )

        return out


def small_grid_search_aic_ets(
    y: pd.Series,
    *,
    seasonal_period: int = 12,
    trend: Iterable[Optional[str]] = ("add", None),
    seasonal: Iterable[Optional[str]] = ("add", None),
    damped_trend: Iterable[bool] = (True, False),
) -> EtsSpec:
    """Lightweight ETS spec search using AIC.

    ETS generally converges more reliably than SARIMA on short series.
    This search is intentionally small to keep dashboard interactions responsive.
    """

    y = pd.to_numeric(y, errors="coerce").astype(float)
    y = y.sort_index()
    if y.index.freq is None:
        y = y.asfreq("ME")

    best_aic = np.inf
    best_spec = EtsSpec(
        trend=None,
        seasonal=None,
        seasonal_periods=int(seasonal_period),
        damped_trend=False,
    )

    for t in trend:
        for s in seasonal:
            for dmp in damped_trend:
                if t is None and dmp:
                    continue
                sp = int(seasonal_period) if s is not None else int(seasonal_period)
                spec = EtsSpec(
                    trend=t, seasonal=s, seasonal_periods=sp, damped_trend=bool(dmp)
                )
                try:
                    model = ExponentialSmoothing(
                        y,
                        trend=spec.trend,
                        seasonal=spec.seasonal,
                        seasonal_periods=(
                            int(spec.seasonal_periods) if spec.seasonal else None
                        ),
                        damped_trend=bool(spec.damped_trend) if spec.trend else False,
                        initialization_method="estimated",
                    )
                    res = model.fit(optimized=True)
                    aic = float(getattr(res, "aic", np.inf))
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_spec = spec
                except Exception:
                    continue

    return best_spec
