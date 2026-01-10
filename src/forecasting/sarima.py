from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

Order = Tuple[int, int, int]
SeasonalOrder = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SarimaSpec:
    """Model specification for SARIMA/SARIMAX."""

    order: Order = (1, 1, 1)
    seasonal_order: SeasonalOrder = (1, 1, 1, 12)
    trend: str = "n"  # 'n' (none) is a safe default for counts


class SarimaForecaster:
    """Fit a SARIMA model and produce multi-step forecasts with intervals."""

    def __init__(
        self,
        *,
        spec: SarimaSpec = SarimaSpec(),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.spec = spec
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self._model = None
        self._results = None
        self._y: Optional[pd.Series] = None

    @property
    def is_fitted(self) -> bool:
        return self._results is not None

    def fit(self, y: pd.Series) -> "SarimaForecaster":
        """Fit SARIMA to a monthly series.

        Args:
            y: Monthly series with DatetimeIndex and no missing values.

        Returns:
            self
        """
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("y must have a DatetimeIndex")

        y = pd.to_numeric(y, errors="coerce").astype(float)
        y = y.sort_index()

        if y.isna().any():
            raise ValueError(
                "y contains NaN values; fill missing months before fitting"
            )

        # Encourage statsmodels to keep a monthly DatetimeIndex in forecast outputs.
        if y.index.freq is None:
            y = y.asfreq("ME")

        self._y = y

        self._model = SARIMAX(
            y,
            order=self.spec.order,
            seasonal_order=self.spec.seasonal_order,
            trend=self.spec.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        # disp=False to avoid console spam
        self._results = self._model.fit(disp=False)
        return self

    def forecast(self, steps: int, *, conf_level: float = 0.95) -> pd.DataFrame:
        """Forecast future values with confidence intervals.

        Args:
            steps: Number of months ahead.
            conf_level: Confidence level, e.g. 0.95 for 95% interval.

        Returns:
            DataFrame with index as forecast months and columns:
            - yhat
            - yhat_lower
            - yhat_upper
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit(y) first.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if not (0.0 < conf_level < 1.0):
            raise ValueError("conf_level must be in (0, 1)")

        alpha = 1.0 - conf_level
        pred = self._results.get_forecast(steps=steps)
        mean = pred.predicted_mean
        conf = pred.conf_int(alpha=alpha)

        # statsmodels labels columns like 'lower y' / 'upper y' sometimes.
        lower_col = conf.columns[0]
        upper_col = conf.columns[1]

        out = pd.DataFrame(
            {
                "yhat": mean.astype(float),
                "yhat_lower": conf[lower_col].astype(float),
                "yhat_upper": conf[upper_col].astype(float),
            }
        )

        return out


def small_grid_search_aic(
    y: pd.Series,
    *,
    seasonal_period: int = 12,
    p: Iterable[int] = (0, 1, 2),
    d: Iterable[int] = (0, 1),
    q: Iterable[int] = (0, 1, 2),
    P: Iterable[int] = (0, 1),
    D: Iterable[int] = (0, 1),
    Q: Iterable[int] = (0, 1),
    trend: str = "n",
) -> SarimaSpec:
    """Lightweight SARIMA order search using AIC.

    This is intentionally small to keep dashboard interactions responsive.
    For production-grade tuning, you would typically run this offline.
    """

    best_aic = np.inf
    best_spec = SarimaSpec()

    y = pd.to_numeric(y, errors="coerce").astype(float)
    y = y.sort_index()
    if y.index.freq is None:
        y = y.asfreq("ME")

    for pi in p:
        for di in d:
            for qi in q:
                for Pi in P:
                    for Di in D:
                        for Qi in Q:
                            order = (int(pi), int(di), int(qi))
                            seasonal_order = (
                                int(Pi),
                                int(Di),
                                int(Qi),
                                int(seasonal_period),
                            )
                            try:
                                model = SARIMAX(
                                    y,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    trend=trend,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                res = model.fit(disp=False)
                                if np.isfinite(res.aic) and res.aic < best_aic:
                                    best_aic = float(res.aic)
                                    best_spec = SarimaSpec(
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        trend=trend,
                                    )
                            except Exception:
                                # Some parameter combos fail to converge; skip.
                                continue

    return best_spec
