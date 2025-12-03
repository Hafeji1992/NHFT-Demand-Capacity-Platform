"""
ARIMA Forecaster Module.

Provides ARIMA (AutoRegressive Integrated Moving Average) forecasting.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from demand_capacity_platform.forecasting.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecaster for time series prediction.

    Implements Auto-ARIMA functionality for automatic parameter selection
    or allows manual specification of (p, d, q) parameters.

    Attributes:
        order: ARIMA order (p, d, q).
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).

    Example:
        >>> forecaster = ARIMAForecaster(order=(1, 1, 1))
        >>> result = forecaster.fit_predict(time_series, steps=30)
        >>> print(result.forecast)
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto: bool = True
    ):
        """
        Initialize the ARIMA forecaster.

        Args:
            order: ARIMA order (p, d, q). If None and auto=True, will be determined automatically.
            seasonal_order: Seasonal ARIMA order (P, D, Q, s).
            auto: Whether to automatically determine order.
        """
        super().__init__(name="ARIMA")
        self.order = order or (1, 1, 1)
        self.seasonal_order = seasonal_order
        self.auto = auto
        self._model = None
        self._model_fit = None

    def _determine_d(self, y: pd.Series) -> int:
        """
        Determine the differencing order using ADF test.

        Args:
            y: Time series data.

        Returns:
            Differencing order (0, 1, or 2).
        """
        max_d = 2
        d = 0

        for i in range(max_d + 1):
            if i > 0:
                y = y.diff().dropna()

            try:
                result = adfuller(y, autolag="AIC")
                p_value = result[1]

                if p_value < 0.05:  # Stationary
                    d = i
                    break
            except Exception:
                d = 1
                break
        else:
            d = max_d

        return d

    def _select_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA order using AIC criterion.

        Args:
            y: Time series data.

        Returns:
            Best (p, d, q) order.
        """
        d = self._determine_d(y)

        best_aic = np.inf
        best_order = (1, d, 1)

        p_range = range(0, 4)
        q_range = range(0, 4)

        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                except Exception:
                    continue

        logger.info("Selected ARIMA order: %s with AIC: %.2f", best_order, best_aic)
        return best_order

    def fit(self, y: pd.Series) -> "ARIMAForecaster":
        """
        Fit the ARIMA model to the time series data.

        Args:
            y: Time series data with datetime index.

        Returns:
            Self for method chaining.
        """
        y = self._validate_input(y)
        self._train_data = y

        if self.auto and self.order == (1, 1, 1):
            self.order = self._select_order(y)

        logger.info("Fitting ARIMA%s model", self.order)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self._model = SARIMAX(
                    y,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self._model = ARIMA(
                    y,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

            self._model_fit = self._model.fit()

        self._is_fitted = True
        logger.info("ARIMA model fitted. AIC: %.2f", self._model_fit.aic)

        return self

    def predict(
        self,
        steps: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecasts for future periods.

        Args:
            steps: Number of periods to forecast.
            confidence_level: Confidence level for prediction intervals.

        Returns:
            ForecastResult with forecasts and confidence intervals.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        alpha = 1 - confidence_level

        forecast = self._model_fit.get_forecast(steps=steps)

        # Get forecast values
        pred_mean = forecast.predicted_mean

        # Get confidence intervals
        conf_int = forecast.conf_int(alpha=alpha)

        # Create proper datetime index for forecasts
        last_date = self._train_data.index[-1]
        freq = pd.infer_freq(self._train_data.index) or "D"
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else "d"),
            periods=steps,
            freq=freq
        )

        pred_mean.index = forecast_index
        lower_bound = pd.Series(conf_int.iloc[:, 0].values, index=forecast_index)
        upper_bound = pd.Series(conf_int.iloc[:, 1].values, index=forecast_index)

        # Calculate in-sample metrics
        fitted_values = self._model_fit.fittedvalues
        metrics = self.calculate_metrics(self._train_data, fitted_values)

        return ForecastResult(
            forecast=pred_mean,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            model_name=f"ARIMA{self.order}",
            fit_metrics=metrics
        )

    def get_model_summary(self) -> str:
        """Get a summary of the fitted model."""
        if not self._is_fitted:
            return "Model not fitted"
        return str(self._model_fit.summary())
