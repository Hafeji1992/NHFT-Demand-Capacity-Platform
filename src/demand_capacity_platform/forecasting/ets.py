"""
ETS (Exponential Smoothing) Forecaster Module.

Provides Exponential Smoothing State Space Model forecasting.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demand_capacity_platform.forecasting.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class ETSForecaster(BaseForecaster):
    """
    Exponential Smoothing (ETS) forecaster for time series prediction.

    Implements Holt-Winters Exponential Smoothing with support for
    trend and seasonal components.

    Attributes:
        trend: Type of trend component ("add", "mul", or None).
        seasonal: Type of seasonal component ("add", "mul", or None).
        seasonal_periods: Number of periods in seasonal cycle.
        damped_trend: Whether to damp the trend.

    Example:
        >>> forecaster = ETSForecaster(trend="add", seasonal="add", seasonal_periods=7)
        >>> result = forecaster.fit_predict(time_series, steps=30)
        >>> print(result.forecast)
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        auto: bool = True
    ):
        """
        Initialize the ETS forecaster.

        Args:
            trend: Type of trend ("add", "mul", or None).
            seasonal: Type of seasonal component ("add", "mul", or None).
            seasonal_periods: Number of periods in seasonal cycle.
            damped_trend: Whether to apply trend damping.
            auto: Whether to automatically select components.
        """
        super().__init__(name="ETS")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.auto = auto
        self._model = None
        self._model_fit = None

    def _select_model(self, y: pd.Series) -> dict:
        """
        Automatically select ETS model components.

        Args:
            y: Time series data.

        Returns:
            Dictionary with best model parameters.
        """
        best_aic = np.inf
        best_params = {
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False
        }

        trends = [None, "add"]
        damped_options = [False, True]

        # Detect potential seasonality
        if len(y) >= 14:
            # Try weekly seasonality
            seasonal_periods = [None, 7]
            seasonals = [None, "add"]
        else:
            seasonal_periods = [None]
            seasonals = [None]

        for trend in trends:
            for damped in damped_options:
                if trend is None and damped:
                    continue

                for seasonal in seasonals:
                    for sp in seasonal_periods:
                        if seasonal is not None and sp is None:
                            continue
                        if seasonal is None and sp is not None:
                            continue
                        if sp is not None and len(y) < 2 * sp:
                            continue

                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model = ExponentialSmoothing(
                                    y,
                                    trend=trend,
                                    seasonal=seasonal,
                                    seasonal_periods=sp,
                                    damped_trend=damped if trend else False,
                                    initialization_method="estimated"
                                )
                                fitted = model.fit(optimized=True)

                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_params = {
                                        "trend": trend,
                                        "seasonal": seasonal,
                                        "seasonal_periods": sp,
                                        "damped_trend": damped if trend else False
                                    }
                        except Exception:
                            continue

        logger.info("Selected ETS model: %s with AIC: %.2f", best_params, best_aic)
        return best_params

    def fit(self, y: pd.Series) -> "ETSForecaster":
        """
        Fit the ETS model to the time series data.

        Args:
            y: Time series data with datetime index.

        Returns:
            Self for method chaining.
        """
        y = self._validate_input(y)
        self._train_data = y

        if self.auto:
            params = self._select_model(y)
            self.trend = params["trend"]
            self.seasonal = params["seasonal"]
            self.seasonal_periods = params["seasonal_periods"]
            self.damped_trend = params["damped_trend"]

        logger.info(
            "Fitting ETS model (trend=%s, seasonal=%s, periods=%s, damped=%s)",
            self.trend, self.seasonal, self.seasonal_periods, self.damped_trend
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend if self.trend else False,
                initialization_method="estimated"
            )

            self._model_fit = self._model.fit(optimized=True)

        self._is_fitted = True
        logger.info("ETS model fitted. AIC: %.2f", self._model_fit.aic)

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

        # Generate forecast
        forecast = self._model_fit.forecast(steps)

        # Create proper datetime index for forecasts
        last_date = self._train_data.index[-1]
        freq = pd.infer_freq(self._train_data.index) or "D"
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else "d"),
            periods=steps,
            freq=freq
        )

        forecast.index = forecast_index

        # Calculate prediction intervals (simulation-based)
        sim_results = self._model_fit.simulate(
            nsimulations=steps,
            repetitions=1000,
            anchor="end"
        )

        alpha = (1 - confidence_level) / 2
        lower_bound = pd.Series(
            np.percentile(sim_results, alpha * 100, axis=1),
            index=forecast_index
        )
        upper_bound = pd.Series(
            np.percentile(sim_results, (1 - alpha) * 100, axis=1),
            index=forecast_index
        )

        # Calculate in-sample metrics
        fitted_values = self._model_fit.fittedvalues
        metrics = self.calculate_metrics(self._train_data, fitted_values)

        # Build model name
        model_name = "ETS("
        model_name += "A" if self.trend == "add" else ("M" if self.trend == "mul" else "N")
        model_name += "d" if self.damped_trend else ""
        model_name += ","
        model_name += "A" if self.seasonal == "add" else ("M" if self.seasonal == "mul" else "N")
        model_name += ")"

        return ForecastResult(
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            model_name=model_name,
            fit_metrics=metrics
        )
