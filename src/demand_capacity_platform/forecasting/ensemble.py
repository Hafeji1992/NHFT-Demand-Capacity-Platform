"""
Ensemble Forecaster Module.

Combines multiple forecasting models for improved predictions.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from demand_capacity_platform.forecasting.arima import ARIMAForecaster
from demand_capacity_platform.forecasting.base import BaseForecaster, ForecastResult
from demand_capacity_platform.forecasting.ets import ETSForecaster
from demand_capacity_platform.forecasting.ml_forecaster import MLForecaster

logger = logging.getLogger(__name__)


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster combining multiple models.

    Combines ARIMA, ETS, and ML models using weighted averaging
    based on cross-validation performance.

    Attributes:
        models: List of forecaster instances.
        weights: Weights for combining forecasts.
        method: Ensemble method ("mean", "weighted", "median").

    Example:
        >>> forecaster = EnsembleForecaster()
        >>> result = forecaster.fit_predict(time_series, steps=30)
        >>> print(result.forecast)
    """

    def __init__(
        self,
        models: Optional[list[BaseForecaster]] = None,
        weights: Optional[list[float]] = None,
        method: str = "weighted"
    ):
        """
        Initialize the ensemble forecaster.

        Args:
            models: List of forecaster instances.
            weights: Weights for combining forecasts.
            method: Ensemble method ("mean", "weighted", "median").
        """
        super().__init__(name="Ensemble")

        self.models = models or [
            ARIMAForecaster(auto=True),
            ETSForecaster(auto=True),
            MLForecaster()
        ]
        self.weights = weights
        self.method = method
        self._model_results: list[ForecastResult] = []

    def _calculate_cv_weights(self, y: pd.Series) -> list[float]:
        """
        Calculate weights based on cross-validation performance.

        Args:
            y: Time series data.

        Returns:
            List of weights for each model.
        """
        # Use last 20% of data for validation
        split_idx = int(len(y) * 0.8)
        train = y.iloc[:split_idx]
        test = y.iloc[split_idx:]
        test_steps = len(test)

        errors = []

        for model in self.models:
            try:
                model.fit(train)
                result = model.predict(test_steps)

                # Calculate RMSE on test set
                forecast_aligned = result.forecast.reindex(test.index)
                if forecast_aligned.isna().all():
                    # If index alignment fails, use positional comparison
                    pred_values = result.forecast.values[:len(test)]
                    rmse = np.sqrt(np.mean((test.values - pred_values) ** 2))
                else:
                    rmse = np.sqrt(np.mean((test - forecast_aligned) ** 2))

                errors.append(rmse)
            except Exception as e:
                logger.warning("Model %s failed CV: %s", model.name, str(e))
                errors.append(np.inf)

        # Convert errors to weights (lower error = higher weight)
        errors = np.array(errors)

        # Handle infinite errors
        finite_mask = np.isfinite(errors)
        if not np.any(finite_mask):
            # All models failed, use equal weights
            return [1.0 / len(self.models)] * len(self.models)

        # Set infinite errors to max finite error * 10
        max_finite = np.max(errors[finite_mask])
        errors[~finite_mask] = max_finite * 10

        # Inverse error weighting
        inverse_errors = 1.0 / errors
        weights = inverse_errors / inverse_errors.sum()

        logger.info("CV weights: %s", dict(zip([m.name for m in self.models], weights)))

        return list(weights)

    def fit(self, y: pd.Series) -> "EnsembleForecaster":
        """
        Fit all models to the time series data.

        Args:
            y: Time series data with datetime index.

        Returns:
            Self for method chaining.
        """
        y = self._validate_input(y)
        self._train_data = y

        logger.info("Fitting ensemble with %d models", len(self.models))

        # Calculate weights if using weighted method
        if self.method == "weighted" and self.weights is None:
            self.weights = self._calculate_cv_weights(y)

        # Fit all models on full data
        for model in self.models:
            try:
                model.fit(y)
                logger.info("Fitted model: %s", model.name)
            except Exception as e:
                logger.warning("Failed to fit model %s: %s", model.name, str(e))

        self._is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate ensemble forecasts for future periods.

        Args:
            steps: Number of periods to forecast.
            confidence_level: Confidence level for prediction intervals.

        Returns:
            ForecastResult with forecasts and confidence intervals.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = []
        lower_bounds = []
        upper_bounds = []
        valid_weights = []
        model_names = []
        all_metrics = []

        for i, model in enumerate(self.models):
            try:
                if not model.is_fitted:
                    continue

                result = model.predict(steps, confidence_level)
                forecasts.append(result.forecast)

                if result.lower_bound is not None:
                    lower_bounds.append(result.lower_bound)
                if result.upper_bound is not None:
                    upper_bounds.append(result.upper_bound)

                weight = self.weights[i] if self.weights else 1.0 / len(self.models)
                valid_weights.append(weight)
                model_names.append(result.model_name)
                all_metrics.append(result.fit_metrics)

                self._model_results.append(result)

            except Exception as e:
                logger.warning("Model %s prediction failed: %s", model.name, str(e))

        if not forecasts:
            raise ValueError("All models failed to produce forecasts")

        # Combine forecasts
        forecasts_df = pd.DataFrame({f"m{i}": f for i, f in enumerate(forecasts)})

        if self.method == "mean":
            combined_forecast = forecasts_df.mean(axis=1)
        elif self.method == "median":
            combined_forecast = forecasts_df.median(axis=1)
        else:  # weighted
            # Normalize weights
            norm_weights = np.array(valid_weights) / np.sum(valid_weights)
            combined_forecast = (forecasts_df * norm_weights).sum(axis=1)

        # Combine confidence intervals
        if lower_bounds:
            lower_df = pd.DataFrame({f"m{i}": lb for i, lb in enumerate(lower_bounds)})
            upper_df = pd.DataFrame({f"m{i}": ub for i, ub in enumerate(upper_bounds)})

            if self.method == "mean":
                combined_lower = lower_df.mean(axis=1)
                combined_upper = upper_df.mean(axis=1)
            elif self.method == "median":
                combined_lower = lower_df.min(axis=1)
                combined_upper = upper_df.max(axis=1)
            else:  # weighted
                combined_lower = (lower_df * norm_weights).sum(axis=1)
                combined_upper = (upper_df * norm_weights).sum(axis=1)
        else:
            combined_lower = None
            combined_upper = None

        # Aggregate metrics
        combined_metrics = {}
        for metric in ["MAE", "RMSE", "MAPE"]:
            values = [m[metric] for m in all_metrics if m and metric in m and not np.isnan(m[metric])]
            if values:
                combined_metrics[metric] = np.mean(values)

        return ForecastResult(
            forecast=combined_forecast,
            lower_bound=combined_lower,
            upper_bound=combined_upper,
            confidence_level=confidence_level,
            model_name=f"Ensemble({'+'.join(model_names)})",
            fit_metrics=combined_metrics
        )

    def get_individual_forecasts(self) -> list[ForecastResult]:
        """
        Get forecasts from individual models.

        Returns:
            List of ForecastResult objects from each model.
        """
        return self._model_results
