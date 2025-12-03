"""
Base Forecaster Module.

Provides abstract base class for all forecasting models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """
    Container for forecast results.

    Attributes:
        forecast: Point forecasts.
        lower_bound: Lower confidence interval.
        upper_bound: Upper confidence interval.
        confidence_level: Confidence level for intervals.
        model_name: Name of the forecasting model.
        fit_metrics: Dictionary of fit metrics (MAE, RMSE, etc.).
    """
    forecast: pd.Series
    lower_bound: Optional[pd.Series] = None
    upper_bound: Optional[pd.Series] = None
    confidence_level: float = 0.95
    model_name: str = "Unknown"
    fit_metrics: Optional[dict] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast result to a DataFrame."""
        data = {"forecast": self.forecast}
        if self.lower_bound is not None:
            data["lower_bound"] = self.lower_bound
        if self.upper_bound is not None:
            data["upper_bound"] = self.upper_bound
        return pd.DataFrame(data)


class BaseForecaster(ABC):
    """
    Abstract base class for forecasting models.

    All forecasting models should inherit from this class
    and implement the fit and predict methods.
    """

    def __init__(self, name: str = "BaseForecaster"):
        """
        Initialize the forecaster.

        Args:
            name: Name of the forecaster.
        """
        self.name = name
        self._is_fitted = False
        self._train_data: Optional[pd.Series] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(self, y: pd.Series) -> "BaseForecaster":
        """
        Fit the model to the time series data.

        Args:
            y: Time series data with datetime index.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
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
        pass

    def fit_predict(
        self,
        y: pd.Series,
        steps: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Fit the model and generate forecasts.

        Args:
            y: Time series data with datetime index.
            steps: Number of periods to forecast.
            confidence_level: Confidence level for prediction intervals.

        Returns:
            ForecastResult with forecasts and confidence intervals.
        """
        self.fit(y)
        return self.predict(steps, confidence_level)

    @staticmethod
    def calculate_metrics(
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray]
    ) -> dict:
        """
        Calculate forecast accuracy metrics.

        Args:
            actual: Actual values.
            predicted: Predicted values.

        Returns:
            Dictionary with MAE, RMSE, MAPE metrics.
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        errors = actual - predicted

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # MAPE (avoid division by zero)
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / actual[non_zero_mask])) * 100
        else:
            mape = np.nan

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }

    def _validate_input(self, y: pd.Series) -> pd.Series:
        """
        Validate and prepare input time series.

        Args:
            y: Input time series.

        Returns:
            Validated time series.

        Raises:
            ValueError: If input is invalid.
        """
        if not isinstance(y, pd.Series):
            raise ValueError("Input must be a pandas Series")

        if len(y) < 10:
            raise ValueError("Time series must have at least 10 observations")

        # Ensure datetime index
        if not isinstance(y.index, pd.DatetimeIndex):
            try:
                y.index = pd.to_datetime(y.index)
            except Exception:
                raise ValueError("Index must be convertible to datetime")

        # Sort by index
        y = y.sort_index()

        # Handle missing values
        if y.isna().any():
            y = y.interpolate(method="linear").bfill().ffill()

        return y
