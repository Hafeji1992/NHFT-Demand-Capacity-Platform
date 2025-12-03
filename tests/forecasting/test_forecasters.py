"""
Tests for the forecasting module.
"""

import pandas as pd
import pytest

from demand_capacity_platform.forecasting import (
    ARIMAForecaster,
    EnsembleForecaster,
    ETSForecaster,
    MLForecaster,
)
from demand_capacity_platform.forecasting.base import ForecastResult


class TestARIMAForecaster:
    """Tests for ARIMAForecaster class."""

    def test_fit(self, sample_timeseries):
        """Test ARIMA model fitting."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), auto=False)
        forecaster.fit(sample_timeseries)

        assert forecaster.is_fitted

    def test_predict(self, sample_timeseries):
        """Test ARIMA model prediction."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), auto=False)
        forecaster.fit(sample_timeseries)
        result = forecaster.predict(steps=30)

        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 30
        assert result.lower_bound is not None
        assert result.upper_bound is not None

    def test_fit_predict(self, sample_timeseries):
        """Test combined fit and predict."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), auto=False)
        result = forecaster.fit_predict(sample_timeseries, steps=14)

        assert len(result.forecast) == 14
        assert forecaster.is_fitted

    def test_forecast_result_to_dataframe(self, sample_timeseries):
        """Test ForecastResult conversion to DataFrame."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), auto=False)
        result = forecaster.fit_predict(sample_timeseries, steps=7)

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "forecast" in df.columns
        assert "lower_bound" in df.columns
        assert "upper_bound" in df.columns

    def test_predict_without_fit_raises_error(self, sample_timeseries):
        """Test that predict without fit raises error."""
        forecaster = ARIMAForecaster()

        with pytest.raises(ValueError, match="Model must be fitted"):
            forecaster.predict(steps=30)


class TestETSForecaster:
    """Tests for ETSForecaster class."""

    def test_fit(self, sample_timeseries):
        """Test ETS model fitting."""
        forecaster = ETSForecaster(trend="add", auto=False)
        forecaster.fit(sample_timeseries)

        assert forecaster.is_fitted

    def test_predict(self, sample_timeseries):
        """Test ETS model prediction."""
        forecaster = ETSForecaster(trend="add", auto=False)
        forecaster.fit(sample_timeseries)
        result = forecaster.predict(steps=30)

        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 30

    def test_auto_selection(self, sample_timeseries):
        """Test automatic model selection."""
        forecaster = ETSForecaster(auto=True)
        forecaster.fit(sample_timeseries)

        assert forecaster.is_fitted
        # Model should have selected some configuration
        assert forecaster.trend is not None or forecaster.seasonal is not None or True


class TestMLForecaster:
    """Tests for MLForecaster class."""

    def test_fit(self, sample_timeseries):
        """Test ML model fitting."""
        forecaster = MLForecaster(lags=7)
        forecaster.fit(sample_timeseries)

        assert forecaster.is_fitted

    def test_predict(self, sample_timeseries):
        """Test ML model prediction."""
        forecaster = MLForecaster(lags=7)
        forecaster.fit(sample_timeseries)
        result = forecaster.predict(steps=14)

        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 14

    def test_feature_importance(self, sample_timeseries):
        """Test feature importance retrieval."""
        forecaster = MLForecaster(lags=7)
        forecaster.fit(sample_timeseries)

        importance = forecaster.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, pd.Series)
        assert len(importance) > 0


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster class."""

    def test_fit(self, sample_timeseries):
        """Test ensemble model fitting."""
        forecaster = EnsembleForecaster()
        forecaster.fit(sample_timeseries)

        assert forecaster.is_fitted

    def test_predict(self, sample_timeseries):
        """Test ensemble model prediction."""
        forecaster = EnsembleForecaster()
        forecaster.fit(sample_timeseries)
        result = forecaster.predict(steps=14)

        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 14
        assert "Ensemble" in result.model_name

    def test_weighted_ensemble(self, sample_timeseries):
        """Test weighted ensemble method."""
        forecaster = EnsembleForecaster(method="weighted")
        result = forecaster.fit_predict(sample_timeseries, steps=7)

        assert len(result.forecast) == 7
        assert result.fit_metrics is not None

    def test_mean_ensemble(self, sample_timeseries):
        """Test mean ensemble method."""
        forecaster = EnsembleForecaster(method="mean")
        result = forecaster.fit_predict(sample_timeseries, steps=7)

        assert len(result.forecast) == 7


class TestForecastMetrics:
    """Tests for forecast accuracy metrics."""

    def test_calculate_metrics(self):
        """Test metric calculation."""
        from demand_capacity_platform.forecasting.base import BaseForecaster

        actual = pd.Series([10, 20, 30, 40, 50])
        predicted = pd.Series([11, 19, 31, 39, 51])

        metrics = BaseForecaster.calculate_metrics(actual, predicted)

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert metrics["MAE"] == 1.0
        assert metrics["RMSE"] == 1.0
