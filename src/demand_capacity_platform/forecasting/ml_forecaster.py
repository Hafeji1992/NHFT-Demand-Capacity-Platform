"""
Machine Learning Forecaster Module.

Provides ML-based time series forecasting using scikit-learn models.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from demand_capacity_platform.forecasting.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class MLForecaster(BaseForecaster):
    """
    Machine Learning forecaster for time series prediction.

    Uses lagged features and optional exogenous variables to train
    ML models for forecasting.

    Attributes:
        model: Scikit-learn estimator to use.
        lags: Number of lag features to create.
        include_date_features: Whether to include date-based features.

    Example:
        >>> forecaster = MLForecaster(model=RandomForestRegressor(), lags=14)
        >>> result = forecaster.fit_predict(time_series, steps=30)
        >>> print(result.forecast)
    """

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        lags: int = 7,
        include_date_features: bool = True,
        scale_features: bool = True
    ):
        """
        Initialize the ML forecaster.

        Args:
            model: Scikit-learn estimator. Defaults to GradientBoostingRegressor.
            lags: Number of lag features to create.
            include_date_features: Whether to include date-based features.
            scale_features: Whether to scale features.
        """
        super().__init__(name="ML")
        self.model = model or GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.lags = lags
        self.include_date_features = include_date_features
        self.scale_features = scale_features
        self._scaler = StandardScaler() if scale_features else None
        self._feature_names: list = []

    def _create_lag_features(self, y: pd.Series) -> pd.DataFrame:
        """
        Create lag features from time series.

        Args:
            y: Time series data.

        Returns:
            DataFrame with lag features.
        """
        df = pd.DataFrame({"y": y})

        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Rolling statistics
        df["rolling_mean_7"] = df["y"].shift(1).rolling(window=7, min_periods=1).mean()
        df["rolling_std_7"] = df["y"].shift(1).rolling(window=7, min_periods=1).std()
        df["rolling_mean_14"] = df["y"].shift(1).rolling(window=14, min_periods=1).mean()

        return df

    def _create_date_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create date-based features.

        Args:
            index: Datetime index.

        Returns:
            DataFrame with date features.
        """
        df = pd.DataFrame(index=index)

        df["day_of_week"] = index.dayofweek
        df["day_of_month"] = index.day
        df["month"] = index.month
        df["quarter"] = index.quarter
        df["is_weekend"] = (index.dayofweek >= 5).astype(int)
        df["is_month_start"] = index.is_month_start.astype(int)
        df["is_month_end"] = index.is_month_end.astype(int)

        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _prepare_features(
        self,
        y: pd.Series,
        for_prediction: bool = False
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for training or prediction.

        Args:
            y: Time series data.
            for_prediction: If True, don't drop rows with NaN.

        Returns:
            Tuple of (features DataFrame, target Series).
        """
        # Create lag features
        df = self._create_lag_features(y)

        # Add date features
        if self.include_date_features:
            date_features = self._create_date_features(y.index)
            df = pd.concat([df, date_features], axis=1)

        # Extract target
        target = df["y"] if not for_prediction else None
        df = df.drop(columns=["y"])

        if not for_prediction:
            # Drop rows with NaN values (from lagging)
            valid_idx = df.dropna().index
            df = df.loc[valid_idx]
            target = target.loc[valid_idx]

        self._feature_names = list(df.columns)

        return df, target

    def fit(self, y: pd.Series) -> "MLForecaster":
        """
        Fit the ML model to the time series data.

        Args:
            y: Time series data with datetime index.

        Returns:
            Self for method chaining.
        """
        y = self._validate_input(y)
        self._train_data = y

        logger.info(
            "Fitting ML model (%s) with %d lags",
            type(self.model).__name__,
            self.lags
        )

        X, target = self._prepare_features(y)

        if self.scale_features and self._scaler is not None:
            X_scaled = pd.DataFrame(
                self._scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            X_scaled = X

        self.model.fit(X_scaled, target)

        self._is_fitted = True
        logger.info("ML model fitted with %d features", len(self._feature_names))

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

        # Extend the series with forecasts iteratively
        extended_series = self._train_data.copy()

        last_date = self._train_data.index[-1]
        freq = pd.infer_freq(self._train_data.index) or "D"

        predictions = []

        for i in range(steps):
            # Create features for next step
            next_date = last_date + pd.Timedelta(i + 1, unit=freq[0].lower() if freq else "d")

            # Extend with current predictions
            if predictions:
                pred_index = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else "d"),
                    periods=len(predictions),
                    freq=freq
                )
                temp_series = pd.concat([
                    extended_series,
                    pd.Series(predictions, index=pred_index)
                ])
            else:
                temp_series = extended_series

            # Create lag features for current prediction
            lag_values = temp_series.iloc[-self.lags:].values[::-1]
            rolling_7 = temp_series.iloc[-7:].mean()
            rolling_std_7 = temp_series.iloc[-7:].std()
            rolling_14 = temp_series.iloc[-14:].mean() if len(temp_series) >= 14 else temp_series.mean()

            features = list(lag_values) + [rolling_7, rolling_std_7, rolling_14]

            if self.include_date_features:
                date_features = self._create_date_features(pd.DatetimeIndex([next_date]))
                features.extend(date_features.values[0])

            X_pred = pd.DataFrame([features], columns=self._feature_names)

            if self.scale_features and self._scaler is not None:
                X_pred = pd.DataFrame(
                    self._scaler.transform(X_pred),
                    columns=X_pred.columns
                )

            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)

        # Create forecast series
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else "d"),
            periods=steps,
            freq=freq
        )

        forecast = pd.Series(predictions, index=forecast_index)

        # Estimate prediction intervals using residuals
        train_preds = self._get_fitted_values()
        residuals = self._train_data.loc[train_preds.index] - train_preds
        residual_std = residuals.std()

        z_score = 1.96 if confidence_level == 0.95 else 1.645

        # Widen intervals for further horizons
        horizon_multiplier = np.sqrt(np.arange(1, steps + 1))

        lower_bound = pd.Series(
            forecast.values - z_score * residual_std * horizon_multiplier,
            index=forecast_index
        )
        upper_bound = pd.Series(
            forecast.values + z_score * residual_std * horizon_multiplier,
            index=forecast_index
        )

        # Calculate in-sample metrics
        metrics = self.calculate_metrics(
            self._train_data.loc[train_preds.index],
            train_preds
        )

        return ForecastResult(
            forecast=forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            model_name=f"ML({type(self.model).__name__})",
            fit_metrics=metrics
        )

    def _get_fitted_values(self) -> pd.Series:
        """Get in-sample fitted values."""
        X, _ = self._prepare_features(self._train_data)

        if self.scale_features and self._scaler is not None:
            X = pd.DataFrame(
                self._scaler.transform(X),
                index=X.index,
                columns=X.columns
            )

        fitted = self.model.predict(X)
        return pd.Series(fitted, index=X.index)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if the model supports it.

        Returns:
            Series with feature importances or None.
        """
        if not self._is_fitted:
            return None

        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_,
                index=self._feature_names
            ).sort_values(ascending=False)
        elif hasattr(self.model, "coef_"):
            return pd.Series(
                np.abs(self.model.coef_),
                index=self._feature_names
            ).sort_values(ascending=False)

        return None
