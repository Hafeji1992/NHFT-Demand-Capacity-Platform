"""
Test configuration and fixtures.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_referrals():
    """Generate sample referral data for testing."""
    np.random.seed(42)
    n = 500

    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    data = {
        "referral_id": range(1, n + 1),
        "referral_date": np.random.choice(date_range, n),
        "service": np.random.choice(
            ["Adult Mental Health", "CAMHS", "Community Nursing"],
            n,
            p=[0.5, 0.3, 0.2],
        ),
        "priority": np.random.choice(["Urgent", "Routine", "Soon"], n, p=[0.15, 0.65, 0.20]),
        "wait_days": np.random.exponential(scale=14, size=n).astype(int),
        "status": np.random.choice(["Accepted", "Rejected", "Pending"], n, p=[0.75, 0.10, 0.15]),
    }

    df = pd.DataFrame(data)
    df["referral_date"] = pd.to_datetime(df["referral_date"])
    return df.sort_values("referral_date").reset_index(drop=True)


@pytest.fixture
def sample_capacity():
    """Generate sample capacity data for testing."""
    return pd.DataFrame({
        "service": ["Adult Mental Health", "CAMHS", "Community Nursing"],
        "staff_fte": [10.0, 6.0, 8.0],
        "available_hours_weekly": [300.0, 180.0, 240.0],
        "avg_appointment_mins": [50, 60, 45],
    })


@pytest.fixture
def sample_timeseries():
    """Generate a sample time series for testing."""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")

    # Generate data with trend and seasonality
    trend = np.linspace(5, 10, 365)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
    noise = np.random.normal(0, 1, 365)

    values = trend + seasonal + noise
    values = np.maximum(0, values)  # Ensure non-negative

    return pd.Series(values, index=dates, name="demand")
