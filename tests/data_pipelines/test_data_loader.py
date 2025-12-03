"""
Tests for the data pipelines module.
"""

import pandas as pd

from demand_capacity_platform.data_pipelines import DataLoader


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_generate_sample_referrals(self):
        """Test generation of sample referral data."""
        loader = DataLoader()
        df = loader.generate_sample_referrals(n=100, seed=42)

        assert len(df) == 100
        assert "referral_id" in df.columns
        assert "referral_date" in df.columns
        assert "service" in df.columns
        assert "priority" in df.columns
        assert df["referral_date"].dtype == "datetime64[ns]"

    def test_generate_sample_referrals_with_custom_services(self):
        """Test referral generation with custom services."""
        loader = DataLoader()
        services = ["Service A", "Service B"]
        df = loader.generate_sample_referrals(n=50, services=services, seed=42)

        assert set(df["service"].unique()).issubset(set(services))

    def test_generate_sample_activity(self):
        """Test generation of sample activity data."""
        loader = DataLoader()
        df = loader.generate_sample_activity(n=200, seed=42)

        assert len(df) == 200
        assert "activity_id" in df.columns
        assert "activity_date" in df.columns
        assert "duration_mins" in df.columns
        assert "attended" in df.columns

    def test_generate_sample_capacity(self):
        """Test generation of sample capacity data."""
        loader = DataLoader()
        df = loader.generate_sample_capacity()

        assert "service" in df.columns
        assert "staff_fte" in df.columns
        assert "available_hours_weekly" in df.columns
        assert "avg_appointment_mins" in df.columns
        assert len(df) > 0

    def test_prepare_timeseries(self, sample_referrals):
        """Test time series preparation from DataFrame."""
        loader = DataLoader()
        ts = loader.prepare_timeseries(sample_referrals, "referral_date")

        assert isinstance(ts, pd.Series)
        assert isinstance(ts.index, pd.DatetimeIndex)
        assert ts.name == "value"
        assert len(ts) > 0

    def test_prepare_timeseries_with_value_column(self, sample_referrals):
        """Test time series preparation with value column."""
        loader = DataLoader()
        sample_referrals["value"] = 1
        ts = loader.prepare_timeseries(
            sample_referrals,
            "referral_date",
            value_column="value"
        )

        assert isinstance(ts, pd.Series)
        assert len(ts) > 0

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        loader = DataLoader()

        df1 = loader.generate_sample_referrals(n=100, seed=42)
        df2 = loader.generate_sample_referrals(n=100, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        loader = DataLoader()

        df1 = loader.generate_sample_referrals(n=100, seed=42)
        df2 = loader.generate_sample_referrals(n=100, seed=123)

        # Should have different values (not exactly equal)
        assert not df1.equals(df2)
