"""
Data Loader Module.

Provides utilities for loading data from various sources
including CSV files, Excel files, and generating sample data.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Utility class for loading and generating healthcare data.

    This class provides methods to load data from files and
    generate sample data for testing and demonstration.

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_csv("referrals.csv")
        >>> sample = loader.generate_sample_referrals(n=1000)
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Base directory for data files.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        logger.info("DataLoader initialized with data_dir: %s", self.data_dir)

    def load_csv(
        self,
        filename: str,
        parse_dates: Optional[list[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filename: Name of the CSV file.
            parse_dates: List of columns to parse as dates.
            **kwargs: Additional arguments for pd.read_csv.

        Returns:
            DataFrame with loaded data.
        """
        filepath = self.data_dir / filename if self.data_dir else Path(filename)
        logger.info("Loading CSV from: %s", filepath)
        return pd.read_csv(filepath, parse_dates=parse_dates, **kwargs)

    def load_excel(
        self,
        filename: str,
        sheet_name: Optional[Union[str, int]] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from an Excel file.

        Args:
            filename: Name of the Excel file.
            sheet_name: Name or index of sheet to load.
            **kwargs: Additional arguments for pd.read_excel.

        Returns:
            DataFrame with loaded data.
        """
        filepath = self.data_dir / filename if self.data_dir else Path(filename)
        logger.info("Loading Excel from: %s", filepath)
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def generate_sample_referrals(
        n: int = 1000,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        services: Optional[list[str]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate sample referral data for testing.

        Args:
            n: Number of referrals to generate.
            start_date: Start date for referrals.
            end_date: End date for referrals.
            services: List of service names.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with sample referral data.
        """
        if seed is not None:
            np.random.seed(seed)

        if services is None:
            services = [
                "Adult Mental Health",
                "CAMHS",
                "Community Nursing",
                "Physiotherapy",
                "Occupational Therapy"
            ]

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        data = {
            "referral_id": range(1, n + 1),
            "referral_date": np.random.choice(date_range, n),
            "service": np.random.choice(services, n),
            "priority": np.random.choice(["Urgent", "Routine", "Soon"], n, p=[0.15, 0.65, 0.20]),
            "wait_days": np.random.exponential(scale=14, size=n).astype(int),
            "status": np.random.choice(
                ["Accepted", "Rejected", "Pending"], n, p=[0.75, 0.10, 0.15]
            )
        }

        df = pd.DataFrame(data)
        df["referral_date"] = pd.to_datetime(df["referral_date"])
        return df.sort_values("referral_date").reset_index(drop=True)

    @staticmethod
    def generate_sample_activity(
        n: int = 5000,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        services: Optional[list[str]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate sample activity/appointment data for testing.

        Args:
            n: Number of activities to generate.
            start_date: Start date for activities.
            end_date: End date for activities.
            services: List of service names.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with sample activity data.
        """
        if seed is not None:
            np.random.seed(seed)

        if services is None:
            services = [
                "Adult Mental Health",
                "CAMHS",
                "Community Nursing",
                "Physiotherapy",
                "Occupational Therapy"
            ]

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        data = {
            "activity_id": range(1, n + 1),
            "activity_date": np.random.choice(date_range, n),
            "service": np.random.choice(services, n),
            "activity_type": np.random.choice(
                ["Assessment", "Follow-up", "Group", "Home Visit"],
                n,
                p=[0.20, 0.50, 0.15, 0.15]
            ),
            "duration_mins": np.random.choice([30, 45, 60, 90], n, p=[0.25, 0.35, 0.30, 0.10]),
            "attended": np.random.choice([True, False], n, p=[0.88, 0.12])
        }

        df = pd.DataFrame(data)
        df["activity_date"] = pd.to_datetime(df["activity_date"])
        return df.sort_values("activity_date").reset_index(drop=True)

    @staticmethod
    def generate_sample_capacity(
        services: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample capacity data for testing.

        Args:
            services: List of service names.

        Returns:
            DataFrame with sample capacity data.
        """
        if services is None:
            services = [
                "Adult Mental Health",
                "CAMHS",
                "Community Nursing",
                "Physiotherapy",
                "Occupational Therapy"
            ]

        data = {
            "service": services,
            "staff_fte": [10.5, 6.0, 15.0, 8.0, 5.5],
            "available_hours_weekly": [367.5, 210.0, 525.0, 280.0, 192.5],
            "avg_appointment_mins": [50, 60, 45, 40, 55]
        }

        return pd.DataFrame(data)

    @staticmethod
    def prepare_timeseries(
        df: pd.DataFrame,
        date_column: str,
        value_column: Optional[str] = None,
        freq: str = "D"
    ) -> pd.Series:
        """
        Prepare a time series from a DataFrame.

        Args:
            df: Input DataFrame.
            date_column: Name of the date column.
            value_column: Name of the value column (if None, counts rows).
            freq: Frequency for resampling.

        Returns:
            Time series indexed by date.
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        if value_column:
            ts = df[value_column].resample(freq).sum()
        else:
            ts = df.resample(freq).size()

        # Fill missing values with 0
        ts = ts.asfreq(freq, fill_value=0)
        ts.name = "value"

        return ts
