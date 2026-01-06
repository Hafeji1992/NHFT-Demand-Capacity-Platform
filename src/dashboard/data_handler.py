"""
Data Handler Module for NHFT Demand-Capacity Dashboard
========================================================
Handles loading, preprocessing, and providing data for the Dash application.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Data Handler Class
# ---------------------------------------------------------------------
class DataHandler:
    """Manages data loading and preprocessing for the dashboard."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialise the data handler.

        Args:
            data_dir: Directory containing CSV files. If None, uses project data/ folder.
        """
        if data_dir is None:
            # Assume dashboard is in src/dashboard, so go up 2 levels to project root
            project_root = Path(__file__).resolve().parents[2]
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        self.patient_df: Optional[pd.DataFrame] = None
        self.staffing_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

        logger.info(f"📁 Data directory: {self.data_dir}")

    # -----------------------------------------------------------------
    # Data Loading Methods
    # -----------------------------------------------------------------
    def load_patient_data(self, filename: str = "patient_data.csv") -> pd.DataFrame:
        """
        Load patient data from CSV.

        Args:
            filename: Name of the patient data CSV file

        Returns:
            pandas.DataFrame: Patient data

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Patient data not found at: {filepath}")

        logger.info(f"📂 Loading patient data from: {filepath}")

        self.patient_df = pd.read_csv(filepath, parse_dates=["periodend"])

        logger.info(f"✅ Loaded {len(self.patient_df):,} patient records")

        return self.patient_df

    def load_staffing_data(self, filename: str = "staffing_data.csv") -> pd.DataFrame:
        """
        Load staffing data from CSV.

        Args:
            filename: Name of the staffing data CSV file

        Returns:
            pandas.DataFrame: Staffing data

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Staffing data not found at: {filepath}")

        logger.info(f"📂 Loading staffing data from: {filepath}")

        self.staffing_df = pd.read_csv(filepath)

        logger.info(f"✅ Loaded {len(self.staffing_df):,} staffing records")

        return self.staffing_df

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both patient and staffing data.

        Returns:
            Tuple of (patient_df, staffing_df)
        """
        self.load_patient_data()
        self.load_staffing_data()

        return self.patient_df, self.staffing_df

    # -----------------------------------------------------------------
    # Data Preprocessing Methods
    # -----------------------------------------------------------------
    def preprocess_patient_data(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Preprocess patient data for dashboard use.

        Args:
            df: DataFrame to preprocess. If None, uses loaded patient_df.

        Returns:
            Preprocessed DataFrame
        """
        if df is None:
            if self.patient_df is None:
                raise ValueError(
                    "No patient data loaded. Call load_patient_data() first."
                )
            df = self.patient_df.copy()
        else:
            df = df.copy()

        # Ensure periodend is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["periodend"]):
            df["periodend"] = pd.to_datetime(df["periodend"])

        # Extract temporal features
        df["year"] = df["periodend"].dt.year
        df["month"] = df["periodend"].dt.month
        df["month_name"] = df["periodend"].dt.strftime("%b")
        df["quarter"] = df["periodend"].dt.quarter
        df["year_month"] = df["periodend"].dt.to_period("M").astype(str)

        # Sort by date
        df = df.sort_values("periodend")

        logger.info("✅ Patient data preprocessed")

        return df

    def preprocess_staffing_data(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Preprocess staffing data for dashboard use.

        Args:
            df: DataFrame to preprocess. If None, uses loaded staffing_df.

        Returns:
            Preprocessed DataFrame
        """
        if df is None:
            if self.staffing_df is None:
                raise ValueError(
                    "No staffing data loaded. Call load_staffing_data() first."
                )
            df = self.staffing_df.copy()
        else:
            df = df.copy()

        # Ensure consistent data types
        df["staff"] = pd.to_numeric(df["staff"], errors="coerce")

        # Remove any rows with null staff counts
        df = df.dropna(subset=["staff"])

        logger.info("✅ Staffing data preprocessed")

        return df

    # -----------------------------------------------------------------
    # Data Merging Methods
    # -----------------------------------------------------------------
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge patient and staffing data on provider code and service line.

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If data hasn't been loaded
        """
        if self.patient_df is None or self.staffing_df is None:
            raise ValueError("Both datasets must be loaded before merging.")

        logger.info("🔗 Merging patient and staffing datasets...")

        # Aggregate staffing by provider and service line
        staffing_agg = (
            self.staffing_df.groupby(["providercodecurrent", "service_line"])
            .agg({"staff": "sum"})
            .reset_index()
        )

        # Merge with patient data
        self.merged_df = self.patient_df.merge(
            staffing_agg, on=["providercodecurrent", "service_line"], how="left"
        )

        logger.info(f"✅ Merged dataset contains {len(self.merged_df):,} records")

        return self.merged_df

    # -----------------------------------------------------------------
    # Data Filtering Methods
    # -----------------------------------------------------------------
    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: DataFrame to filter
            start_date: Start date (inclusive). Format: 'YYYY-MM-DD'
            end_date: End date (inclusive). Format: 'YYYY-MM-DD'

        Returns:
            Filtered DataFrame
        """
        if "periodend" not in df.columns:
            raise ValueError("DataFrame must have 'periodend' column")

        filtered = df.copy()

        if start_date:
            filtered = filtered[filtered["periodend"] >= pd.to_datetime(start_date)]

        if end_date:
            filtered = filtered[filtered["periodend"] <= pd.to_datetime(end_date)]

        return filtered

    def filter_by_provider(
        self, df: pd.DataFrame, providers: List[str]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by provider codes.

        Args:
            df: DataFrame to filter
            providers: List of provider codes

        Returns:
            Filtered DataFrame
        """
        if not providers:
            return df

        return df[df["providercodecurrent"].isin(providers)]

    def filter_by_service_line(
        self, df: pd.DataFrame, service_lines: List[str]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by service lines.

        Args:
            df: DataFrame to filter
            service_lines: List of service lines

        Returns:
            Filtered DataFrame
        """
        if not service_lines:
            return df

        return df[df["service_line"].isin(service_lines)]

    # -----------------------------------------------------------------
    # Data Aggregation Methods
    # -----------------------------------------------------------------
    def aggregate_by_period(
        self, df: pd.DataFrame, metrics: List[str], period: str = "month"
    ) -> pd.DataFrame:
        """
        Aggregate metrics by time period.

        Args:
            df: DataFrame to aggregate
            metrics: List of metric columns to sum
            period: Time period ('month', 'quarter', 'year')

        Returns:
            Aggregated DataFrame
        """
        if period == "month":
            group_col = "year_month"
        elif period == "quarter":
            group_col = "quarter"
        elif period == "year":
            group_col = "year"
        else:
            raise ValueError(f"Invalid period: {period}")

        # Ensure temporal columns exist
        if group_col not in df.columns:
            df = self.preprocess_patient_data(df)

        agg_dict = {metric: "sum" for metric in metrics if metric in df.columns}

        return df.groupby(group_col).agg(agg_dict).reset_index()

    def aggregate_by_service_line(
        self, df: pd.DataFrame, metrics: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate metrics by service line.

        Args:
            df: DataFrame to aggregate
            metrics: List of metric columns to sum

        Returns:
            Aggregated DataFrame
        """
        agg_dict = {metric: "sum" for metric in metrics if metric in df.columns}

        return df.groupby("service_line").agg(agg_dict).reset_index()

    # -----------------------------------------------------------------
    # Getter Methods for Dashboard
    # -----------------------------------------------------------------
    def get_unique_providers(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Get list of unique provider codes."""
        if df is None:
            df = self.patient_df if self.patient_df is not None else self.staffing_df

        if df is None:
            return []

        return sorted(df["providercodecurrent"].dropna().unique().tolist())

    def get_unique_service_lines(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Get list of unique service lines."""
        if df is None:
            df = self.patient_df if self.patient_df is not None else self.staffing_df

        if df is None:
            return []

        return sorted(df["service_line"].dropna().unique().tolist())

    def get_date_range(self, df: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
        """Get min and max dates from patient data."""
        if df is None:
            df = self.patient_df

        if df is None or "periodend" not in df.columns:
            return None, None

        min_date = df["periodend"].min().strftime("%Y-%m-%d")
        max_date = df["periodend"].max().strftime("%Y-%m-%d")

        return min_date, max_date

    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate summary statistics for the dashboard.

        Args:
            df: DataFrame to summarise. If None, uses patient_df.

        Returns:
            Dictionary of summary statistics
        """
        if df is None:
            df = self.patient_df

        if df is None:
            return {}

        # Get latest period data
        latest_period = df["periodend"].max()
        latest_data = df[df["periodend"] == latest_period]

        summary = {
            "total_referrals": int(latest_data["referrals"].sum()),
            "total_waiters": int(latest_data["waiters"].sum()),
            "total_caseload": int(latest_data["caseload"].sum()),
            "total_contacts": int(latest_data["totalcontacts"].sum()),
            "avg_wait_18plus_pct": (
                latest_data["waiters18plusweeks"].sum()
                / latest_data["waiters"].sum()
                * 100
                if latest_data["waiters"].sum() > 0
                else 0
            ),
            "latest_period": latest_period.strftime("%Y-%m-%d"),
        }

        return summary


# ---------------------------------------------------------------------
# Standalone Usage Functions
# ---------------------------------------------------------------------
def get_data_handler(data_dir: Optional[Path] = None) -> DataHandler:
    """
    Factory function to create and initialise a DataHandler.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Initialised DataHandler instance
    """
    handler = DataHandler(data_dir)

    try:
        handler.load_all_data()
        handler.patient_df = handler.preprocess_patient_data()
        handler.staffing_df = handler.preprocess_staffing_data()
        logger.info("✅ Data handler initialised successfully")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        raise

    return handler


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Test the data handler
    try:
        handler = get_data_handler()

        print("\n" + "=" * 70)
        print("DATA HANDLER TEST")
        print("=" * 70)

        # Test patient data
        print(f"\n📊 Patient Data Shape: {handler.patient_df.shape}")
        print(f"Date Range: {handler.get_date_range()}")

        # Test staffing data
        print(f"\n👥 Staffing Data Shape: {handler.staffing_df.shape}")

        # Test unique values
        print(f"\n🏥 Unique Providers: {len(handler.get_unique_providers())}")
        print(f"📋 Unique Service Lines: {len(handler.get_unique_service_lines())}")

        # Test summary statistics
        print("\n📈 Summary Statistics:")
        for key, value in handler.get_summary_statistics().items():
            print(f"  {key}: {value}")

        print("\n✅ All tests passed!")

    except Exception as e:
        logger.exception(f"❌ Test failed: {e}")
