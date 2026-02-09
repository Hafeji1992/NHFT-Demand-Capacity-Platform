"""
NHFT Staffing Data Ingestion
=============================
Extracts staffing capacity metrics from ESR and writes a cleaned CSV for analysis and the dashboard.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

try:
    # Prefer package import to avoid collisions with site-packages (e.g. `connect.py`)
    from data_engineering.connect import SQLServerConnection
except ImportError:  # pragma: no cover
    # Fallback for running this file directly
    from connect import SQLServerConnection

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------
class DataQualityException(Exception):
    """Custom exception for data quality failures."""

    pass


# ---------------------------------------------------------------------
# Staffing Data Extractor
# ---------------------------------------------------------------------
class StaffingDataExtractor:
    """Extracts staffing capacity data from ESR and saves to CSV."""

    # SQL query as class constant for better maintainability
    STAFFING_QUERY = """
		-- ============================================================================
        -- STAFFING DATA EXTRACTION QUERY
        -- ============================================================================
        -- Purpose: Extract staffing capacity metrics from ESR appraisal review data,
        -- pulling staff counts by provider, service line, and staff group.
        -- ============================================================================
        SELECT
            SUBSTRING(ARD.[Org L6], CHARINDEX('L5 ', ARD.[Org L6]) + 3, 3) AS [ProviderCodeCurrent], -- Provider code: the 3 digits after 'L5 '
			SL.[Service_Line],
            ARD.[Staff Group],
            COUNT(DISTINCT ARD.[Assignment Number]) AS [Staff]

        FROM [ISEVSQLMIS-BLK].[ESR].[dbo].[tbl_dt_Appraisal_Review_Detail] AS ARD
		LEFT JOIN [MIS_Config].[dbo].[tbl_org_current_RL9_Service_Line] AS SL ON SUBSTRING(ARD.[Org L6], CHARINDEX('L5 ', ARD.[Org L6]) + 3, 3) = SL.[Service_Codes]
		
		WHERE SL.[Status] = 'ACTIVE'
		AND SL.[RTT_Report_Enabled] = 1 -- RTT Reporting Only Services

        GROUP BY
            ARD.[Org L6],
			SL.[Service_Line],
            ARD.[Staff Group]

        ORDER BY
            ARD.[Org L6],
            SL.[Service_Line],
            ARD.[Staff Group];
    """

    # Expected columns after normalisation
    EXPECTED_COLUMNS = [
        "provider_code_current",
        "service_line",
        "staff_group",
        "staff",
    ]

    # -----------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        run_quality_checks: bool = True,
    ):
        """
        Initialise SQL Server connection and output directory.

        Args:
            config_path: Path to config.ini file (optional, auto-detected if None)
            output_dir: Directory to save CSV files (defaults to data/)
            run_quality_checks: Whether to run data quality checks (default: True)
        """
        self.sql_conn = SQLServerConnection(config_path)
        self.conn = self.sql_conn.connect()
        self.run_quality_checks = run_quality_checks

        if not self.conn:
            raise ConnectionError("Failed to connect to the SQL Server database.")

        # Set up output directory using pathlib for better path handling
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            self.output_dir = project_root / "data"
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")

    # -----------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------
    @staticmethod
    def _normalise_column_names(columns: list[str]) -> list[str]:
        """
        Convert column names to lowercase with underscores.

        Args:
            columns: List of column names

        Returns:
            List of normalised column names
        """

        def to_snake_case(name: str) -> str:
            # Replace common symbols with readable tokens
            name = name.strip()

            # Convert CamelCase/PascalCase (including acronyms) to snake_case
            name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
            name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

            # Split letter/digit boundaries (e.g., Waiters18 -> Waiters_18)
            name = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", name)
            name = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", name)

            # Normalise separators
            name = name.replace(" ", "_")
            name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
            name = re.sub(r"_+", "_", name).strip("_").lower()

            return name

        return [to_snake_case(col) for col in columns]

    # -----------------------------------------------------------------
    # Schema Validation
    # -----------------------------------------------------------------
    def _check_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if DataFrame has expected columns.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for missing columns
        missing = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {', '.join(sorted(missing))}")

        # Check for unexpected columns
        unexpected = set(df.columns) - set(self.EXPECTED_COLUMNS)
        if unexpected:
            issues.append(f"Unexpected columns: {', '.join(sorted(unexpected))}")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Data Completeness Checks
    # -----------------------------------------------------------------
    def _check_data_completeness(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for missing values in critical columns.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Critical columns that should not have nulls
        critical_cols = ["provider_code_current", "service_line", "staff_group"]

        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100

                if null_count > 0:
                    issues.append(
                        f"{col}: {null_count:,} missing values ({null_pct:.1f}%)"
                    )

        # Check for completely null rows
        completely_null = df.isnull().all(axis=1).sum()
        if completely_null > 0:
            issues.append(f"Found {completely_null:,} completely empty rows")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Data Type Checks
    # -----------------------------------------------------------------
    def _check_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if data types are appropriate.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check staff is numeric
        if "staff" in df.columns and not pd.api.types.is_numeric_dtype(df["staff"]):
            issues.append(f"staff is not numeric type (found: {df['staff'].dtype})")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Data Range Checks
    # -----------------------------------------------------------------
    def _check_data_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if numeric values are within reasonable ranges.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Staff count should not be negative
        if "staff" in df.columns:
            negative = (df["staff"] < 0).sum()
            if negative > 0:
                issues.append(f"staff: {negative:,} negative values found")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------
    def _check_duplicates(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for duplicate records.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for duplicate combinations of provider, service_line, and staff_group
        key_cols = ["provider_code_current", "service_line", "staff_group"]

        if all(col in df.columns for col in key_cols):
            dupes = df.duplicated(subset=key_cols, keep=False).sum()
            if dupes > 0:
                issues.append(
                    f"Found {dupes:,} duplicate rows based on " f"{', '.join(key_cols)}"
                )

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Data Quality Checks
    # -----------------------------------------------------------------
    def run_data_quality_checks(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run all data quality checks and return results.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with check results
        """
        logger.info("🔍 Running data quality checks...")

        checks = {
            "Schema Validation": self._check_schema,
            "Data Completeness": self._check_data_completeness,
            "Data Types": self._check_data_types,
            "Data Ranges": self._check_data_ranges,
            "Duplicate Detection": self._check_duplicates,
        }

        results = {"passed": [], "failed": [], "issues": {}}

        for name, func in checks.items():
            valid, issues = func(df)

            if valid:
                results["passed"].append(name)
                logger.info(f"  ✅ {name}: PASSED")
            else:
                results["failed"].append(name)
                results["issues"][name] = issues
                logger.warning(f"  ⚠️ {name}: FAILED")
                for issue in issues:
                    logger.warning(f"     - {issue}")

        # Summary
        total_checks = len(checks)
        passed_checks = len(results["passed"])
        failed_checks = len(results["failed"])

        logger.info(
            f"\n📊 Quality Check Summary: {passed_checks}/{total_checks} passed"
        )

        if failed_checks > 0:
            logger.warning(f"⚠️ {failed_checks} check(s) failed - review issues above")
        else:
            logger.info("✅ All data quality checks passed!")

        return results

    # -----------------------------------------------------------------
    # Data Extraction
    # -----------------------------------------------------------------
    def extract_staffing_data(self) -> pd.DataFrame:
        """
        Extract staffing data from ESR.

        Returns:
            pandas.DataFrame: The extracted data

        Raises:
            Exception: If data extraction fails
            DataQualityException: If quality checks fail
        """
        logger.info("📊 Fetching staffing data from ESR...")

        try:
            cursor = self.conn.cursor()
            cursor.execute(self.STAFFING_QUERY)

            # Fetch all rows
            logger.info("⏳ Fetching rows...")
            rows = cursor.fetchall()
            total_rows = len(rows)

            if total_rows == 0:
                logger.warning("⚠️ No data retrieved from database")
                return pd.DataFrame()

            # Get column names
            columns = [c[0] for c in cursor.description]

            logger.info(f"📊 Processing {total_rows:,} rows...")

            # Create DataFrame with progress bar
            with tqdm(total=total_rows, desc="Loading data", unit="rows") as pbar:
                df = pd.DataFrame.from_records(rows, columns=columns)
                pbar.update(total_rows)

            # Normalise column names
            df.columns = self._normalise_column_names(df.columns)

            logger.info(f"✅ Retrieved {len(df):,} rows from source views.")
            logger.info(
                f"📊 Columns: {', '.join(df.columns)} ({len(df.columns)} total)"
            )

            # Run data quality checks
            if self.run_quality_checks:
                quality_results = self.run_data_quality_checks(df)

                # Optionally raise exception if critical checks fail
                critical_checks = ["Schema Validation", "Data Types"]
                failed_critical = [
                    c for c in critical_checks if c in quality_results["failed"]
                ]

                if failed_critical:
                    raise DataQualityException(
                        f"Critical data quality checks failed: {', '.join(failed_critical)}"
                    )

            return df

        except Exception as e:
            logger.error(f"❌ Error extracting data: {str(e)}")
            raise
        finally:
            if "cursor" in locals():
                cursor.close()

    # -----------------------------------------------------------------
    # Data Persistence
    # -----------------------------------------------------------------
    def save_to_csv(
        self, df: pd.DataFrame, filename: str = "staffing_data.csv"
    ) -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: pandas DataFrame to save
            filename: Name of the CSV file (default: staffing_data.csv)

        Returns:
            Path: Path to the saved CSV file
        """
        if df.empty:
            logger.warning("⚠️ DataFrame is empty. No file saved.")
            return None

        filepath = self.output_dir / filename
        logger.info(f"💾 Saving data to CSV: {filepath}")

        try:
            with tqdm(total=1, desc="Writing CSV", unit="file") as pbar:
                df.to_csv(filepath, index=False)
                pbar.update(1)

            file_size_kb = filepath.stat().st_size / 1024
            logger.info(f"✅ Data saved to: {filepath}")
            logger.info(f"📄 File size: {file_size_kb:.2f} KB")
            logger.info(f"📊 Rows: {len(df):,} | Columns: {len(df.columns)}")

            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving CSV: {str(e)}")
            raise

    # -----------------------------------------------------------------
    # Connection Management
    # -----------------------------------------------------------------
    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.sql_conn.close()
            logger.info("✅ Database connection closed.")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()


# ---------------------------------------------------------------------
# Utility Loader
# ---------------------------------------------------------------------
def load_staffing_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Utility function to load staffing data from CSV.

    Args:
        csv_path: Path to CSV file. If None, loads staffing_data.csv
                  from default location.

    Returns:
        pandas.DataFrame: The staffing data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    if csv_path is None:
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "data" / "staffing_data.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Staffing data file not found at: {csv_path}")

    logger.info(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


# ---------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Use context manager for automatic connection cleanup
    try:
        with StaffingDataExtractor() as extractor:
            # Extract data from ESR
            df = extractor.extract_staffing_data()

            # Save as staffing_data.csv
            output_file = extractor.save_to_csv(df)

            if output_file:
                logger.info("🎉 Staffing data extraction completed successfully!")
                logger.info(f"📂 File saved: {output_file}")

    except DataQualityException as e:
        logger.error(f"❌ Data quality validation failed: {e}")
        logger.error("⚠️ Data was not saved due to quality issues")
    except Exception:
        logger.exception("❌ Error occurred during staffing data extraction")
