import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from data_engineering.connect import SQLServerConnection

# Configure logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataQualityException(Exception):
    """Custom exception for data quality failures."""
    pass


class StaffingDataExtractor:
    """
    Extracts staffing counts by Service Line (Org L6) and Staff Group from ESR and saves to CSV.
    """

    STAFFING_QUERY = """
        SELECT
            COUNT(DISTINCT [Assignment Number]) AS [Staff],
            [Staff Group],
            [Org L6] AS [Service_Line]
        FROM [ISEVSQLMIS-BLK].[ESR].[dbo].[tbl_dt_Appraisal_Review_Detail]
        GROUP BY [Org L6], [Staff Group]
        ORDER BY [Org L6];
    """

    # Expected columns after normalisation
    EXPECTED_COLUMNS = [
        "staff",
        "staff_group",
        "service_line"
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        run_quality_checks: bool = True
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

        # Default output directory: project_root/data
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            self.output_dir = project_root / "data"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")

    @staticmethod
    def _normalise_column_names(columns: list[str]) -> list[str]:
        """
        Convert column names to lowercase with underscores.
        """
        return [
            col.lower()
            .strip()
            .replace(" ", "_")
            .replace("<", "under")
            .replace(">", "over")
            .replace("+", "plus")
            for col in columns
        ]

    def _check_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if DataFrame has expected columns.
        """
        issues = []

        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {', '.join(sorted(missing_cols))}")

        unexpected_cols = set(df.columns) - set(self.EXPECTED_COLUMNS)
        if unexpected_cols:
            issues.append(f"Unexpected columns: {', '.join(sorted(unexpected_cols))}")

        return len(issues) == 0, issues

    def _check_data_completeness(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for missing values in critical columns.
        """
        issues = []
        critical_cols = ["service_line", "staff_group"]

        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"{col}: {null_count:,} missing values")

        return len(issues) == 0, issues

    def _check_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate expected data types.
        """
        issues = []

        if "staff" in df.columns and not pd.api.types.is_numeric_dtype(df["staff"]):
            issues.append(f"staff is not numeric type (found: {df['staff'].dtype})")

        return len(issues) == 0, issues

    def _check_data_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Staff counts should not be negative.
        """
        issues = []

        if "staff" in df.columns:
            negative_count = (df["staff"] < 0).sum()
            if negative_count > 0:
                issues.append(f"staff: {negative_count:,} negative values found")

        return len(issues) == 0, issues

    def _check_duplicates(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check duplicates by (service_line, staff_group).
        """
        issues = []
        key_cols = ["service_line", "staff_group"]

        if all(col in df.columns for col in key_cols):
            duplicates = df.duplicated(subset=key_cols, keep=False).sum()
            if duplicates > 0:
                issues.append(
                    f"Found {duplicates:,} duplicate rows based on {', '.join(key_cols)}"
                )

        return len(issues) == 0, issues

    def run_data_quality_checks(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run all data quality checks and return results.
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

        for check_name, check_func in checks.items():
            is_valid, issues = check_func(df)

            if is_valid:
                results["passed"].append(check_name)
                logger.info(f"  ✅ {check_name}: PASSED")
            else:
                results["failed"].append(check_name)
                results["issues"][check_name] = issues
                logger.warning(f"  ⚠️ {check_name}: FAILED")
                for issue in issues:
                    logger.warning(f"     - {issue}")

        total_checks = len(checks)
        passed_checks = len(results["passed"])
        failed_checks = len(results["failed"])

        logger.info(f"\n📊 Quality Check Summary: {passed_checks}/{total_checks} passed")

        if failed_checks > 0:
            logger.warning(f"⚠️ {failed_checks} check(s) failed - review issues above")
        else:
            logger.info("✅ All data quality checks passed!")

        return results

    def extract_staffing_data(self) -> pd.DataFrame:
        """
        Extract staffing data from SQL Server.

        Returns:
            DataFrame: The extracted staffing data.

        Raises:
            DataQualityException: If critical checks fail.
        """
        logger.info("👥 Fetching staffing data from SQL Server...")

        try:
            cursor = self.conn.cursor()
            cursor.execute(self.STAFFING_QUERY)

            logger.info("⏳ Fetching rows...")
            rows = cursor.fetchall()
            total_rows = len(rows)

            if total_rows == 0:
                logger.warning("⚠️ No staffing data retrieved from database")
                return pd.DataFrame()

            columns = [column[0] for column in cursor.description]
            logger.info(f"📊 Processing {total_rows:,} rows...")

            with tqdm(total=total_rows, desc="Loading data", unit="rows") as pbar:
                df = pd.DataFrame.from_records(rows, columns=columns)
                pbar.update(total_rows)

            # Normalise column names
            df.columns = self._normalise_column_names(df.columns)

            logger.info(f"✅ Retrieved {len(df):,} rows of staffing data.")
            logger.info(f"📊 Columns: {', '.join(df.columns)}")

            # Run quality checks
            if self.run_quality_checks:
                quality_results = self.run_data_quality_checks(df)
                critical_checks = ["Schema Validation", "Data Types"]
                failed_critical = [c for c in critical_checks if c in quality_results["failed"]]

                if failed_critical:
                    raise DataQualityException(
                        f"Critical data quality checks failed: {', '.join(failed_critical)}"
                    )

            return df

        except Exception as e:
            logger.error(f"❌ Error extracting staffing data: {str(e)}")
            raise
        finally:
            if "cursor" in locals():
                cursor.close()

    def save_to_csv(self, df: pd.DataFrame, filename: str = "staffing_data.csv") -> Optional[Path]:
        """
        Save staffing DataFrame to CSV.

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file, or None if nothing saved
        """
        if df.empty:
            logger.warning("⚠️ DataFrame is empty. No file saved.")
            return None

        filepath = self.output_dir / filename
        logger.info(f"💾 Saving staffing data to CSV: {filepath}")

        try:
            with tqdm(total=1, desc="Writing CSV", unit="file") as pbar:
                df.to_csv(filepath, index=False)
                pbar.update(1)

            file_size_kb = filepath.stat().st_size / 1024
            logger.info(f"✅ Staffing data saved to: {filepath}")
            logger.info(f"📄 File size: {file_size_kb:.2f} KB")
            logger.info(f"📊 Rows: {len(df):,} | Columns: {len(df.columns)}")

            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving staffing CSV: {str(e)}")
            raise

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


def load_staffing_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Utility function to load staffing data from CSV.

    Args:
        csv_path: Path to CSV file. If None, loads staffing_data.csv from default location.

    Returns:
        DataFrame: The staffing dataset
    """
    if csv_path is None:
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "data" / "staffing_data.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Staffing data file not found at: {csv_path}")

    logger.info(f"📂 Loading staffing data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    try:
        with StaffingDataExtractor() as extractor:
            df = extractor.extract_staffing_data()
            output_file = extractor.save_to_csv(df)

            if output_file:
                logger.info("🎉 Staffing data extraction completed successfully!")
                logger.info(f"📂 File saved: {output_file}")

    except DataQualityException as e:
        logger.error(f"❌ Data quality validation failed: {e}")
        logger.error("⚠️ Data was not saved due to quality issues")
    except Exception:
        logger.exception("❌ Error occurred during staffing data extraction")
