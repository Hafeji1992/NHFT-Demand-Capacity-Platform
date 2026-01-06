"""
NHFT Patient Data Ingestion
============================
Extracts patient demand metrics from SQL Server and writes a cleaned CSV for analysis and the dashboard.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import date, timedelta
import pandas as pd
from tqdm import tqdm

from data_engineering.connect import SQLServerConnection

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
# Patient Data Extractor
# ---------------------------------------------------------------------
class PatientDataExtractor:
    """Extracts patient data from SQL Server and saves to CSV."""

    @staticmethod
    def get_last_full_month_end(reference_date: Optional[date] = None) -> pd.Timestamp:
        """Return the last day of the most recently completed month.

        Example:
            If today is 2026-01-06, this returns 2025-12-31.

        Args:
            reference_date: Optional date to compute the cutoff from. Defaults to today.

        Returns:
            A pandas Timestamp representing the previous month-end.
        """
        if reference_date is None:
            reference_date = date.today()

        first_of_month = reference_date.replace(day=1)
        last_day_previous_month = first_of_month - timedelta(days=1)
        return pd.Timestamp(last_day_previous_month)

    # SQL query as class constant for better maintainability
    PATIENT_QUERY = """
        -- ============================================================================
        -- PATIENT DATA EXTRACTION QUERY
        -- ============================================================================
        -- Purpose: Extract Patient-related metrics from waiting times and
        -- contact attendance views, pulling referrals, contacts, waiters, caseload,
        -- discharge, and treatment metrics.
        -- ============================================================================
        SELECT
            -- Identifiers
            WT.[ProviderCodeCurrent],
            WT.[Service_Line],
            WT.[PeriodEnd],

            -- ======================================================================== 
            -- REFERRAL AND CLOCK STOP METRICS
            -- ========================================================================
            WT.[Referrals],
            WT.[ClockStopActuals],
            WT.[DischargesNoClockStop],

            -- ======================================================================== 
            -- CALCULATED RATIOS
            -- ========================================================================
            CASE 
                WHEN WT.[Referrals] > 0 
                THEN CAST(WT.[ClockStopActuals] AS FLOAT) / WT.[Referrals]
                ELSE NULL 
            END AS [ReferralClockStopRatio],

            CASE 
                WHEN WT.[Referrals] > 0 
                THEN CAST(WT.[DischargesNoClockStop] AS FLOAT) / WT.[Referrals]
                ELSE NULL 
            END AS [ReferralDischargedNoClockStopRatio],

            CASE 
                WHEN WT.[ClockStopActuals] > 0 
                THEN CAST(WT.[Referrals] AS FLOAT) / WT.[ClockStopActuals]
                ELSE NULL 
            END AS [DemandRatio],

            -- ======================================================================== 
            -- CONTACT METRICS
            -- ========================================================================
            CA.[TotalContacts],
            CA.[FTFContacts],

            -- ======================================================================== 
            -- CASELOAD METRICS
            -- ========================================================================
            WT.[Caseload],
            CA.[TotalCaseloadContacts],
            CA.[FTFCaseloadContacts],

            CASE 
                WHEN WT.[Caseload] > 0 
                THEN CAST(CA.[TotalCaseloadContacts] AS FLOAT) / WT.[Caseload]
                ELSE NULL 
            END AS [TotalContactsPerCaseload],

            CASE 
                WHEN WT.[Caseload] > 0 
                THEN CAST(CA.[FTFCaseloadContacts] AS FLOAT) / WT.[Caseload]
                ELSE NULL 
            END AS [FTFContactsPerCaseload],

            -- ======================================================================== 
            -- WAITER METRICS
            -- ========================================================================
            WT.[Waiters],
            WT.[Waiters] - WT.[Waiters18Plus] AS [Waiters<18Weeks],
            WT.[Waiters18Plus] AS [Waiters18+Weeks],

            -- ======================================================================== 
            -- DISCHARGE AND TREATMENT METRICS
            -- ========================================================================
            WT.[AverageLengthOfTreatment],
            WT.[AverageContactsAtDischarge],
            WT.[AverageFTFContactsAtDischarge],
            WT.[DischargesWithClockStop] AS [DischargesFromCaseload]

        -- ============================================================================
        -- SUBQUERY 1: WAITING TIMES AND REFERRAL DATA
        -- ============================================================================
        FROM (
            SELECT
                WT.[ProviderCodeCurrent],
                SL.[Service_Line],
                WT.[PeriodEnd],

                -- -------------------------------------------------------------------- 
                -- Discharge Quality Metrics
                -- --------------------------------------------------------------------
                AVG(CASE WHEN WT.[Discharged] = 1 
                    THEN WT.[ContactsPerReferral] END) AS [AverageContactsAtDischarge],
                AVG(CASE WHEN WT.[Discharged] = 1 
                    THEN WT.[ContactsFTFPerReferral] END) AS [AverageFTFContactsAtDischarge],
                AVG(CASE WHEN WT.[Discharged] = 1 
                    THEN DATEDIFF(DAY, WT.[Ref_Start], WT.[FirstContact]) END) AS [AverageLengthOfTreatment],

                -- -------------------------------------------------------------------- 
                -- Current Caseload
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 0 
                    THEN WT.[Ref_ID] END) AS [Caseload],

                -- -------------------------------------------------------------------- 
                -- Clock Stop Activity
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[FirstContactInPeriod] = 1 
                    THEN WT.[Ref_ID] END) AS [ClockStopActuals],

                -- -------------------------------------------------------------------- 
                -- Discharge Breakdown
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 1 AND WT.[FirstContact] IS NULL 
                    THEN WT.[Ref_ID] END) AS [DischargesNoClockStop],
                COUNT(CASE WHEN WT.[Discharged] = 1 AND WT.[FirstContact] IS NOT NULL 
                    THEN WT.[Ref_ID] END) AS [DischargesWithClockStop],

                -- -------------------------------------------------------------------- 
                -- New Referrals
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[RefStartThisPeriod] = 1 
                    THEN WT.[Ref_ID] END) AS [Referrals],

                -- -------------------------------------------------------------------- 
                -- Waiting List Metrics
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 1 
                    AND WT.[RTTExclusion] = 0 THEN WT.[Ref_ID] END) AS [Waiters],
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 1 
                    AND WT.[RTTExclusion] = 0 AND WT.[WaitingTime] > 7 * 18 
                    THEN WT.[Ref_ID] END) AS [Waiters18Plus]

            FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes] AS WT
            LEFT JOIN [MIS_Config].[dbo].[tbl_org_current_RL9_Service_Line] AS SL
                ON WT.[ProviderCodeCurrent] = SL.[Service_Codes]
            GROUP BY
                WT.[ProviderCodeCurrent],
                SL.[Service_Line],
                WT.[PeriodEnd]
        ) AS WT

        -- ============================================================================
        -- SUBQUERY 2: CONTACT ATTENDANCE DATA
        -- ============================================================================
        LEFT JOIN (
            SELECT
                CA.[ProviderCodeCurrent],
                CA.[PeriodEnd],

                -- -------------------------------------------------------------------- 
                -- Face-to-Face Contact Metrics
                -- --------------------------------------------------------------------
                -- FTF contacts after first contact (caseload activity)
                COUNT(CASE WHEN (CA.[FirstAttendance_FTF] = 1 OR CA.[FollowUp_FTF] = 1) 
                    AND CA.[Contact_Date] > WT.[FirstContact] THEN CA.[Ref_ID] END) AS [FTFCaseloadContacts],
                -- All FTF contacts (including first contacts)
                COUNT(CASE WHEN CA.[FirstAttendance_FTF] = 1 OR CA.[FollowUp_FTF] = 1 
                    THEN CA.[Ref_ID] END) AS [FTFContacts],

                -- -------------------------------------------------------------------- 
                -- Total Contact Metrics
                -- --------------------------------------------------------------------
                -- Total contacts after first contact (caseload activity)
                COUNT(CASE WHEN CA.[PatientSeen] = 1 AND CA.[Contact_Date] > WT.[FirstContact] 
                    THEN CA.[Ref_ID] END) AS [TotalCaseloadContacts],
                -- All contacts where patient was seen
                COUNT(CASE WHEN CA.[PatientSeen] = 1 
                    THEN CA.[Ref_ID] END) AS [TotalContacts]

            FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_ContactAttendances] AS CA
            LEFT JOIN [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes] AS WT
                ON CA.[Ref_ID] = WT.[Ref_ID]
                AND CA.[PeriodEnd] = WT.[PeriodEnd]
            GROUP BY
                CA.[ProviderCodeCurrent],
                CA.[PeriodEnd]
        ) AS CA
            ON WT.[ProviderCodeCurrent] = CA.[ProviderCodeCurrent]
            AND WT.[PeriodEnd] = CA.[PeriodEnd]

        ORDER BY
            WT.[ProviderCodeCurrent],
            WT.[Service_Line],
            WT.[PeriodEnd]
    """

    # Expected columns after normalisation
    EXPECTED_COLUMNS = [
        "providercodecurrent",
        "service_line",
        "periodend",
        "referrals",
        "clockstopactuals",
        "dischargesnoclockstop",
        "referralclockstopratio",
        "referraldischargednoclockstopratio",
        "demandratio",
        "totalcontacts",
        "ftfcontacts",
        "caseload",
        "totalcaseloadcontacts",
        "ftfcaseloadcontacts",
        "totalcontactspercaseload",
        "ftfcontactspercaseload",
        "waiters",
        "waitersunder18weeks",
        "waiters18plusweeks",
        "averagelengthoftreatment",
        "averagecontactsatdischarge",
        "averageftfcontactsatdischarge",
        "dischargesfromcaseload",
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
        return [
            col.lower()
            .replace(" ", "_")
            .replace("<", "under")
            .replace(">", "over")
            .replace("+", "plus")
            for col in columns
        ]

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
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {', '.join(sorted(missing_cols))}")

        # Check for unexpected columns
        unexpected_cols = set(df.columns) - set(self.EXPECTED_COLUMNS)
        if unexpected_cols:
            issues.append(f"Unexpected columns: {', '.join(sorted(unexpected_cols))}")

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
        critical_cols = ["providercodecurrent", "periodend", "service_line"]

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

        # Check periodend is datetime
        if "periodend" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["periodend"]
        ):
            issues.append("periodend column is not datetime type")

        # Check numeric columns
        numeric_cols = [
            "referrals",
            "clockstopactuals",
            "waiters",
            "caseload",
            "totalcontacts",
            "ftfcontacts",
        ]

        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"{col} is not numeric type (found: {df[col].dtype})")

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

        # Count columns should not be negative
        count_cols = [
            "referrals",
            "clockstopactuals",
            "waiters",
            "caseload",
            "totalcontacts",
            "ftfcontacts",
            "dischargesnoclockstop",
            "dischargesfromcaseload",
        ]

        for col in count_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count:,} negative values found")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Logical Consistency Checks
    # -----------------------------------------------------------------
    def _check_logical_consistency(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check logical relationships between columns.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Total waiters should equal under18 + 18plus waiters
        if all(
            col in df.columns
            for col in ["waiters", "waitersunder18weeks", "waiters18plusweeks"]
        ):
            mismatch = df[
                (df["waiters"].notna())
                & (df["waitersunder18weeks"].notna())
                & (df["waiters18plusweeks"].notna())
                & (
                    abs(
                        df["waiters"]
                        - (df["waitersunder18weeks"] + df["waiters18plusweeks"])
                    )
                    > 0.1
                )
            ]

            if len(mismatch) > 0:
                issues.append(
                    f"Waiter calculation mismatch: {len(mismatch):,} rows where "
                    f"waiters ≠ waitersunder18weeks + waiters18plusweeks"
                )

        # FTF contacts should not exceed total contacts
        if "ftfcontacts" in df.columns and "totalcontacts" in df.columns:
            invalid = (df["ftfcontacts"] > df["totalcontacts"]).sum()
            if invalid > 0:
                issues.append(f"FTF contacts exceed total contacts in {invalid:,} rows")

        # FTF caseload contacts should not exceed total caseload contacts
        if (
            "ftfcaseloadcontacts" in df.columns
            and "totalcaseloadcontacts" in df.columns
        ):
            invalid = (df["ftfcaseloadcontacts"] > df["totalcaseloadcontacts"]).sum()
            if invalid > 0:
                issues.append(
                    f"FTF caseload contacts exceed total caseload contacts in {invalid:,} rows"
                )

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

        # Check for duplicate combinations of provider, service_line, and period
        key_cols = ["providercodecurrent", "service_line", "periodend"]

        if all(col in df.columns for col in key_cols):
            duplicates = df.duplicated(subset=key_cols, keep=False).sum()
            if duplicates > 0:
                issues.append(
                    f"Found {duplicates:,} duplicate rows based on "
                    f"{', '.join(key_cols)}"
                )

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Date Continuity Check
    # -----------------------------------------------------------------
    def _check_date_continuity(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for gaps in time series data.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if "periodend" not in df.columns:
            return True, issues

        # Get unique periods sorted
        periods = df["periodend"].dropna().sort_values().unique()

        if len(periods) < 2:
            return True, issues

        # Check for large gaps (more than 60 days suggests missing months)
        period_series = pd.Series(periods)
        gaps = period_series.diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=60)]

        if len(large_gaps) > 0:
            issues.append(
                f"Found {len(large_gaps)} large gap(s) in time series "
                f"(> 60 days between periods)"
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
            "Logical Consistency": self._check_logical_consistency,
            "Duplicate Detection": self._check_duplicates,
            "Date Continuity": self._check_date_continuity,
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
    def extract_patient_data(self) -> pd.DataFrame:
        """
        Extract patient data from SQL Server views.

        Returns:
            pandas.DataFrame: The extracted data

        Raises:
            Exception: If data extraction fails
            DataQualityException: If quality checks fail
        """
        logger.info("📊 Fetching patient data from SQL Server...")

        try:
            cursor = self.conn.cursor()
            cursor.execute(self.PATIENT_QUERY)

            # Fetch all rows
            logger.info("⏳ Fetching rows...")
            rows = cursor.fetchall()
            total_rows = len(rows)

            if total_rows == 0:
                logger.warning("⚠️ No data retrieved from database")
                return pd.DataFrame()

            # Get column names
            columns = [column[0] for column in cursor.description]

            logger.info(f"📊 Processing {total_rows:,} rows...")

            # Create DataFrame with progress bar
            with tqdm(total=total_rows, desc="Loading data", unit="rows") as pbar:
                df = pd.DataFrame.from_records(rows, columns=columns)
                pbar.update(total_rows)

            # Normalise column names
            df.columns = self._normalise_column_names(df.columns)

            # Convert periodend to datetime
            if "periodend" in df.columns:
                df["periodend"] = pd.to_datetime(df["periodend"])

                # Cap to last fully completed month (prevents partial-month leakage)
                cutoff = self.get_last_full_month_end()
                before_rows = len(df)
                df = df[df["periodend"].notna() & (df["periodend"] <= cutoff)]
                removed = before_rows - len(df)

                if removed > 0:
                    logger.info(
                        f"📅 Applied month-end cutoff at {cutoff.date()}: removed {removed:,} row(s)"
                    )
                logger.info(
                    f"📅 Max periodend in extracted dataset: {df['periodend'].max().date() if not df.empty else 'N/A'}"
                )

            logger.info(f"✅ Retrieved {len(df):,} rows from source views.")
            logger.info(
                f"📊 Columns: {', '.join(df.columns[:5])}... ({len(df.columns)} total)"
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
    def save_to_csv(self, df: pd.DataFrame, filename: str = "patient_data.csv") -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: pandas DataFrame to save
            filename: Name of the CSV file (default: patient_data.csv)

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
def load_patient_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Utility function to load patient data from CSV.

    Args:
        csv_path: Path to CSV file. If None, loads patient_data.csv
                  from default location.

    Returns:
        pandas.DataFrame: The patient data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    if csv_path is None:
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "data" / "patient_data.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Patient data file not found at: {csv_path}")

    logger.info(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["periodend"])
    logger.info(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


# ---------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Use context manager for automatic connection cleanup
    try:
        with PatientDataExtractor() as extractor:
            # Extract data from SQL Server
            df = extractor.extract_patient_data()

            # Save as patient_data.csv
            output_file = extractor.save_to_csv(df)

            if output_file:
                logger.info("🎉 Patient data extraction completed successfully!")
                logger.info(f"📂 File saved: {output_file}")

    except DataQualityException as e:
        logger.error(f"❌ Data quality validation failed: {e}")
        logger.error("⚠️ Data was not saved due to quality issues")
    except Exception:
        logger.exception("❌ Error occurred during patient data extraction")
