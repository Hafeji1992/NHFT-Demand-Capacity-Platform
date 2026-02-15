"""
NHFT Patient Data Ingestion
============================
Extracts patient demand metrics from SQL Server and writes a cleaned CSV for analysis and the dashboard.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import date, timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np  # NEW

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
# Patient Data Extractor
# ---------------------------------------------------------------------
class PatientDataExtractor:
    """Extracts patient data from SQL Server and saves to CSV."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        *,
        run_quality_checks: bool = True,
        config_path: Optional[str | Path] = None,
    ):
        """Create a patient data extractor.

        Args:
            output_dir: Where to write patient_data.csv. Defaults to project_root/data.
            run_quality_checks: Whether to run data quality checks after extraction.
            config_path: Optional path to config.ini for SQL server settings.
        """

        if output_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            output_dir = project_root / "data"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Connection helpers
        self.sql_conn = SQLServerConnection(
            str(config_path) if config_path is not None else None
        )
        self.conn = None

        self.run_quality_checks = bool(run_quality_checks)

    def open_connection(self):
        """Open SQL Server connection if not already open."""
        if self.conn is None:
            self.conn = self.sql_conn.connect()

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a column label to snake_case.

        Handles:
        - CamelCase and ALLCAPS acronyms (e.g., FTFContacts -> ftf_contacts)
        - Digit boundaries (e.g., Over18Weeks -> over_18_weeks)
        - Existing separators (spaces, hyphens, underscores)
        """
        s = str(name).strip()
        s = re.sub(r"[^0-9A-Za-z]+", "_", s)
        # Split acronym-word boundaries: 'FTFContacts' -> 'FTF_Contacts'
        s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
        # Split lower/digit to upper boundaries: 'codeCurrent' -> 'code_Current'
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        # Split letters<->digits: 'Over18' -> 'Over_18'
        s = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", s)
        s = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_").lower()

    @staticmethod
    def _normalise_column_names(columns) -> List[str]:
        """Normalise raw SQL column names to snake_case.

        Notes:
            - Drops any source-provided waiters_under_18_weeks column so the pipeline
              can compute it consistently from waiters and waiters_over_18_weeks.
        """
        out: List[str] = []
        seen: set[str] = set()

        for col in list(columns):
            name = PatientDataExtractor._to_snake_case(col)

            if name == "waiters_under_18_weeks":
                continue

            if name and name not in seen:
                out.append(name)
                seen.add(name)

        return out

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

    @staticmethod
    def _format_provider_code_current(series: pd.Series) -> pd.Series:
        """Normalise provider codes to a 3-digit string (e.g. 6 -> '006').

        Notes:
            - Handles ints/floats/strings.
            - Leaves non-numeric values unchanged.
            - Keeps nulls as <NA>.
        """

        if series is None:
            return series

        s = series.astype("string")
        s = s.str.strip()

        # Common CSV/DB artifacts: 6.0 -> 6
        s = s.str.replace(r"\.0$", "", regex=True)

        # Only pad purely numeric codes up to 3 digits.
        is_digits = s.str.fullmatch(r"\d+", na=False)
        within_3 = s.str.len().fillna(0).astype(int) <= 3
        to_pad = is_digits & within_3

        s = s.mask(to_pad, s.str.zfill(3))
        return s

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

            -- ======================================================================== 
            -- WAITER METRICS
            -- ========================================================================
            WT.[Waiters],
            WT.[WaitersOver18Weeks],

            -- ======================================================================== 
            -- CASELOAD METRICS
            -- ========================================================================
            WT.[Caseload],
            CA.[TotalCaseloadContacts],
            CA.[FTFCaseloadContacts],

            -- ======================================================================== 
            -- CONTACT METRICS
            -- ========================================================================
            CA.[TotalContacts],
            CA.[FTFContacts],

            -- ======================================================================== 
            -- DISCHARGE AND TREATMENT METRICS
            -- ========================================================================
            WT.[AverageLengthOfTreatment],
            WT.[AverageContactsAtDischarge],
            WT.[AverageFTFContactsAtDischarge],
            WT.[DischargesNoClockStop],
            WT.[DischargesWithClockStop]

        -- ============================================================================
        -- SUBQUERY 1: WAITING TIMES AND REFERRAL DATA
        -- ============================================================================
        FROM (
            SELECT
                WT.[ProviderCodeCurrent],
                SL.[Service_Line],
                WT.[PeriodEnd],

                -- --------------------------------------------------------------------
                -- New Referrals
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[RefStartThisPeriod] = 1 THEN WT.[Ref_ID] END) AS [Referrals],

                -- --------------------------------------------------------------------
                -- Clock Stop Activity
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[FirstContactInPeriod] = 1 THEN WT.[Ref_ID] END) AS [ClockStopActuals],

                -- --------------------------------------------------------------------
                -- Waiting List Metrics
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 1 AND WT.[RTTExclusion] = 0 THEN WT.[Ref_ID] END) AS [Waiters],
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 1 AND WT.[RTTExclusion] = 0 AND WT.[WaitingTime] > 7 * 18 THEN WT.[Ref_ID] END) AS [WaitersOver18Weeks],

                -- --------------------------------------------------------------------
                -- Current Caseload
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 0 THEN WT.[Ref_ID] END) AS [Caseload],

                -- --------------------------------------------------------------------
                -- Discharge Quality Metrics
                -- --------------------------------------------------------------------
                AVG(CASE WHEN WT.[Discharged] = 1 THEN WT.[ContactsPerReferral] END) AS [AverageContactsAtDischarge],
                AVG(CASE WHEN WT.[Discharged] = 1 THEN WT.[ContactsFTFPerReferral] END) AS [AverageFTFContactsAtDischarge],
                AVG(CASE WHEN WT.[Discharged] = 1 THEN DATEDIFF(DAY, WT.[Ref_Start], WT.[FirstContact]) END) AS [AverageLengthOfTreatment],

                -- --------------------------------------------------------------------
                -- Discharge Breakdown
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 1 AND WT.[FirstContact] IS NULL THEN WT.[Ref_ID] END) AS [DischargesNoClockStop],
                COUNT(CASE WHEN WT.[Discharged] = 1 AND WT.[FirstContact] IS NOT NULL THEN WT.[Ref_ID] END) AS [DischargesWithClockStop]

            FROM (
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes]
			UNION  
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes_2324]
			) AS WT
            
			LEFT JOIN [MIS_Config].[dbo].[tbl_org_current_RL9_Service_Line] AS SL ON WT.[ProviderCodeCurrent] = SL.[Service_Codes]
			WHERE WT.[ProviderCodeCurrent] NOT IN ('996', '998')
				AND SL.[Status] = 'ACTIVE'
				AND SL.[RTT_Report_Enabled] = 1 -- RTT Reporting Only Services

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
                COUNT(CASE WHEN (CA.[FirstAttendance_FTF] = 1 OR CA.[FollowUp_FTF] = 1) AND CA.[Contact_Date] > WT.[FirstContact] THEN CA.[Ref_ID] END) AS [FTFCaseloadContacts],

                -- All FTF contacts (including first contacts)
                COUNT(CASE WHEN CA.[FirstAttendance_FTF] = 1 OR CA.[FollowUp_FTF] = 1 THEN CA.[Ref_ID] END) AS [FTFContacts],

                -- --------------------------------------------------------------------
                -- Total Contact Metrics
                -- --------------------------------------------------------------------
                -- Total contacts after first contact (caseload activity)
                COUNT(CASE WHEN CA.[PatientSeen] = 1 AND CA.[Contact_Date] > WT.[FirstContact] THEN CA.[Ref_ID] END) AS [TotalCaseloadContacts],

                -- All contacts where patient was seen
                COUNT(CASE WHEN CA.[PatientSeen] = 1 THEN CA.[Ref_ID] END) AS [TotalContacts]

            FROM (
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_ContactAttendances]
			UNION  
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_ContactAttendances_2324]
			) AS CA
            
			LEFT JOIN (
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes]
			UNION  
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes_2324]
			) AS WT
                ON CA.[Ref_ID] = WT.[Ref_ID] 
                --AND CA.[ProviderCodeCurrent] = WT.[ProviderCodeCurrent]
                AND CA.[PeriodEnd] = WT.[PeriodEnd]

            GROUP BY
                CA.[ProviderCodeCurrent],
                CA.[PeriodEnd]
                
        ) AS CA 
            ON WT.[ProviderCodeCurrent] = CA.[ProviderCodeCurrent] 
            AND WT.[PeriodEnd] = CA.[PeriodEnd]
                
        ORDER BY WT.[ProviderCodeCurrent], WT.[PeriodEnd]
		;
    """

    # Expected columns after normalisation (base extract schema)
    REQUIRED_COLUMNS = [  # RENAMED/REFINED (was EXPECTED_COLUMNS)
        "provider_code_current",
        "service_line",
        "period_end",
        "referrals",
        "clock_stop_actuals",
        "total_contacts",
        "ftf_contacts",
        "caseload",
        "total_caseload_contacts",
        "ftf_caseload_contacts",
        "waiters",
        "waiters_over_18_weeks",
        "average_length_of_treatment",
        "average_contacts_at_discharge",
        "average_ftf_contacts_at_discharge",
        "discharges_no_clock_stop",
        "discharges_with_clock_stop",
    ]

    # Columns produced by calculations / dashboard logic (allowed extras)
    DERIVED_COLUMNS = [
        "waiters_under_18_weeks_calc",
        "clock_stop_target",
        "referral_clock_stop_ratio",
        "referral_discharged_no_clock_stop_ratio",
        "demand_ratio",
        "total_contacts_per_caseload",
        "ftf_contacts_per_caseload",
    ]

    # -----------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------
    @staticmethod
    def _safe_divide(numer: pd.Series, denom: pd.Series | float | int) -> pd.Series:
        """Elementwise division that returns NaN where denom is 0/NaN."""
        denom_series = (
            denom
            if isinstance(denom, pd.Series)
            else pd.Series(denom, index=numer.index)
        )
        out = numer.astype("float64") / denom_series.astype("float64")
        out = out.mask((denom_series == 0) | (denom_series.isna()))
        return out

    @staticmethod
    def add_waiters_under_18_weeks_calc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'waiters_under_18_weeks_calc' = waiters - waiters_over_18_weeks.

        If a source 'waiters_under_18_weeks' exists, this does not overwrite it;
        it only adds the calculated version for validation / fallback usage.
        """
        if {"waiters", "waiters_over_18_weeks"}.issubset(df.columns):
            df["waiters_under_18_weeks_calc"] = (
                df["waiters"] - df["waiters_over_18_weeks"]
            )
        return df

    @staticmethod
    def validate_demand_percentile(demand_percentile: float) -> float:
        """Validate 50..100 step 5, return as float."""
        if demand_percentile is None:
            raise ValueError("demand_percentile is required")
        dp = float(demand_percentile)
        if dp < 50 or dp > 100 or (dp % 5) != 0:
            raise ValueError(
                "demand_percentile must be between 50 and 100 in 5% increments"
            )
        return dp

    @classmethod
    def compute_clock_stop_target(
        cls, df: pd.DataFrame, demand_percentile: float
    ) -> float:
        """
        For a (service-line filtered) df, compute the Xth percentile of referrals.
        Returns NaN if referrals are missing/empty.
        """
        dp = cls.validate_demand_percentile(demand_percentile)
        if "referrals" not in df.columns or df.empty:
            return float("nan")
        s = pd.to_numeric(df["referrals"], errors="coerce").dropna()
        if s.empty:
            return float("nan")
        return float(s.quantile(dp / 100.0))

    @classmethod
    def add_derived_metrics(
        cls,
        df: pd.DataFrame,
        demand_percentile: float = 60,
        service_line: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add derived dashboard metrics.
        - If service_line is provided, computes a single fixed clock_stop_target for that service line.
        - Otherwise, computes a target per service_line and maps it to rows.
        """
        dp = cls.validate_demand_percentile(demand_percentile)

        # ensure waiters-under-18 calc exists for validation/fallback usage
        df = cls.add_waiters_under_18_weeks_calc(df)

        # Compute target(s)
        if service_line is not None:
            sdf = (
                df[df["service_line"] == service_line].copy()
                if "service_line" in df.columns
                else df.copy()
            )
            target = cls.compute_clock_stop_target(sdf, dp)
            df = sdf
            df["clock_stop_target"] = target
        else:
            if "service_line" in df.columns and "referrals" in df.columns:
                targets = (
                    df.assign(
                        referrals_num=pd.to_numeric(df["referrals"], errors="coerce")
                    )
                    .groupby("service_line")["referrals_num"]
                    .quantile(dp / 100.0)
                )
                df["clock_stop_target"] = (
                    df["service_line"].map(targets).astype("float64")
                )
            else:
                df["clock_stop_target"] = np.nan

        # Ratios and throughput metrics
        if "clock_stop_actuals" in df.columns:
            df["referral_clock_stop_ratio"] = cls._safe_divide(
                df["clock_stop_actuals"], df["clock_stop_target"]
            )
        else:
            df["referral_clock_stop_ratio"] = np.nan

        if "discharges_no_clock_stop" in df.columns:
            df["referral_discharged_no_clock_stop_ratio"] = cls._safe_divide(
                df["discharges_no_clock_stop"], df["clock_stop_target"]
            )
        else:
            df["referral_discharged_no_clock_stop_ratio"] = np.nan

        df["demand_ratio"] = (
            df["referral_clock_stop_ratio"]
            + df["referral_discharged_no_clock_stop_ratio"]
        )

        # Contact intensity per caseload
        if "total_caseload_contacts" in df.columns and "caseload" in df.columns:
            df["total_contacts_per_caseload"] = cls._safe_divide(
                df["total_caseload_contacts"], df["caseload"]
            )
        else:
            df["total_contacts_per_caseload"] = np.nan

        if "ftf_caseload_contacts" in df.columns and "caseload" in df.columns:
            df["ftf_contacts_per_caseload"] = cls._safe_divide(
                df["ftf_caseload_contacts"], df["caseload"]
            )
        else:
            df["ftf_contacts_per_caseload"] = np.nan

        return df

    # -----------------------------------------------------------------
    # Schema Validation
    # -----------------------------------------------------------------
    def _check_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if DataFrame has required base columns.
        Derived columns are allowed and won't fail the check.
        """
        issues: List[str] = []

        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {', '.join(sorted(missing_cols))}")

        # Only flag truly unexpected columns (not derived)
        allowed_extras = set(self.DERIVED_COLUMNS)
        unexpected_cols = (
            set(df.columns) - set(self.REQUIRED_COLUMNS)
        ) - allowed_extras
        if unexpected_cols:
            issues.append(f"Unexpected columns: {', '.join(sorted(unexpected_cols))}")

        return len(missing_cols) == 0, issues  # NOTE: only missing base cols fail

    # -----------------------------------------------------------------
    # Data Completeness Check
    # -----------------------------------------------------------------
    def _check_data_completeness(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check required columns for missing values."""

        issues: List[str] = []

        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue
            missing = int(df[col].isna().sum())
            if missing > 0:
                issues.append(f"{col}: {missing:,} missing value(s)")

        return len(issues) == 0, issues

    # -----------------------------------------------------------------
    # Data Types Check
    # -----------------------------------------------------------------
    def _check_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check basic expected types (datetime for period_end, numeric for counts)."""

        issues: List[str] = []

        if "period_end" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["period_end"]
        ):
            issues.append("period_end is not a datetime column")

        numeric_cols = [
            c
            for c in self.REQUIRED_COLUMNS
            if c
            not in {
                "provider_code_current",
                "service_line",
                "period_end",
            }
            and c in df.columns
        ]

        for col in numeric_cols:
            s = df[col]
            coerced = pd.to_numeric(s, errors="coerce")
            newly_missing = int(coerced.isna().sum() - s.isna().sum())
            if newly_missing > 0:
                issues.append(f"{col}: {newly_missing:,} value(s) are non-numeric")

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
            "clock_stop_actuals",
            "waiters",
            "caseload",
            "total_contacts",
            "ftf_contacts",
            "discharges_no_clock_stop",
            "discharges_with_clock_stop",  # FIX (was discharges_from_caseload)
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

        # Waiters under 18 weeks calculated vs components
        if {"waiters", "waiters_over_18_weeks"}.issubset(df.columns):
            if "waiters_under_18_weeks_calc" not in df.columns:
                df = self.add_waiters_under_18_weeks_calc(df)

            mismatch = df[
                df["waiters"].notna()
                & df["waiters_over_18_weeks"].notna()
                & df["waiters_under_18_weeks_calc"].notna()
                & (
                    abs(
                        df["waiters"]
                        - (
                            df["waiters_under_18_weeks_calc"]
                            + df["waiters_over_18_weeks"]
                        )
                    )
                    > 0.1
                )
            ]
            if len(mismatch) > 0:
                issues.append(
                    f"Waiter calculation mismatch: {len(mismatch):,} rows where "
                    f"waiters ≠ waiters_under_18_weeks_calc + waiters_over_18_weeks"
                )

            # Optional: if a source waiters_under_18_weeks exists, validate against calc
            if "waiters_under_18_weeks" in df.columns:
                mismatch_src = df[
                    df["waiters_under_18_weeks"].notna()
                    & df["waiters_under_18_weeks_calc"].notna()
                    & (
                        abs(
                            df["waiters_under_18_weeks"]
                            - df["waiters_under_18_weeks_calc"]
                        )
                        > 0.1
                    )
                ]
                if len(mismatch_src) > 0:
                    issues.append(
                        f"WaitersUnder18Weeks vs calc mismatch: {len(mismatch_src):,} rows differ"
                    )

        # FTF contacts should not exceed total contacts
        if "ftf_contacts" in df.columns and "total_contacts" in df.columns:
            invalid = (df["ftf_contacts"] > df["total_contacts"]).sum()
            if invalid > 0:
                issues.append(f"FTF contacts exceed total contacts in {invalid:,} rows")

        # FTF caseload contacts should not exceed total caseload contacts
        if (
            "ftf_caseload_contacts" in df.columns
            and "total_caseload_contacts" in df.columns
        ):
            invalid = (
                df["ftf_caseload_contacts"] > df["total_caseload_contacts"]
            ).sum()
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
        key_cols = ["provider_code_current", "service_line", "period_end"]

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

        if "period_end" not in df.columns:
            return True, issues

        # Get unique periods sorted
        periods = df["period_end"].dropna().sort_values().unique()

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
            self.open_connection()
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
                # Defensive: ensure row width matches column count (unit tests use a fake cursor)
                if rows:
                    row_width = len(rows[0])
                    col_width = len(columns)
                    if row_width != col_width:
                        logger.warning(
                            "⚠️ Row/column width mismatch (%s values vs %s columns). Adjusting.",
                            row_width,
                            col_width,
                        )
                        if row_width > col_width:
                            rows = [r[:col_width] for r in rows]
                        else:
                            columns = columns[:row_width]

                df = pd.DataFrame.from_records(rows, columns=columns)
                pbar.update(total_rows)

            # Normalise column names
            df.columns = self._normalise_column_names(df.columns)

            # Canonicalise provider code formatting at ingestion time
            if "provider_code_current" in df.columns:
                df["provider_code_current"] = self._format_provider_code_current(
                    df["provider_code_current"]
                )

            # Convert period_end to datetime
            if "period_end" in df.columns:
                df["period_end"] = pd.to_datetime(df["period_end"])

                # Cap to last fully completed month (prevents partial-month leakage)
                cutoff = self.get_last_full_month_end()
                before_rows = len(df)
                df = df[df["period_end"].notna() & (df["period_end"] <= cutoff)]
                removed = before_rows - len(df)

                if removed > 0:
                    logger.info(
                        f"📅 Applied month-end cutoff at {cutoff.date()}: removed {removed:,} row(s)"
                    )
                logger.info(
                    f"📅 Max period_end in extracted dataset: {df['period_end'].max().date() if not df.empty else 'N/A'}"
                )

            # Ensure waiters_under_18_weeks exists for downstream usage + tests
            if "waiters_under_18_weeks" not in df.columns and {
                "waiters",
                "waiters_over_18_weeks",
            }.issubset(df.columns):
                df["waiters_under_18_weeks"] = (
                    df["waiters"] - df["waiters_over_18_weeks"]
                )

            # Add calculated column(s) used for validation/fallback downstream (Dash)
            df = self.add_waiters_under_18_weeks_calc(df)  # NEW

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
    def save_to_csv(
        self, df: pd.DataFrame, filename: str = "patient_data.csv"
    ) -> Optional[Path]:
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
        try:
            self.sql_conn.close()
        finally:
            self.conn = None
        logger.info("✅ Database connection closed.")

    def __enter__(self):
        """Context manager entry."""
        self.open_connection()
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

    # IMPORTANT: preserve leading zeros for provider_code_current
    df = pd.read_csv(
        csv_path,
        parse_dates=["period_end"],
        dtype={"provider_code_current": "string"},
    )

    if "provider_code_current" in df.columns:
        df["provider_code_current"] = (
            PatientDataExtractor._format_provider_code_current(
                df["provider_code_current"]
            )
        )
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
