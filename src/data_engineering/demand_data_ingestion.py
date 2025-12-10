import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm
from data_engineering.connect import SQLServerConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DemandDataExtractor:
    """Extracts demand data from SQL Server and saves to CSV."""

    # SQL query as class constant for better maintainability
    DEMAND_QUERY = """
        -- ============================================================================
        -- DEMAND DATA EXTRACTION QUERY
        -- ============================================================================
        -- Purpose: Extract demand-related metrics from waiting times and
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

    def __init__(self, config_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialise SQL Server connection and output directory.

        Args:
            config_path: Path to config.ini file (optional, auto-detected if None)
            output_dir: Directory to save CSV files (defaults to data/)
        """
        self.sql_conn = SQLServerConnection(config_path)
        self.conn = self.sql_conn.connect()

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
            .replace("<", "under_")
            .replace(">", "over_")
            .replace("+", "plus")
            for col in columns
        ]

    def extract_demand_data(self) -> pd.DataFrame:
        """
        Extract demand data from SQL Server views.

        Returns:
            pandas.DataFrame: The extracted data

        Raises:
            Exception: If data extraction fails
        """
        logger.info("📊 Fetching demand data from SQL Server...")

        try:
            cursor = self.conn.cursor()
            cursor.execute(self.DEMAND_QUERY)

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
            if 'periodend' in df.columns:
                df['periodend'] = pd.to_datetime(df['periodend'])

            logger.info(f"✅ Retrieved {len(df):,} rows from source views.")
            logger.info(f"📊 Columns: {', '.join(df.columns[:5])}... ({len(df.columns)} total)")

            return df

        except Exception as e:
            logger.error(f"❌ Error extracting data: {str(e)}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()

    def save_to_csv(self, df: pd.DataFrame, filename: str = "demand_data.csv") -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: pandas DataFrame to save
            filename: Name of the CSV file (default: demand_data.csv)

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


def load_demand_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Utility function to load demand data from CSV.

    Args:
        csv_path: Path to CSV file. If None, loads demand_data.csv
                  from default location.

    Returns:
        pandas.DataFrame: The demand data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    if csv_path is None:
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / "data" / "demand_data.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Demand data file not found at: {csv_path}")

    logger.info(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["periodend"])
    logger.info(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    # Use context manager for automatic connection cleanup
    try:
        with DemandDataExtractor() as extractor:
            # Extract data from SQL Server
            df = extractor.extract_demand_data()

            # Save as demand_data.csv
            output_file = extractor.save_to_csv(df)

            if output_file:
                logger.info("🎉 Demand data extraction completed successfully!")
                logger.info(f"📂 File saved: {output_file}")

    except Exception:
        logger.exception("❌ Error occurred during demand data extraction")