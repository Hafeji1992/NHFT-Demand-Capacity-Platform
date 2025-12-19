import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import sys

# Add parent directory to path to import from data_engineering
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engineering.patient_data_ingestion import load_patient_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PatientSummaryStatistics:
    """Generate comprehensive summary statistics from patient data."""

    def __init__(self):
        """Initialise summary statistics generator."""
        logger.info("📊 Initialising Patient Summary Statistics")

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics from the patient data.

        Args:
            df: pandas DataFrame with patient data

        Returns:
            Dictionary containing all summary statistics

        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df.empty:
            raise ValueError("Cannot generate statistics from empty DataFrame")

        logger.info("📊 Generating summary statistics...")

        # Log available columns for debugging
        logger.info(f"Available columns: {', '.join(df.columns)}")

        # Verify required columns exist
        required_cols = ['periodend', 'service_line', 'providercodecurrent']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        summary = {
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(df),
            'date_range': {
                'start': df['periodend'].min(),
                'end': df['periodend'].max(),
                'months': df['periodend'].nunique()
            },
            'dimensions': {
                'service_lines': df['service_line'].nunique(),
                'providers': df['providercodecurrent'].nunique()
            },
            'key_metrics': {
                'total_referrals': int(df['referrals'].sum()),
                'total_clock_stops': int(df['clockstopactuals'].sum()),
                'total_waiters': int(df['waiters'].sum()),
                'waiters_18plus': int(df['waiters18plusweeks'].sum()),
                'waiters_under_18': int(df['waitersunder18weeks'].sum()),
                'total_caseload': int(df['caseload'].sum()),
                'total_contacts': int(df['totalcontacts'].sum()),
                'ftf_contacts': int(df['ftfcontacts'].sum()),
                'discharges_with_clock_stop': int(df['dischargesfromcaseload'].sum()),
                'discharges_no_clock_stop': int(df['dischargesnoclockstop'].sum())
            },
            'average_metrics': {
                'avg_length_of_treatment': float(df['averagelengthoftreatment'].mean()),
                'avg_contacts_at_discharge': float(df['averagecontactsatdischarge'].mean()),
                'avg_ftf_contacts_at_discharge': float(df['averageftfcontactsatdischarge'].mean()),
                'avg_referral_clock_stop_ratio': float(df['referralclockstopratio'].mean()),
                'avg_contacts_per_caseload': float(df['totalcontactspercaseload'].mean())
            }
        }

        logger.info("✅ Summary statistics generated.")
        return summary

    def generate_service_line_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate breakdown by service line.

        Args:
            df: pandas DataFrame with patient data

        Returns:
            Service line breakdown DataFrame
        """
        logger.info("📋 Generating service line breakdown...")

        agg_dict = {
            'referrals': 'sum',
            'clockstopactuals': 'sum',
            'waiters': 'sum',
            'waiters18plusweeks': 'sum',
            'caseload': 'sum',
            'totalcontacts': 'sum',
            'ftfcontacts': 'sum',
            'averagelengthoftreatment': 'mean',
            'averagecontactsatdischarge': 'mean'
        }

        service_breakdown = (
            df.groupby('service_line', dropna=False)
            .agg(agg_dict)
            .round(2)
            .sort_values('referrals', ascending=False)
        )

        logger.info(f"✅ Breakdown generated for {len(service_breakdown)} service lines.")
        return service_breakdown

    def generate_provider_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate breakdown by provider.

        Args:
            df: pandas DataFrame with patient data

        Returns:
            Provider breakdown DataFrame
        """
        logger.info("📋 Generating provider breakdown...")

        agg_dict = {
            'referrals': 'sum',
            'clockstopactuals': 'sum',
            'waiters': 'sum',
            'waiters18plusweeks': 'sum',
            'caseload': 'sum',
            'totalcontacts': 'sum',
            'ftfcontacts': 'sum',
            'averagelengthoftreatment': 'mean',
            'averagecontactsatdischarge': 'mean'
        }

        provider_breakdown = (
            df.groupby('providercodecurrent', dropna=False)
            .agg(agg_dict)
            .round(2)
            .sort_values('referrals', ascending=False)
        )

        logger.info(f"✅ Breakdown generated for {len(provider_breakdown)} providers.")
        return provider_breakdown

    def generate_time_series_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time series summary by period.

        Args:
            df: pandas DataFrame with patient data

        Returns:
            Time series summary DataFrame
        """
        logger.info("📈 Generating time series summary...")

        agg_dict = {
            'referrals': 'sum',
            'clockstopactuals': 'sum',
            'waiters': 'sum',
            'waiters18plusweeks': 'sum',
            'caseload': 'sum',
            'totalcontacts': 'sum',
            'ftfcontacts': 'sum'
        }

        time_series = (
            df.groupby('periodend')
            .agg(agg_dict)
            .round(2)
            .sort_index()
        )

        logger.info(f"✅ Time series generated for {len(time_series)} periods.")
        return time_series

    def _format_summary_text(self, summary: Dict[str, Any]) -> str:
        """
        Format summary statistics as text.

        Args:
            summary: Dictionary of summary statistics

        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PATIENT DATA SUMMARY STATISTICS")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Generated: {summary['generation_date']}")
        lines.append("")

        lines.append("DATASET OVERVIEW")
        lines.append("-" * 80)
        lines.append(f"Total Records: {summary['total_records']:,}")
        lines.append(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        lines.append(f"Number of Months: {summary['date_range']['months']}")
        lines.append(f"Service Lines: {summary['dimensions']['service_lines']}")
        lines.append(f"Providers: {summary['dimensions']['providers']}")
        lines.append("")

        lines.append("KEY METRICS (TOTALS)")
        lines.append("-" * 80)
        for metric, value in summary['key_metrics'].items():
            lines.append(f"{metric.replace('_', ' ').title()}: {value:,}")

        lines.append("")
        lines.append("AVERAGE METRICS")
        lines.append("-" * 80)
        for metric, value in summary['average_metrics'].items():
            lines.append(f"{metric.replace('_', ' ').title()}: {value:.2f}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print summary statistics to console.

        Args:
            summary: Dictionary of summary statistics
        """
        print("\n" + self._format_summary_text(summary) + "\n")

    def generate_all_statistics(
            self,
            df: pd.DataFrame,
            print_summary: bool = True
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all summary statistics.

        Args:
            df: pandas DataFrame with patient data
            print_summary: Whether to print summary to console

        Returns:
            Tuple containing (summary_dict, service_breakdown, provider_breakdown, time_series)
        """
        # Generate all statistics
        summary = self.generate_summary_statistics(df)
        service_breakdown = self.generate_service_line_breakdown(df)
        provider_breakdown = self.generate_provider_breakdown(df)
        time_series = self.generate_time_series_summary(df)

        # Print to console if requested
        if print_summary:
            self.print_summary(summary)

        return summary, service_breakdown, provider_breakdown, time_series


def main():
    """Main execution function."""
    try:
        # Load the patient data
        logger.info("📂 Loading patient data...")
        df = load_patient_data()

        # Initialise summary statistics generator
        stats = PatientSummaryStatistics()

        # Generate all statistics
        summary, service_breakdown, provider_breakdown, time_series = stats.generate_all_statistics(
            df, print_summary=True
        )

        logger.info("🎉 Summary statistics generated successfully!")
        logger.info(f"\n📊 DataFrames available for analysis:")
        logger.info(f"  - service_breakdown: {len(service_breakdown)} rows")
        logger.info(f"  - provider_breakdown: {len(provider_breakdown)} rows")
        logger.info(f"  - time_series: {len(time_series)} rows")

        # Display preview of breakdowns
        print("\n" + "=" * 80)
        print("SERVICE LINE BREAKDOWN (Top 10)")
        print("=" * 80)
        print(service_breakdown.head(10))

        print("\n" + "=" * 80)
        print("PROVIDER BREAKDOWN (Top 10)")
        print("=" * 80)
        print(provider_breakdown.head(10))

        print("\n" + "=" * 80)
        print("TIME SERIES SUMMARY (Last 10 Periods)")
        print("=" * 80)
        print(time_series.tail(10))

    except FileNotFoundError as e:
        logger.error(f"❌ Data file not found: {e}")
        logger.info("💡 Please run patient_data_ingestion.py first to generate the data file.")
    except ValueError as e:
        logger.error(f"❌ Data validation error: {e}")
    except Exception:
        logger.exception("❌ Error occurred during summary statistics generation")


if __name__ == "__main__":
    main()