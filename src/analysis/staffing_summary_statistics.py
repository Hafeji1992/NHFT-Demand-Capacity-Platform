import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import sys

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engineering.staffing_data_ingestion import load_staffing_data


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Staffing Summary Statistics
# ---------------------------------------------------------------------
class StaffingSummaryStatistics:
    """Generate comprehensive summary statistics from staffing (capacity) data."""

    # -----------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------
    def __init__(self):
        """Initialise summary statistics generator."""
        logger.info("📊 Initialising Staffing Summary Statistics")

    # -----------------------------------------------------------------
    # Global Summary
    # -----------------------------------------------------------------
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics from the staffing data.

        Args:
            df: pandas DataFrame with staffing data

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
        required_cols = [
            "providercodecurrent",
            "service_line",
            "staff_group",
            "staff",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        summary = {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": len(df),
            "dimensions": {
                "providers": df["providercodecurrent"].nunique(),
                "service_lines": df["service_line"].nunique(),
                "staff_groups": df["staff_group"].nunique(),
            },
            "key_metrics": {
                "total_staff": int(df["staff"].sum()),
                "average_staff_per_service": float(
                    df.groupby(["providercodecurrent", "service_line"])["staff"].sum().mean()
                ),
                "median_staff_per_service": float(
                    df.groupby(["providercodecurrent", "service_line"])["staff"].sum().median()
                ),
            },
        }

        logger.info("✅ Summary statistics generated.")
        return summary

    # -----------------------------------------------------------------
    # Service Line Breakdown
    # -----------------------------------------------------------------
    def generate_service_line_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate breakdown by service line.

        Args:
            df: pandas DataFrame with staffing data

        Returns:
            Service line breakdown DataFrame
        """
        logger.info("📋 Generating service line breakdown...")

        breakdown = (
            df.groupby("service_line", dropna=False)
            .agg(
                total_staff=("staff", "sum"),
                staff_groups=("staff_group", "nunique"),
            )
            .sort_values("total_staff", ascending=False)
        )

        logger.info(f"✅ Breakdown generated for {len(breakdown)} service lines.")
        return breakdown

    # -----------------------------------------------------------------
    # Staff Group Breakdown
    # -----------------------------------------------------------------
    def generate_staff_group_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate breakdown by staff group.

        Args:
            df: pandas DataFrame with staffing data

        Returns:
            Staff group breakdown DataFrame
        """
        logger.info("📋 Generating staff group breakdown...")

        breakdown = (
            df.groupby("staff_group", dropna=False)
            .agg(
                total_staff=("staff", "sum"),
                providers=("providercodecurrent", "nunique"),
                service_lines=("service_line", "nunique"),
            )
            .sort_values("total_staff", ascending=False)
        )

        logger.info(f"✅ Breakdown generated for {len(breakdown)} staff groups.")
        return breakdown

    # -----------------------------------------------------------------
    # Formatting Helpers
    # -----------------------------------------------------------------
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
        lines.append("STAFFING DATA SUMMARY STATISTICS")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Generated: {summary['generation_date']}")
        lines.append("")

        lines.append("DATASET OVERVIEW")
        lines.append("-" * 80)
        lines.append(f"Total Records: {summary['total_records']:,}")
        lines.append(f"Providers: {summary['dimensions']['providers']}")
        lines.append(f"Service Lines: {summary['dimensions']['service_lines']}")
        lines.append(f"Staff Groups: {summary['dimensions']['staff_groups']}")
        lines.append("")

        lines.append("KEY METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Staff: {summary['key_metrics']['total_staff']:,}")
        lines.append(
            f"Average Staff Per Service: "
            f"{summary['key_metrics']['average_staff_per_service']:.2f}"
        )
        lines.append(
            f"Median Staff Per Service: "
            f"{summary['key_metrics']['median_staff_per_service']:.2f}"
        )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Print Helpers
    # -----------------------------------------------------------------
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print summary statistics to console.

        Args:
            summary: Dictionary of summary statistics
        """
        print("\n" + self._format_summary_text(summary) + "\n")

    # -----------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------
    def generate_all_statistics(
            self,
            df: pd.DataFrame,
            print_summary: bool = True,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        Generate all summary statistics.

        Args:
            df: pandas DataFrame with staffing data
            print_summary: Whether to print summary to console

        Returns:
            Tuple containing (summary_dict, service_breakdown, staff_group_breakdown)
        """
        # Generate all statistics
        summary = self.generate_summary_statistics(df)
        service_breakdown = self.generate_service_line_breakdown(df)
        staff_group_breakdown = self.generate_staff_group_breakdown(df)

        # Print to console if requested
        if print_summary:
            self.print_summary(summary)

        return summary, service_breakdown, staff_group_breakdown


# ---------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------
def main():
    """Main execution function."""
    try:
        # Load the staffing data
        logger.info("📂 Loading staffing data...")
        df = load_staffing_data()

        # Initialise summary statistics generator
        stats = StaffingSummaryStatistics()

        # Generate all statistics
        summary, service_breakdown, staff_group_breakdown = stats.generate_all_statistics(
            df, print_summary=True
        )

        logger.info("🎉 Summary statistics generated successfully!")
        logger.info(f"\n📊 DataFrames available for analysis:")
        logger.info(f"  - service_breakdown: {len(service_breakdown)} rows")
        logger.info(f"  - staff_group_breakdown: {len(staff_group_breakdown)} rows")

        # Display preview of breakdowns
        print("\n" + "=" * 80)
        print("SERVICE LINE BREAKDOWN (Top 10)")
        print("=" * 80)
        print(service_breakdown.head(10))

        print("\n" + "=" * 80)
        print("STAFF GROUP BREAKDOWN")
        print("=" * 80)
        print(staff_group_breakdown)

    except FileNotFoundError as e:
        logger.error(f"❌ Data file not found: {e}")
        logger.info("💡 Please run staffing_data_ingestion.py first to generate the data file.")
    except ValueError as e:
        logger.error(f"❌ Data validation error: {e}")
    except Exception:
        logger.exception("❌ Error occurred during summary statistics generation")


if __name__ == "__main__":
    main()