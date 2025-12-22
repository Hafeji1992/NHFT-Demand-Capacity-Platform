import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import sys

# Add parent directory to path to import from data_engineering
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engineering.staffing_data_ingestion import load_staffing_data

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StaffingSummaryStatistics:
    """
    Generate summary statistics from aggregated staffing data.
    """

    def __init__(self):
        logger.info("👥 Initialising Staffing Summary Statistics")

    # -----------------------------------------------------------------
    # Core summary
    # -----------------------------------------------------------------
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate high-level summary statistics from staffing data.

        Args:
            df: pandas DataFrame with staffing data

        Returns:
            Dictionary containing summary statistics
        """
        if df.empty:
            raise ValueError("Cannot generate statistics from empty DataFrame")

        required_cols = ["staff", "staff_group", "service_line"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info("📊 Generating staffing summary statistics...")

        summary = {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": len(df),
            "dimensions": {
                "service_lines": df["service_line"].nunique(dropna=False),
                "staff_groups": df["staff_group"].nunique(dropna=False)
            },
            "key_metrics": {
                "total_staff": int(df["staff"].sum()),
                "average_staff_per_service": float(
                    df.groupby("service_line")["staff"].sum().mean()
                ),
                "average_staff_per_group": float(
                    df.groupby("staff_group")["staff"].sum().mean()
                )
            }
        }

        logger.info("✅ Staffing summary statistics generated.")
        return summary

    # -----------------------------------------------------------------
    # Breakdown by service line
    # -----------------------------------------------------------------
    def generate_service_line_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate staffing breakdown by service line.
        """
        logger.info("📋 Generating service line staffing breakdown...")

        service_breakdown = (
            df.groupby("service_line", dropna=False)
            .agg(
                total_staff=("staff", "sum"),
                staff_groups=("staff_group", "nunique")
            )
            .sort_values("total_staff", ascending=False)
        )

        logger.info(f"✅ Generated breakdown for {len(service_breakdown)} service lines.")
        return service_breakdown

    # -----------------------------------------------------------------
    # Breakdown by staff group
    # -----------------------------------------------------------------
    def generate_staff_group_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate staffing breakdown by staff group.
        """
        logger.info("📋 Generating staff group breakdown...")

        group_breakdown = (
            df.groupby("staff_group", dropna=False)
            .agg(
                total_staff=("staff", "sum"),
                service_lines=("service_line", "nunique")
            )
            .sort_values("total_staff", ascending=False)
        )

        logger.info(f"✅ Generated breakdown for {len(group_breakdown)} staff groups.")
        return group_breakdown

    # -----------------------------------------------------------------
    # Service line × staff group matrix
    # -----------------------------------------------------------------
    def generate_service_group_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a pivot table of staff counts by service line and staff group.
        """
        logger.info("📊 Generating service line × staff group matrix...")

        matrix = pd.pivot_table(
            df,
            values="staff",
            index="service_line",
            columns="staff_group",
            aggfunc="sum",
            fill_value=0
        )

        logger.info("✅ Service × staff group matrix generated.")
        return matrix

    # -----------------------------------------------------------------
    # Formatting helpers
    # -----------------------------------------------------------------
    def _format_summary_text(self, summary: Dict[str, Any]) -> str:
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
        lines.append(f"Service Lines: {summary['dimensions']['service_lines']}")
        lines.append(f"Staff Groups: {summary['dimensions']['staff_groups']}")
        lines.append("")
        lines.append("KEY METRICS")
        lines.append("-" * 80)
        for metric, value in summary["key_metrics"].items():
            lines.append(f"{metric.replace('_', ' ').title()}: {value:,.2f}")
        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def print_summary(self, summary: Dict[str, Any]) -> None:
        print("\n" + self._format_summary_text(summary) + "\n")

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------
    def generate_all_statistics(
        self,
        df: pd.DataFrame,
        print_summary: bool = True
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        summary = self.generate_summary_statistics(df)
        service_breakdown = self.generate_service_line_breakdown(df)
        staff_group_breakdown = self.generate_staff_group_breakdown(df)
        service_group_matrix = self.generate_service_group_matrix(df)

        if print_summary:
            self.print_summary(summary)

        return summary, service_breakdown, staff_group_breakdown, service_group_matrix


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
def main():
    try:
        logger.info("📂 Loading staffing data...")
        df = load_staffing_data()

        stats = StaffingSummaryStatistics()

        summary, service_breakdown, staff_group_breakdown, matrix = (
            stats.generate_all_statistics(df, print_summary=True)
        )

        logger.info("🎉 Staffing summary statistics generated successfully!")
        logger.info(f"  - Service line breakdown: {len(service_breakdown)} rows")
        logger.info(f"  - Staff group breakdown: {len(staff_group_breakdown)} rows")

        print("\n" + "=" * 80)
        print("SERVICE LINE BREAKDOWN (Top 10)")
        print("=" * 80)
        print(service_breakdown.head(10))

        print("\n" + "=" * 80)
        print("STAFF GROUP BREAKDOWN")
        print("=" * 80)
        print(staff_group_breakdown)

        print("\n" + "=" * 80)
        print("SERVICE LINE × STAFF GROUP MATRIX")
        print("=" * 80)
        print(matrix.head(10))

    except FileNotFoundError as e:
        logger.error(f"❌ Staffing data file not found: {e}")
        logger.info("💡 Please run staffing_data_ingestion.py first.")
    except ValueError as e:
        logger.error(f"❌ Data validation error: {e}")
    except Exception:
        logger.exception("❌ Error occurred during staffing summary statistics generation")


if __name__ == "__main__":
    main()
