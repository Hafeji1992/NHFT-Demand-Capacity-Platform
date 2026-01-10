"""
NHFT Master Data Ingestion
==========================
Runs the end-to-end ingestion pipeline to refresh patient and staffing datasets.

Outputs (default):
- data/patient_data.csv
- data/staffing_data.csv
- data/last_refreshed.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# Ensure src is on the path so `data_engineering.*` imports work when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engineering.patient_data_ingestion import PatientDataExtractor
from data_engineering.staffing_data_ingestion import StaffingDataExtractor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _write_last_refreshed(
    data_dir: Path,
    patient_max_periodend: Optional[str],
    patient_cutoff: Optional[str],
) -> Path:
    payload = {
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "patient": {
            "periodend_cutoff": patient_cutoff,
            "max_periodend_in_file": patient_max_periodend,
        },
    }

    out_path = data_dir / "last_refreshed.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NHFT master data ingestion")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to config.ini (optional; auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory to write CSV outputs (defaults to project data/)",
    )
    parser.add_argument(
        "--skip-patient",
        action="store_true",
        help="Skip patient ingestion",
    )
    parser.add_argument(
        "--skip-staffing",
        action="store_true",
        help="Skip staffing ingestion",
    )
    parser.add_argument(
        "--no-quality-checks",
        action="store_true",
        help="Disable data quality checks",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.output_dir) if args.output_dir else (project_root / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting NHFT master ingestion")

    patient_max_periodend: Optional[str] = None
    patient_cutoff: Optional[str] = None

    if not args.skip_patient:
        logger.info("\n=== Patient ingestion ===")
        with PatientDataExtractor(
            config_path=args.config_path,
            output_dir=str(data_dir),
            run_quality_checks=not args.no_quality_checks,
        ) as extractor:
            cutoff_ts = extractor.get_last_full_month_end()
            patient_cutoff = cutoff_ts.strftime("%Y-%m-%d")

            df = extractor.extract_patient_data()
            period_col = "period_end" if "period_end" in df.columns else "periodend"
            if not df.empty and period_col in df.columns:
                patient_max_periodend = (
                    pd.to_datetime(df[period_col]).max().strftime("%Y-%m-%d")
                )

            extractor.save_to_csv(df, filename="patient_data.csv")
    else:
        logger.info("Skipping patient ingestion")

    if not args.skip_staffing:
        logger.info("\n=== Staffing ingestion ===")
        with StaffingDataExtractor(
            config_path=args.config_path,
            output_dir=str(data_dir),
            run_quality_checks=not args.no_quality_checks,
        ) as extractor:
            df = extractor.extract_staffing_data()
            extractor.save_to_csv(df, filename="staffing_data.csv")
    else:
        logger.info("Skipping staffing ingestion")

    # Write a simple refresh marker for the dashboard
    try:
        out_path = _write_last_refreshed(
            data_dir=data_dir,
            patient_max_periodend=patient_max_periodend,
            patient_cutoff=patient_cutoff,
        )
        logger.info(f"🕒 Wrote refresh metadata: {out_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to write last_refreshed.json: {e}")

    logger.info("✅ Master ingestion complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
