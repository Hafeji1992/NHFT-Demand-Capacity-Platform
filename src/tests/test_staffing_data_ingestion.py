import sys
from pathlib import Path
import logging

import pandas as pd
import pytest

# Ensure imports work for both:
# - `from data_engineering...` (package import)
# - ingestion scripts that do `from connect import ...` (module in data_engineering folder)
#
# Note: tests live in `src/tests`, so the repo root is `parents[2]`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATA_ENGINEERING_ROOT = SRC_ROOT / "data_engineering"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(DATA_ENGINEERING_ROOT))

# Avoid noisy logging errors under pytest capture on Windows
logging.disable(logging.CRITICAL)

from data_engineering.staffing_data_ingestion import (
    StaffingDataExtractor,
    load_staffing_data,
)


# ---------------------------------------------------------------------
# Fake DB infrastructure
# ---------------------------------------------------------------------
class FakeCursor:
    def __init__(self, rows, columns):
        self._rows = rows
        self.description = [(c,) for c in columns]

    def execute(self, _query):
        return None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


# ---------------------------------------------------------------------
# Fake DB connection
# ---------------------------------------------------------------------
class FakeConnection:
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = columns

    def cursor(self):
        return FakeCursor(self._rows, self._columns)


# ---------------------------------------------------------------------
# Fake SQLServerConnection
# ---------------------------------------------------------------------
class FakeSQLServerConnection:
    """Replaces SQLServerConnection in tests."""

    def __init__(self, _config_path=None):
        pass

    def connect(self):
        # Build a minimal dataset that aligns with EXPECTED_COLUMNS after normalisation
        columns = [
            "provider_code_current",
            "service_line",
            "staff_group",
            "staff",
        ]

        rows = [
            ("008", "Early Intervention Service", "Allied Health Professionals", 2),
            (
                "008",
                "Early Intervention Service",
                "Nursing and Midwifery Registered",
                14,
            ),
        ]

        return FakeConnection(rows, columns)

    def close(self):
        pass


# ---------------------------------------------------------------------
# Unit Test - Normalise Column Names
# ---------------------------------------------------------------------
def test_normalise_column_names():
    cols = ["ProviderCodeCurrent", "Service_Line", "StaffGroup"]
    normalised = StaffingDataExtractor._normalise_column_names(cols)

    assert normalised == [
        "provider_code_current",
        "service_line",
        "staff_group",
    ]


# ---------------------------------------------------------------------
# Unit Tests - StaffingDataExtractor
# ---------------------------------------------------------------------
def test_extract_staffing_data_with_mocked_db(monkeypatch):
    """
    Ensures extract_staffing_data:
    - executes the query
    - returns a DataFrame
    - normalises column names
    """
    monkeypatch.setattr(
        "data_engineering.staffing_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection,
    )

    extractor = StaffingDataExtractor(run_quality_checks=False)
    df = extractor.extract_staffing_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    expected_cols = {
        "provider_code_current",
        "service_line",
        "staff_group",
        "staff",
    }
    assert set(df.columns) == expected_cols
    assert df["staff"].sum() == 16


# ---------------------------------------------------------------------
# Unit Tests - Schema Validation
# ---------------------------------------------------------------------
def test_schema_validation_failure(monkeypatch):
    """
    Schema validation should fail if expected columns are missing.
    """
    monkeypatch.setattr(
        "data_engineering.staffing_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection,
    )

    extractor = StaffingDataExtractor(run_quality_checks=True)
    df = extractor.extract_staffing_data()

    # Force schema failure
    df = df.drop(columns=["staff"])

    results = extractor.run_data_quality_checks(df)

    assert "Schema Validation" in results["failed"]


# ---------------------------------------------------------------------
# Unit Tests - Duplicate Detection
# ---------------------------------------------------------------------
def test_duplicate_detection(monkeypatch):
    """
    Duplicate provider/service/staff_group combinations should be detected.
    """
    monkeypatch.setattr(
        "data_engineering.staffing_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection,
    )

    extractor = StaffingDataExtractor(run_quality_checks=True)
    df = extractor.extract_staffing_data()

    # Introduce a duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    results = extractor.run_data_quality_checks(df)
    assert "Duplicate Detection" in results["failed"]


# ---------------------------------------------------------------------
# Unit Tests - load_staffing_data
# ---------------------------------------------------------------------
def test_load_staffing_data_reads_csv(tmp_path):
    """
    load_staffing_data() should load CSV correctly.
    """
    test_file = tmp_path / "staffing_data.csv"

    df_in = pd.DataFrame(
        {
            "provider_code_current": ["008", "008"],
            "service_line": [
                "Early Intervention Service",
                "Early Intervention Service",
            ],
            "staff_group": [
                "Allied Health Professionals",
                "Nursing and Midwifery Registered",
            ],
            "staff": [2, 14],
        }
    )
    df_in.to_csv(test_file, index=False)

    df_out = load_staffing_data(test_file)

    assert len(df_out) == 2
    assert df_out["staff"].sum() == 16


# ---------------------------------------------------------------------
# Unit Test - load_staffing_data missing file
# ---------------------------------------------------------------------
def test_load_staffing_data_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_staffing_data(missing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
