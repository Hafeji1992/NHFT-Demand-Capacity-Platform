import sys
from pathlib import Path

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

from data_engineering.patient_data_ingestion import (
    DataQualityException,
    PatientDataExtractor,
    load_patient_data,
)


# ---------------------------------------------------------------------
# Fake DB infrastructure
# ---------------------------------------------------------------------
class FakeCursor:
    def __init__(self, rows, columns):
        self._rows = rows
        self.description = [(c,) for c in columns]
        self._closed = False

    def execute(self, _query):
        return None

    def fetchall(self):
        return self._rows

    def close(self):
        self._closed = True


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
            "period_end",
            "referrals",
            "clock_stop_actuals",
            "discharges_no_clock_stop",
            "total_contacts",
            "ftf_contacts",
            "caseload",
            "total_caseload_contacts",
            "ftf_caseload_contacts",
            "waiters",
            "waiters_under_18_weeks",
            "waiters_over_18_weeks",
            "average_length_of_treatment",
            "average_contacts_at_discharge",
            "average_ftf_contacts_at_discharge",
            "discharges_from_caseload",
        ]

        rows = [
            (
                "001",
                "Adult Acute Inpatients",
                "2025-01-31",
                10,
                8,
                1,
                20,
                15,
                30,
                12,
                8,
                5,
                3,
                2,
                42.0,
                3.5,
                2.1,
                7,
            )
        ]
        return FakeConnection(rows, columns)

    def close(self):
        return None


# ---------------------------------------------------------------------
# Unit tests - Normalise Column Names
# ---------------------------------------------------------------------
def test_normalise_column_names():
    cols = ["WaitersUnder18Weeks", "WaitersOver18Weeks", "Service_Line", "FTFContacts"]
    normalised = PatientDataExtractor._normalise_column_names(cols)

    assert normalised == [
        "waiters_under_18_weeks",
        "waiters_over_18_weeks",
        "service_line",
        "ftf_contacts",
    ]


# ---------------------------------------------------------------------
# Unit tests - Extract Patient Data
# ---------------------------------------------------------------------
def test_extract_patient_data_with_mocked_db(monkeypatch):
    """
    Ensures extract_patient_data:
    - executes the query
    - returns a DataFrame
    - normalises column names
    - parses period_end as datetime
    """
    # Patch SQLServerConnection inside the module scope by replacing the class it uses
    monkeypatch.setattr(
        "data_engineering.patient_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection,
    )

    extractor = PatientDataExtractor(run_quality_checks=False)
    df = extractor.extract_patient_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "period_end" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["period_end"])
    assert "service_line" in df.columns
    assert "waiters_under_18_weeks" in df.columns
    assert "waiters_over_18_weeks" in df.columns


# ---------------------------------------------------------------------
# Unit tests - Data Quality Checks
# ---------------------------------------------------------------------
def test_quality_checks_fail_on_schema(monkeypatch):
    """
    If Schema Validation fails and it is treated as critical, the extractor should raise DataQualityException.
    """
    monkeypatch.setattr(
        "data_engineering.patient_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection,
    )

    extractor = PatientDataExtractor(run_quality_checks=True)

    # Force schema check to fail by removing an expected column after extraction
    df = extractor.extract_patient_data()
    df = df.drop(columns=["referrals"])

    # Run checks directly and confirm schema fails
    results = extractor.run_data_quality_checks(df)
    assert "Schema Validation" in results["failed"]

    # Simulate the same critical-fail behaviour by calling _check_schema and raising like extract_patient_data does
    is_valid, _issues = extractor._check_schema(df)
    assert is_valid is False


# ---------------------------------------------------------------------
# Unit tests - Load Patient Data from CSV
# ---------------------------------------------------------------------
def test_load_patient_data_reads_csv(tmp_path, monkeypatch):
    """
    load_patient_data() should load a CSV and parse period_end as datetime.
    """
    test_file = tmp_path / "patient_data.csv"

    df_in = pd.DataFrame(
        {
            "period_end": ["2025-01-31", "2025-02-28"],
            "service_line": ["Adult Inpatients Acute", "Adult Inpatients Acute"],
            "provider_code_current": ["013", "013"],
            "referrals": [10, 20],
            "clock_stop_actuals": [8, 15],
            "waiters": [5, 6],
            "waiters_under_18_weeks": [3, 4],
            "waiters_over_18_weeks": [2, 2],
            "caseload": [30, 35],
            "total_contacts": [20, 25],
            "ftf_contacts": [15, 18],
            "discharges_no_clock_stop": [1, 2],
            "discharges_from_caseload": [7, 8],
        }
    )
    df_in.to_csv(test_file, index=False)

    df_out = load_patient_data(str(test_file))

    assert len(df_out) == 2
    assert pd.api.types.is_datetime64_any_dtype(df_out["period_end"])
    assert df_out["referrals"].sum() == 30


# ---------------------------------------------------------------------
# Unit tests - Load Patient Data - Missing File
# ---------------------------------------------------------------------
def test_load_patient_data_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_patient_data(str(missing))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
