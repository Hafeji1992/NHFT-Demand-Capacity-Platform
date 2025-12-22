import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

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
    """Mock replacement for SQLServerConnection."""
    def __init__(self, _config_path=None):
        pass

    def connect(self):
        columns = [
            "Staff",
            "Staff Group",
            "ProviderCodeCurrent",
            "Service_Line",
        ]

        rows = [
            (12, "Nursing", "001", "Adult Inpatients Acute"),
            (8, "Medical", "001", "Adult Inpatients Acute"),
        ]

        return FakeConnection(rows, columns)

    def close(self):
        pass


# ---------------------------------------------------------------------
# Unit Test - Normalise Column Names
# ---------------------------------------------------------------------
def test_normalise_column_names():
    cols = ["Staff Group", "Service Line", "ProviderCodeCurrent"]
    normalised = StaffingDataExtractor._normalise_column_names(cols)

    assert normalised == [
        "staff_group",
        "service_line",
        "providercodecurrent",
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
        "staff",
        "staff_group",
        "providercodecurrent",
        "service_line",
    }
    assert set(df.columns) == expected_cols
    assert df["staff"].sum() == 20

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

    df_in = pd.DataFrame({
        "staff": [10, 5],
        "staff_group": ["Nursing", "Medical"],
        "providercodecurrent": ["001", "001"],
        "service_line": ["Adult Inpatients Acute", "Adult Inpatients Acute"],
    })
    df_in.to_csv(test_file, index=False)

    df_out = load_staffing_data(test_file)

    assert len(df_out) == 2
    assert df_out["staff"].sum() == 15

# ---------------------------------------------------------------------
# Unit Test - load_staffing_data missing file
# ---------------------------------------------------------------------
def test_load_staffing_data_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_staffing_data(missing)
