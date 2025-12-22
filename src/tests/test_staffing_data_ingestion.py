import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is on the Python path so `data_engineering.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_engineering.staffing_data_ingestion import (
    StaffingDataExtractor,
    load_staffing_data,
    DataQualityException
)


# ---------------------------------------------------------------------
# Test helpers (fake DB connection/cursor)
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


class FakeConnection:
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = columns

    def cursor(self):
        return FakeCursor(self._rows, self._columns)


class FakeSQLServerConnection:
    """Replaces SQLServerConnection in tests."""
    def __init__(self, _config_path=None):
        pass

    def connect(self):
        columns = [
            "Staff",
            "Staff Group",
            "Service_Line"
        ]

        rows = [
            (25, "Nursing", "Adult Inpatients Acute"),
            (10, "Medical", "Adult Inpatients Acute"),
        ]

        return FakeConnection(rows, columns)

    def close(self):
        return None


# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------
def test_normalise_column_names():
    cols = ["Staff Group", "Service_Line", "Staff"]
    normalised = StaffingDataExtractor._normalise_column_names(cols)

    assert normalised == [
        "staff_group",
        "service_line",
        "staff",
    ]


def test_extract_staffing_data_with_mocked_db(monkeypatch):
    """
    Ensures extract_staffing_data:
    - executes the query
    - returns a DataFrame
    - normalises column names
    """
    monkeypatch.setattr(
        "data_engineering.staffing_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection
    )

    extractor = StaffingDataExtractor(run_quality_checks=False)
    df = extractor.extract_staffing_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) == {"staff", "staff_group", "service_line"}
    assert df["staff"].sum() == 35


def test_quality_checks_fail_on_schema(monkeypatch):
    """
    Schema Validation should fail if an expected column is missing.
    """
    monkeypatch.setattr(
        "data_engineering.staffing_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection
    )

    extractor = StaffingDataExtractor(run_quality_checks=True)
    df = extractor.extract_staffing_data()

    # Force schema failure
    df = df.drop(columns=["staff"])

    results = extractor.run_data_quality_checks(df)
    assert "Schema Validation" in results["failed"]

    is_valid, _ = extractor._check_schema(df)
    assert is_valid is False


def test_load_staffing_data_reads_csv(tmp_path):
    """
    load_staffing_data() should load a CSV successfully.
    """
    test_file = tmp_path / "staffing_data.csv"

    df_in = pd.DataFrame({
        "staff": [12, 18],
        "staff_group": ["Nursing", "Medical"],
        "service_line": ["Adult Acute", "Adult Acute"]
    })
    df_in.to_csv(test_file, index=False)

    df_out = load_staffing_data(str(test_file))

    assert len(df_out) == 2
    assert df_out["staff"].sum() == 30
    assert "staff_group" in df_out.columns


def test_load_staffing_data_missing_file_raises(tmp_path):
    """
    load_staffing_data should raise FileNotFoundError if file does not exist.
    """
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_staffing_data(str(missing))
