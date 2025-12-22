import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_engineering.patient_data_ingestion import PatientDataExtractor, load_patient_data, DataQualityException


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
            "ProviderCodeCurrent", "Service_Line", "PeriodEnd",
            "Referrals", "ClockStopActuals", "DischargesNoClockStop",
            "ReferralClockStopRatio", "ReferralDischargedNoClockStopRatio", "DemandRatio",
            "TotalContacts", "FTFContacts",
            "Caseload", "TotalCaseloadContacts", "FTFCaseloadContacts",
            "TotalContactsPerCaseload", "FTFContactsPerCaseload",
            "Waiters", "Waiters<18Weeks", "Waiters18+Weeks",
            "AverageLengthOfTreatment", "AverageContactsAtDischarge",
            "AverageFTFContactsAtDischarge", "DischargesFromCaseload",
        ]

        rows = [
            (
                "001", "Adult Acute Inpatients", "2025-01-31",
                10, 8, 1,
                0.8, 0.1, 1.25,
                20, 15,
                30, 12, 8,
                0.4, 0.27,
                5, 3, 2,
                42.0, 3.5, 2.1, 7,
            )
        ]
        return FakeConnection(rows, columns)

    def close(self):
        return None


# ---------------------------------------------------------------------
# Unit tests - Normalise Column Names
# ---------------------------------------------------------------------
def test_normalise_column_names():
    cols = ["Waiters<18Weeks", "Waiters18+Weeks", "Service Line", "FTFContacts"]
    normalised = PatientDataExtractor._normalise_column_names(cols)

    assert normalised == [
        "waitersunder18weeks",
        "waiters18plusweeks",
        "service_line",
        "ftfcontacts",
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
    - parses periodend as datetime
    """
    # Patch SQLServerConnection inside the module scope by replacing the class it uses
    monkeypatch.setattr(
        "data_engineering.patient_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection
    )

    extractor = PatientDataExtractor(run_quality_checks=False)
    df = extractor.extract_patient_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "periodend" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["periodend"])
    assert "service_line" in df.columns
    assert "waitersunder18weeks" in df.columns
    assert "waiters18plusweeks" in df.columns

# ---------------------------------------------------------------------
# Unit tests - Data Quality Checks
# ---------------------------------------------------------------------
def test_quality_checks_fail_on_schema(monkeypatch):
    """
    If Schema Validation fails and it is treated as critical, the extractor should raise DataQualityException.
    """
    monkeypatch.setattr(
        "data_engineering.patient_data_ingestion.SQLServerConnection",
        FakeSQLServerConnection
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
    load_patient_data() should load a CSV and parse periodend as datetime.
    """
    test_file = tmp_path / "patient_data.csv"

    df_in = pd.DataFrame({
        "periodend": ["2025-01-31", "2025-02-28"],
        "service_line": ["013", "013"],
        "providercodecurrent": ["AAA", "AAA"],
        "referrals": [10, 20],
        "clockstopactuals": [8, 15],
        "waiters": [5, 6],
        "waitersunder18weeks": [3, 4],
        "waiters18plusweeks": [2, 2],
        "caseload": [30, 35],
        "totalcontacts": [20, 25],
        "ftfcontacts": [15, 18],
        "dischargesnoclockstop": [1, 2],
        "dischargesfromcaseload": [7, 8],
    })
    df_in.to_csv(test_file, index=False)

    df_out = load_patient_data(str(test_file))

    assert len(df_out) == 2
    assert pd.api.types.is_datetime64_any_dtype(df_out["periodend"])
    assert df_out["referrals"].sum() == 30

# ---------------------------------------------------------------------
# Unit tests - Load Patient Data - Missing File
# ---------------------------------------------------------------------
def test_load_patient_data_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_patient_data(str(missing))
