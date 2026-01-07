# NHFT Demand & Capacity Modelling Platform  
## Data Dictionary and Metadata Documentation

---

# Patient Data Dictionary

## Dataset Overview

- **Dataset name:** Patient Dataset  
- **System owner:** Northamptonshire Healthcare NHS Foundation Trust (NHFT)  
- **Data controller:** NHFT  
- **Purpose:**  
  This dataset supports demand forecasting, capacity modelling, simulation, and descriptive analytics for NHFT services as part of the NHFT Demand & Capacity Modelling Platform.  
- **Unit of observation:**  
  One record per **ProviderCodeCurrent × Service_Line × PeriodEnd** (monthly grain).  
- **Source systems/views:**
  - MIS_AG.dbo.Vw_tbl_ag_Report_WaitingTimes
  - MIS_AG.dbo.Vw_tbl_ag_Report_ContactAttendances
  - MIS_Config.dbo.tbl_org_current_RL9_Service_Line  
- **Refresh cadence:** Monthly (aligned to MIS reporting refresh).  
- **Ingestion cutoff (time series safety):** Patient data ingestion is capped at the **last fully completed month-end** to avoid partial-month periods in downstream time series analysis.  
  Example: If run on `2026-01-06`, the maximum `periodend` included will be `2025-12-31`.  
- **Storage location:** `data/patient_data.csv` (excluded from version control).  
- **Sensitivity classification:**  
  Aggregated operational activity data. No direct patient identifiers. Managed under NHFT Information Governance standards.

---

# Running Ingestion (Master Script)

Use the master runner to refresh both patient and staffing datasets and update `data/last_refreshed.json`:

```bash
python src/data_engineering/master_data_ingestion.py
```

Optional flags:
- `--skip-patient` / `--skip-staffing`
- `--no-quality-checks`
- `--config PATH_TO_CONFIG_INI`
- `--output-dir PATH_TO_DATA_DIR`

---

## Column Name Normalisation (SQL → DataFrame)

The ingestion scripts intentionally **normalise column names** after extraction.

- NHFT SQL sources often contain `PascalCase`, mixed separators, and symbols (e.g. `<`, `+`).
- For consistent table outputs in the dashboard and CSV exports, the ingested DataFrames use **snake_case** (lowercase with underscores).
- Downstream code (dashboard tables/plots, joins, tests) should rely on the **normalised** column names, not the raw SQL aliases.

Examples:

- `ProviderCodeCurrent` → `provider_code_current`
- `Service_Line` → `service_line`
- `PeriodEnd` → `period_end`
- `Waiters<18Weeks` / `WaitersUnder18Weeks` → `waiters_under_18_weeks`
- `Waiters18+Weeks` / `Waiters18PlusWeeks` → `waiters_over_18_weeks`

---

## Key Assumptions and Constraints

- Each `(provider_code_current, service_line, period_end)` combination should be unique.
- Some ratio fields may be `NULL` where denominators are zero.
- Zero referrals in a period do not imply zero activity in contacts or caseload.

---

## Data Quality Rules (Enforced in Ingestion)

- **Critical non-null fields:** `provider_code_current`, `service_line`, `period_end`
- **Logical consistency:**
  - `waiters = waiters_under_18_weeks + waiters_over_18_weeks`
  - `ftf_contacts ≤ total_contacts`
  - `ftf_caseload_contacts ≤ total_caseload_contacts`
- **Ranges:** Count-based fields must be ≥ 0
- **Uniqueness:** No duplicate records for the same provider, service line, and period

---

## Column-Level Data Dictionary

| Column | Description | Data Type | Constraints | Derivation / Notes |
|------|------------|-----------|-------------|-------------------|
| `provider_code_current` | Provider or service code used for reporting and joins. | string | Not null | Source: Waiting Times view. |
| `service_line` | Service line classification for the provider code. | string | Nullable | Derived from MIS_Config mapping table. |
| `period_end` | Month-end date for reporting period. | date | Not null | Monthly reporting grain. |
| `referrals` | Number of new referrals starting in the period. | integer | ≥ 0 | Count where `RefStartThisPeriod = 1`. |
| `clock_stop_actuals` | Referrals receiving first contact in the period. | integer | ≥ 0 | Count where `FirstContactInPeriod = 1`. |
| `discharges_no_clock_stop` | Discharges with no first contact recorded. | integer | ≥ 0 | Discharged = 1 AND FirstContact IS NULL. |
| `referral_clock_stop_ratio` | Ratio of clock stops to referrals. | float | Nullable | `clock_stop_actuals / referrals`. |
| `referral_discharged_no_clock_stop_ratio` | Ratio of discharges without clock stop to referrals. | float | Nullable | `discharges_no_clock_stop / referrals`. |
| `demand_ratio` | Ratio of referrals to clock stops. | float | Nullable | `referrals / clock_stop_actuals`. |
| `total_contacts` | Total contacts where patient was seen in period. | integer | ≥ 0 | PatientSeen = 1. |
| `ftf_contacts` | Face-to-face contacts in period. | integer | ≥ 0 | FTF attendance flags. |
| `caseload` | Active referrals on caseload (not waiting or discharged). | integer | ≥ 0 | Discharged = 0 AND WaitAssess = 0. |
| `total_caseload_contacts` | Contacts after first contact (caseload activity). | integer | ≥ 0 | Contact_Date > FirstContact. |
| `ftf_caseload_contacts` | Face-to-face caseload contacts. | integer | ≥ 0 | Subset of caseload contacts. |
| `total_contacts_per_caseload` | Average contacts per caseload. | float | Nullable | `total_caseload_contacts / caseload`. |
| `ftf_contacts_per_caseload` | Average FTF contacts per caseload. | float | Nullable | `ftf_caseload_contacts / caseload`. |
| `waiters` | Total referrals waiting for assessment/treatment. | integer | ≥ 0 | RTT included, exclusions removed. |
| `waiters_under_18_weeks` | Waiting referrals under 18 weeks. | integer | ≥ 0 | Split of `waiters` by waiting-time threshold. |
| `waiters_over_18_weeks` | Waiting referrals over 18 weeks. | integer | ≥ 0 | Split of `waiters` by waiting-time threshold. |
| `average_length_of_treatment` | Average days from referral start to first contact (discharged cases). | float | Nullable | AVG datediff for discharged referrals. |
| `average_contacts_at_discharge` | Average contacts per referral at discharge. | float | Nullable | AVG ContactsPerReferral (discharged). |
| `average_ftf_contacts_at_discharge` | Average FTF contacts per referral at discharge. | float | Nullable | AVG ContactsFTFPerReferral (discharged). |
| `discharges_from_caseload` | Discharges with a recorded first contact. | integer | ≥ 0 | DischargesWithClockStop. |

---

## Change Log (Patient Data)

| Date       | Change                                      | Author |
|------------|---------------------------------------------|--------|
| 2025-12-17 | Initial creation of patient data dictionary | Yahya Hafeji |
| 2026-01-07 | Updatead variables to Snake Case for better readability front-end | Yahya Hafeji |

---

## Notes for Analysts and Modellers (Patient Data)

- Ratio fields should be interpreted cautiously in low-volume services.
- Caseload-based metrics assume stable service definitions across periods.
- This dataset is designed for trend analysis and modelling, not patient-level decision making.

---

# Staffing Data Dictionary

## Dataset Overview

- **Dataset name:** Staffing Dataset  
- **System owner:** Northamptonshire Healthcare NHS Foundation Trust (NHFT)  
- **Data controller:** NHFT  
- **Purpose:**  
  This dataset represents workforce capacity and supports demand and capacity modelling, staffing mix analysis, and descriptive workforce analytics as part of the NHFT Demand & Capacity Modelling Platform.  
- **Unit of observation:**  
  One record per **ProviderCodeCurrent × Service_Line × Staff_Group**.  
- **Source system/table:**
  - ESR.dbo.tbl_dt_Appraisal_Review_Detail  
- **Refresh cadence:**  
  Periodic extract aligned to ESR data refresh (typically monthly or ad-hoc for modelling).  
- **Storage location:**  
  `data/staffing_data.csv` (excluded from version control).  
- **Sensitivity classification:**  
  Aggregated workforce data. No personally identifiable staff information. Managed under NHFT Information Governance standards.

---

## Key Modelling Decisions

- `ProviderCodeCurrent` is treated as a service code, not a standalone provider entity.
- `Service_Line` is the primary analytical unit for capacity modelling.
- No time dimension is included in the current staffing extract.
- Staffing counts represent headcount (distinct assignments), not WTE.

---

## Key Assumptions and Constraints

- Each `(provider_code_current, service_line, staff_group)` combination should be unique.
- A staff member contributes to only one service line per assignment.
- Staffing counts reflect headcount, not contracted hours.
- Staff group classifications are assumed to be consistent across ESR extracts.

---

## Data Quality Rules (Enforced in Ingestion)

- **Critical non-null fields:** `provider_code_current`, `service_line`, `staff_group`, `staff`
- **Ranges:** `staff ≥ 0`
- **Uniqueness:** No duplicate records for the same service line and staff group
- **Consistency:** Service line names are derived from ESR organisational hierarchy text

---

## Column-Level Data Dictionary

| Column | Description | Data Type | Constraints | Derivation / Notes |
|------|------------|-----------|-------------|-------------------|
| `provider_code_current` | Service code extracted from ESR organisational hierarchy. | string | Not null | Parsed from `Org L6` (L5 code). |
| `service_line` | Service line name derived from ESR organisational hierarchy. | string | Not null | Text following L5 service code in `Org L6`. |
| `staff_group` | ESR-defined staff group classification. | string | Not null | Source: ESR Staff Group field. |
| `staff` | Count of distinct staff assignments. | integer | ≥ 0 | `COUNT(DISTINCT Assignment Number)`. |

---

## Relationship to Patient Dataset

- Staffing data is designed to join to patient data at the service line level.
- Typical joins use `provider_code_current`.
- Staffing represents capacity snapshots; patient data represents demand over time.

---

## Analytical Use Cases

- Service-level capacity benchmarking
- Workforce mix analysis by staff group
- Demand-to-capacity ratio modelling
- Scenario testing and bottleneck identification

---

## Change Log (Staffing Data)

| Date       | Change                                       | Author |
|------------|----------------------------------------------|--------|
| 2025-12-22 | Initial creation of staffing data dictionary | Yahya Hafeji |
| 2026-01-07 | Updatead variables to Snake Case for better readability front-end | Yahya Hafeji |

---

## Notes for Analysts and Modellers (Staffing Data)

- Staffing counts should not be interpreted as WTE without adjustment.
- Capacity modelling should normalise staffing against activity volumes.
- Future enhancements may include WTE, role-level capacity, or time-stamped staffing snapshots.
