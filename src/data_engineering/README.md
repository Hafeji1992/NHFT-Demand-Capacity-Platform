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
- **Storage location:** `data/patient_data.csv` (excluded from version control).
- **Sensitivity classification:**  
  Aggregated operational activity data. No direct patient identifiers. Managed under NHFT Information Governance standards.

---

## Key Assumptions and Constraints

- Each `(providercodecurrent, service_line, periodend)` combination should be unique.
- Some ratio fields may be `NULL` where denominators are zero.
- Zero referrals in a period do not imply zero activity in contacts or caseload.

---

## Data Quality Rules (Enforced in Ingestion)

- **Critical non-null fields:** `providercodecurrent`, `service_line`, `periodend`
- **Logical consistency:**
  - `waiters = waitersunder18weeks + waiters18plusweeks`
  - `ftfcontacts ≤ totalcontacts`
  - `ftfcaseloadcontacts ≤ totalcaseloadcontacts`
- **Ranges:** Count-based fields must be ≥ 0
- **Uniqueness:** No duplicate records for the same provider, service line, and period

---

## Column-Level Data Dictionary

| Column | Description | Data Type | Constraints | Derivation / Notes |
|------|------------|-----------|-------------|-------------------|
| `providercodecurrent` | Provider or service code used for reporting and joins. | string | Not null | Source: Waiting Times view. |
| `service_line` | Service line classification for the provider code. | string | Nullable | Derived from MIS_Config mapping table. |
| `periodend` | Month-end date for reporting period. | date | Not null | Monthly reporting grain. |
| `referrals` | Number of new referrals starting in the period. | integer | ≥ 0 | Count where `RefStartThisPeriod = 1`. |
| `clockstopactuals` | Referrals receiving first contact in the period. | integer | ≥ 0 | Count where `FirstContactInPeriod = 1`. |
| `dischargesnoclockstop` | Discharges with no first contact recorded. | integer | ≥ 0 | Discharged = 1 AND FirstContact IS NULL. |
| `referralclockstopratio` | Ratio of clock stops to referrals. | float | Nullable | `clockstopactuals / referrals`. |
| `referraldischargednoclockstopratio` | Ratio of discharges without clock stop to referrals. | float | Nullable | `dischargesnoclockstop / referrals`. |
| `demandratio` | Ratio of referrals to clock stops. | float | Nullable | `referrals / clockstopactuals`. |
| `totalcontacts` | Total contacts where patient was seen in period. | integer | ≥ 0 | PatientSeen = 1. |
| `ftfcontacts` | Face-to-face contacts in period. | integer | ≥ 0 | FTF attendance flags. |
| `caseload` | Active referrals on caseload (not waiting or discharged). | integer | ≥ 0 | Discharged = 0 AND WaitAssess = 0. |
| `totalcaseloadcontacts` | Contacts after first contact (caseload activity). | integer | ≥ 0 | Contact_Date > FirstContact. |
| `ftfcaseloadcontacts` | Face-to-face caseload contacts. | integer | ≥ 0 | Subset of caseload contacts. |
| `totalcontactspercaseload` | Average contacts per caseload. | float | Nullable | `totalcaseloadcontacts / caseload`. |
| `ftfcontactspercaseload` | Average FTF contacts per caseload. | float | Nullable | `ftfcaseloadcontacts / caseload`. |
| `waiters` | Total referrals waiting for assessment/treatment. | integer | ≥ 0 | RTT included, exclusions removed. |
| `waitersunder18weeks` | Waiting referrals under 18 weeks. | integer | ≥ 0 | Derived: `waiters - waiters18plusweeks`. |
| `waiters18plusweeks` | Waiting referrals over 18 weeks. | integer | ≥ 0 | WaitingTime > 18 weeks. |
| `averagelengthoftreatment` | Average days from referral start to first contact (discharged cases). | float | Nullable | AVG datediff for discharged referrals. |
| `averagecontactsatdischarge` | Average contacts per referral at discharge. | float | Nullable | AVG ContactsPerReferral (discharged). |
| `averageftfcontactsatdischarge` | Average FTF contacts per referral at discharge. | float | Nullable | AVG ContactsFTFPerReferral (discharged). |
| `dischargesfromcaseload` | Discharges with a recorded first contact. | integer | ≥ 0 | DischargesWithClockStop. |

---

## Change Log

| Date       | Change                                      | Author |
|------------|---------------------------------------------|--------|
| 2025-12-17 | Initial creation of patient data dictionary | Yahya Hafeji |

---

## Notes for Analysts and Modellers

- Ratio fields should be interpreted cautiously in low-volume services.
- Caseload-based metrics assume stable service definitions across periods.
- This dataset is designed for **trend analysis and modelling**, not patient-level decision making.

---
