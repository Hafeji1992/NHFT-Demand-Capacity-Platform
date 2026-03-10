# NHFT Demand & Capacity Dashboard
## Dash Application Documentation

---

## Overview

This folder contains the interactive Dash application for exploring NHFT demand (patient activity) and capacity (staffing mix) data.

- Primary audience: Operational analysts, service leads, managers, and project stakeholders.
- What it does: Applies date/service filters to the patient dataset, surfaces headline KPIs, and provides four tabs:
  - About This Dashboard
  - Overview
  - Demand Analysis
  - Capacity Analysis

The Capacity tab reuses the currently selected provider(s) from the patient filter to subset staffing.

---

## Folder Contents

- app.py
  Main Dash application (layout, callbacks, charts, tables, forecasting overlays, and About tab content).

- data_handler.py
  Loads and preprocesses data/patient_data.csv and data/staffing_data.csv (including column standardisation and date parsing).

---

## Data Requirements

The dashboard expects the ingestion pipeline to have produced:

- data/patient_data.csv
  Monthly demand/activity extract with, at minimum:
  - provider_code_current
  - service_line
  - period_end (parsed as a date; used to build the month slider)
  - Core metrics used in charts/KPIs: referrals, waiters, caseload, total_contacts, discharges_with_clock_stop
  - Additional metrics used in derived fields where present: clock_stop_actuals, discharges_no_clock_stop, waiters_over_18_weeks, waiters_under_18_weeks, and optional contact/discharge averages

- data/staffing_data.csv
  Workforce snapshot with:
  - provider_code_current
  - service_line
  - staff_group
  - staff (numeric)

Notes:
- Patient column headers are standardised to snake_case during load.
- Provider codes are treated as strings and normalised to 3 digits where applicable (for example, 6 becomes 006).
- CSV decoding uses a small encoding fallback chain (useful for Windows extracts).

---

## Running the Dashboard

From the project root:

1. Install dependencies:

   pip install -r requirements.txt

2. Run the Dash app:

   python src/dashboard/app.py

Notes:
- The dashboard is a long-running web server. After it prints the local URL (for example, http://127.0.0.1:8050), it will keep running until stopped.
- If you stop the process (Ctrl+C / VS Code Stop), you may see KeyboardInterrupt in the terminal. This is expected.

If you are already in src/dashboard, you can also run:

..\.\.venv\Scripts\python.exe app.py

3. Open the local URL shown in the console (default http://127.0.0.1:8050).

---

## App Structure and Interactions

### Filters (top of page)

- Date Range: Month-based range slider built from period_end values.
- Service: Multi-select dropdown displaying provider_code_current - service_line pairs.
- Clinician Patient Facing Time: Dropdown (50% to 100%, default 60%). This drives the Clock Stop Target in the Demand table as a percentile of each service line's historical monthly referrals (within the selected filter window).
- Apply Filters: Loads filtered patient data into app state so tabs and KPI cards update together.

### KPI Cards

After filters are applied, the dashboard shows headline KPI cards (sums across the selected date range):

- Referrals
- Waiters
- Caseload
- Contacts
- Discharges
- Total Staff (from staffing snapshot, filtered by selected provider)

Additional derived KPI cards (latest month):
- Demand Ratio: Throughput versus service-line target demand.
- Sustainable Caseload: Estimated caseload level if Demand Ratio equals 1.0.
- Net Caseload Flow: Inflow minus outflow.
- Caseload Throughput: Outflow divided by caseload.
- Clearance Time: Weeks required to clear current waiters at current first-contact rate.

---

## Tabs

### ℹ️ About This Dashboard

Provides an in-app methodology and interpretation guide covering:
- Dashboard purpose and audience
- Demand benchmarking approach
- Meaning of key operational metrics
- Forecasting approach and caveats
- Data sources, mapping, and refresh assumptions
- Data limitations and triangulation approach

### 📈 Overview

- Key Metrics Over Time: Multi-metric time series (Referrals, Waiters, Caseload, Contacts, Discharges, and related signals). Legend clicks hide/show series.
- Forecast toggle (per chart): Optional ETS (Holt-Winters) overlay with 12-month horizon and 95% confidence interval.
- Staff vs Caseload (Latest): Scatter chart comparing latest-month caseload against staff totals, with trendline.
- Waiting List Breakdown: Under vs Over 18 weeks by month.

### 👥 Demand Analysis

- Individual time series cards (each with its own forecast toggle): Referrals, Waiters, Caseload, Contacts, Discharges.
- Waiting List Breakdown (Under vs Over 18 weeks) by month.
- Patient Summary Table by Service Line: Grouped at provider_code_current x service_line x period_end with derived fields such as:
  - clock_stop_target (from Clinician Patient Facing Time setting)
  - demand ratio fields where source columns exist
  - contacts-per-caseload fields where source columns exist

### 💼 Capacity Analysis

- Capacity visuals filtered to provider(s) selected in patient filters.
- Charts:
  - Total Staff by Staff Group
  - Total Staff by Service Line
  - Staff vs Caseload (latest period)
- Staffing by Service Line table:
  - One row per provider_code_current x service_line
  - One column per staff_group

---

## Demand Modelling Approach

The dashboard uses demand benchmarking rather than fixed capacity targets.

How it works:
- A service-line Clock Stop Target is computed from the selected Clinician Patient Facing Time percentile of historical monthly referrals.
- Throughput is compared to this target to create Demand Ratio.
- This ratio is used to derive additional indicators such as Sustainable Caseload.

Interpretation guide:
- Demand Ratio >= 1.0 indicates throughput is broadly keeping pace with demand.
- Demand Ratio < 1.0 indicates a potential demand-capacity gap.
- Positive Net Caseload Flow with low Caseload Throughput suggests accumulating pressure.
- Rising Clearance Time suggests waiting list deterioration risk.

---

## Forecasting Notes

- Forecasts are optional and controlled by per-chart toggles.
- Forecast model: ETS / exponential smoothing (Holt-Winters).
- Forecast horizon: 12 months.
- Confidence interval: 95%.
- Forecasting is skipped for series that are too short or too flat.
- Practical guidance: include at least 24-36 monthly points for more stable seasonal forecasting.

---

## Data Limitations and Capacity Context

Two constraints shape current interpretation:

1. No historical workforce time series is currently available at the required granularity.
2. Staffing and patient service definitions do not naturally align at a fully equivalent service-line grain.

As a result:
- Capacity Analysis is presented as a contextual/reference view rather than a fully time-aligned capacity model.
- Demand benchmarking remains the primary analytical framework, with staffing used for triangulation.
