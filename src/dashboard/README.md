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

- assets/nhs-theme.css
  Custom CSS theme implementing the NHS Design System colour palette and component styles. Dash automatically loads all files under assets/. Defines styling for the header banner, cards, metric cards, tabs, buttons, filter bar, info boxes, footer, and DataTable headers.

---

## Visual Theme

The dashboard uses the official NHS Design System colour palette (NHS Blue #005EB8, NHS Dark Blue #003087, NHS Aqua Green #00A499, etc.) applied through CSS custom properties in assets/nhs-theme.css. All charts also use an NHS-branded colour sequence.

The header displays the NHFT.svg trust logo with an NHS.jpg overlay positioned over the NHS lozenge area. Images are served via a Flask route from the project-level images/ directory.

---

## Data Requirements

The dashboard expects the ingestion pipeline to have produced:

- data/patient_data.csv
  Monthly demand/activity extract with, at minimum:
  - provider_code_current
  - service_line
  - period_end (parsed as a date; used to build the month slider)
  - Core metrics used in charts/KPIs: referrals, waiters, caseload, total_contacts, discharges_with_clock_stop
  - Additional metrics used in derived fields where present: clock_stop_actuals, discharges_no_clock_stop, waiters_over_18_weeks, waiters_under_18_weeks
  - Optional discharge quality metrics (used in Demand tab KPI cards): average_length_of_treatment, average_contacts_at_discharge, average_ftf_contacts_at_discharge
  - Optional contact breakdown metrics (used in demand table ratios): ftf_caseload_contacts, total_caseload_contacts, ftf_contacts

- data/staffing_data.csv
  Workforce snapshot with:
  - provider_code_current
  - service_line
  - staff_group
  - staff (numeric)

Notes:
- Patient column headers are standardised to snake_case during load.
- Provider codes are treated as strings and normalised to 3 digits where applicable (for example, 6 becomes 006). This preserves leading zeros after JSON serialisation/deserialisation.
- CSV decoding uses a UTF-8 → UTF-8-sig → cp1252 → latin1 encoding fallback chain (useful for Windows extracts).

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

- Date Range: Month-based range slider built from period_end values. Dynamic marks highlight April of each year plus the endpoints. A label below the slider updates to show the selected range (for example, Jan 2023 to Dec 2024).
- Service: Multi-select dropdown displaying provider_code_current - service_line pairs.
- Clinician Patient Facing Time: Dropdown (50% to 100% in 5% increments, default 60%). This drives the Clock Stop Target in the Demand table as a percentile of each service line's historical monthly referrals (within the selected filter window). Changing this value recalculates demand ratio and sustainable caseload across all tabs.
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

Provides an in-app methodology and interpretation guide with the following sections:
- Purpose: What the dashboard does and who it is for.
- How Demand Is Measured: Explains the demand benchmarking approach and how the Clock Stop Target is derived.
- Key Operational Metrics: Definitions and interpretation guidance for Caseload Throughput, Net Caseload Flow, Clearance Time, Demand Ratio, and Sustainable Caseload.
- Tabs in the Dashboard: Summary of what each tab contains and how to navigate.
- Forecasting: Model description (ETS/Holt-Winters), horizon, confidence intervals, and practical guidance on minimum data.
- Data Sources & Refresh: Where data originates, mapping assumptions, and refresh expectations.
- Data Limitations, Triangulation & Capacity Approach: Known constraints around workforce time series and service-line alignment, and how the dashboard addresses them.

### 📈 Overview

- Derived KPI Cards (latest month): Demand Ratio, Sustainable Caseload, Net Caseload Flow, Caseload Throughput, and Clearance Time (same metrics shown in the header cards, repeated here for quick reference).
- Key Metrics Over Time: Multi-metric time series (Referrals, Waiters, Caseload, Contacts, Discharges, and related signals). Legend clicks toggle individual series; double-clicking a legend item isolates it.
- Forecast toggle (per chart): Optional ETS (Holt-Winters) overlay with 12-month horizon and 95% confidence interval.
- Staff vs Caseload (Latest): Scatter chart comparing latest-month caseload against staff totals, with trendline.
- Waiting List Breakdown: Under vs Over 18 weeks by month.

### 👥 Demand Analysis

- Individual time series cards (each with its own forecast toggle): Referrals, Waiters, Caseload, Contacts, Discharges.
- Net Caseload Flow Over Time: Monthly time series showing net caseload flow (inflow minus outflow) by service line.
- Discharge Quality KPI Cards (latest month, where data is available):
  - Avg Treatment Length: Average days from referral to first contact for discharged patients.
  - Avg Contacts at Discharge: Average total contacts per patient at discharge.
  - Avg FTF Contacts at Discharge: Average face-to-face contacts per patient at discharge.
- Waiting List Breakdown (Under vs Over 18 weeks) by month.
- Patient Summary Table by Service Line: Grouped at provider_code_current x service_line x period_end with derived fields:
  - clock_stop_target: Computed from the Clinician Patient Facing Time percentile setting (rounded to whole numbers).
  - referral_clock_stop_ratio: Clock-stop discharges divided by referrals.
  - referral_discharged_no_clock_stop_ratio: Non-clock-stop discharges divided by referrals.
  - demand_ratio: Sum of the two ratios above (ratios rounded to 3 decimal places).
  - total_contacts_per_caseload: Total contacts divided by caseload (where source columns exist).
  - ftf_contacts_per_caseload: Face-to-face contacts divided by caseload (where source columns exist).

### 💼 Capacity Analysis

- Capacity visuals filtered to provider(s) selected in patient filters.
- Charts:
  - Total Staff by Staff Group: Horizontal bar chart showing aggregate staff count per staff group.
  - Total Staff by Service Line: Horizontal bar chart showing aggregate staff count per service line.
  - Staff vs Caseload (latest period): Scatter chart comparing latest-month caseload against staff totals per service, with OLS trendline and colour coding by staff-per-100-caseload ratio.
- Staffing by Service Line table:
  - One row per provider_code_current x service_line.
  - One column per staff_group (staff values summed when aggregating).

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
- Minimum data requirement: 12 monthly data points. If fewer are available, the chart displays an annotation ("Forecast needs ≥12 months of history") and no forecast is generated.
- Practical guidance: include at least 24-36 monthly points for more stable seasonal forecasting.

---

## Data Limitations and Capacity Context

Two constraints shape current interpretation:

1. No historical workforce time series is currently available at the required granularity.
2. Staffing and patient service definitions do not naturally align at a fully equivalent service-line grain.

As a result:
- Capacity Analysis is presented as a contextual/reference view rather than a fully time-aligned capacity model.
- Demand benchmarking remains the primary analytical framework, with staffing used for triangulation.

---

## Footer

The page footer dynamically shows the Latest Reporting Period End date, derived from the most recent period_end value in the filtered patient data.
