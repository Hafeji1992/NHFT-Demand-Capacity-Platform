# NHFT Demand & Capacity Dashboard
## Dash Application Documentation

---

## Overview

This folder contains the interactive **Dash** application for exploring NHFT demand (patient activity) and capacity (staffing mix) data.

- **Primary audience:** Operational analysts and project stakeholders.
- **What it does:** Apply a date/service filter to the patient dataset, surface headline KPIs, then provide three analysis tabs (Overview, Demand, Capacity). The Capacity tab reuses the currently selected provider(s) from the patient filter to subset staffing.

---

## Folder Contents

- `app.py`
  Main Dash application (layout, callbacks, charts, tables, optional forecasting overlays).

- `data_handler.py`
  Loads and preprocesses `data/patient_data.csv` and `data/staffing_data.csv` (including column standardisation and basic date parsing).

---

## Data Requirements

The dashboard expects the ingestion pipeline to have produced:

- `data/patient_data.csv`
  Monthly demand/activity extract with, at minimum:
  - `provider_code_current`
  - `service_line`
  - `period_end` (parsed as a date; used to build the month slider)
  - Metrics used in charts/KPIs (expected): `referrals`, `waiters`, `caseload`, `total_contacts`, `discharges_with_clock_stop`
  - Metrics used in derived table fields when present: `clock_stop_actuals`, `discharges_no_clock_stop`, `waiters_over_18_weeks`, `waiters_under_18_weeks` (under-18 will be derived if missing and `waiters` + `waiters_over_18_weeks` exist), plus optional contact/discharge averages.

- `data/staffing_data.csv`
  Workforce snapshot with:
  - `provider_code_current`
  - `service_line`
  - `staff_group`
  - `staff` (numeric)

Notes:
- Patient column headers are standardised to `snake_case` during load (e.g., `ProviderCodeCurrent` -> `provider_code_current`).
- Provider codes are treated as strings and normalised to 3 digits where applicable (e.g., `6` -> `006`).
- CSV decoding uses a small encoding fallback chain (useful for Windows extracts).

---

## Running the Dashboard

From the project root:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Dash app:
   ```bash
   python src/dashboard/app.py
   ```

  Notes:
  - The dashboard is a long-running web server. After it prints the local URL (e.g. `http://127.0.0.1:8050`), it will keep running until you stop it.
  - If you stop the process (Ctrl+C / VS Code Stop), you may see `KeyboardInterrupt` in the terminal. This is expected and just means the server was interrupted.

  If you're already in the `src/dashboard` folder, you can run it like this instead:
  ```bash
  ..\.\.venv\Scripts\python.exe app.py
  ```

3. Open the local URL shown in the console (default `http://127.0.0.1:8050`).

---

## App Structure & Interactions

### Filters (top of page)

- **Date Range:** month-based range slider built from the patient `period_end` values.
- **Service:** multi-select dropdown displaying `provider_code_current - service_line` pairs.
- **Demand Percentile:** dropdown (50% to 100%, default 60%). This is used in the Demand table to compute a **Clock Stop Target** per service line as a percentile of historical monthly referrals (within the currently filtered dataset).
- **Apply Filters:** loads the filtered patient dataset into the app state (tabs and KPI cards update from this filtered dataset).

### KPI Cards

After filters are applied, the dashboard shows KPI cards (sums across the selected date range):

- Referrals, Waiters, Caseload, Contacts, Discharges. The natural flow of a full patient pathway
- Total Staff (sum across staffing rows for the currently selected provider(s); staffing is not time-indexed in the current dataset)

Additional derived KPI (tab-specific):
- **Sustainable Caseload (DR=1.0)** Uses the Demand Ratio framework to estimate the latest-month caseload that would be sustainable if Demand Ratio were 1.0.
- **Demand Ratio (Latest)** Latest-month Demand Ratio computed using the same clock-stop-target approach as the Demand table (driven by the Demand Percentile filter).
- **Net Caseload Flow (Latest)** Latest-month net flow computed as inflow minus outflow (referrals minus discharges where available).
- **Caseload Throughput Rate (Latest)** Latest-month outflow divided by caseload; higher values indicate the caseload is cycling more quickly.

---

## Tabs

### 📈 Overview

- **Key Metrics Over Time:** multi-metric time series (Referrals, Waiters, Caseload, Contacts, Discharges, Clock Stop Actuals). Legend clicks hide/show series.
- **Forecast toggle (per chart):** optional ETS / exponential smoothing (Holt-Winters) overlay with 12-month horizon and 95% confidence interval.
- **Staff vs Caseload (Latest):** scatter plot comparing latest-month caseload (patient data) against total staff (staffing data), including a trendline.
- **Waiting List Breakdown:** stacked bars for Under vs Over 18 weeks by month.

### 👥 Demand Analysis

- Individual time series cards (each with its own ETS forecast toggle): Referrals, Waiters, Caseload, Contacts, Discharges.
- Waiting List Breakdown (Under vs Over 18 weeks) by month.
- **Patients Data by Service Line:** a scrollable table grouped by `provider_code_current × service_line × period_end` with derived fields including:
  - `clock_stop_target` (based on the selected Demand Percentile)
  - demand ratios (e.g., referral/clock-stop ratios) where source columns exist
  - contacts-per-caseload metrics where source columns exist

### 💼 Capacity Analysis

- Capacity visuals are filtered to the provider(s) currently selected in the patient filters.
- Charts:
  - Total Staff by Staff Group
  - Total Staff by Service Line
  - Staff vs Caseload (latest period)
- **Staffing by Service Line:** pivot table with one row per `provider_code_current × service_line` and one column per `staff_group`.

---

## Forecasting Notes (Current Implementation)

- Forecasts are **optional** and controlled by per-chart toggles.
- Forecasts require sufficient history (the app skips forecasting for very short/flat series).
- Model type: ETS / exponential smoothing (Holt-Winters), 12 months ahead, 95% CI.
