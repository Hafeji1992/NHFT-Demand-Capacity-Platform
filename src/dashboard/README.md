# NHFT Demand & Capacity Dashboard  
## Dash Application Documentation (Brief)

---

## Overview

This folder contains the interactive **Dash** application used to explore NHFT demand and capacity data.

- **Primary audience:** Operational analysts and project stakeholders.
- **Purpose:** Provide a clear, filterable view of demand activity (patients dataset) and capacity mix (staffing dataset) to support planning, modelling, and future forecasting work.

---

## Folder Contents

- `app.py`  
  Main Dash application (layout + callbacks + charts/tables).

- `data_handler.py`  
  Loads and preprocesses dashboard datasets from `data/` and provides helper methods for filtering and summarising.

---

## Data Requirements

The dashboard expects the ingestion pipeline to have produced the following files:

- `data/patient_data.csv`  
  Monthly grain at **ProviderCodeCurrent × Service_Line × PeriodEnd**.

- `data/staffing_data.csv`  
  Workforce snapshot at **ProviderCodeCurrent × Service_Line × Staff_Group**.

Notes:
- These CSVs are typically **excluded from version control**.
- The top-level service filter uses **`providercodecurrent`** for consistent filtering across both datasets.

---

## Running the Dashboard

From the project root (recommended):

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Dash app:
   ```bash
   python src/dashboard/app.py
   ```

3. Open the local URL shown in the console (default `http://127.0.0.1:8050`).

---

## Current Dashboard Views

- **Overview**  
  High-level trends and key metrics over time.

- **Demand Analysis**  
  Demand-focused summaries and tables (to be elaborated).

- **Capacity Analysis**  
  Capacity-focused summaries and staffing mix tables (to be elaborated).

---

## Planned Enhancements (Next Iterations)

This README is intentionally brief for now; the dashboard will be expanded to include:

- Detailed definitions and interpretation guidance for Demand & Capacity tab outputs
- Time-series analysis utilities (seasonality, change points, trend decomposition)
- Forecasting (classical + ML-based approaches)
- Demand-to-capacity modelling and scenario testing
- Model evaluation, monitoring, and reproducibility notes
