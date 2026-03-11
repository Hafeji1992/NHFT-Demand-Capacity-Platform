# NHFT Demand Capacity Platform

A configurable Demand and Capacity analytics platform for NHFT, featuring automated data ingestion (to standardised CSV extracts), statistical forecasting utilities, and an interactive Dash interface. Developed for the ZDAT3001 Work-Based Project to support data-driven operational planning.


## 1. Overview

This project develops a Demand and Capacity Modelling Platform for Northamptonshire Healthcare NHS Foundation Trust (NHFT). It forms part of the ZDAT3001 Year 3 Work-Based Project and aims to address a key organisational need: improving forecasting accuracy, capacity planning, and operational decision-making across 500+ community, mental health, and specialist services.

The platform integrates:

- Automated data ingestion and validation
- Statistical forecasting (dashboard default: ETS / Holt-Winters exponential smoothing)
- Interactive Dash application for demand & capacity exploration

Planned / in-progress workstreams (see Project Status) include capacity modelling templates and simulation.

## 2. Problem Statement

NHFT service leads currently rely on manual spreadsheets, retrospective reporting, or ad-hoc forecasting methods. This limits proactive resource planning and obscures developing bottlenecks.

The project asks:

How can NHFT accurately forecast service demand and optimise capacity allocation using existing data sources, ensuring consistency, fairness, and operational efficiency across multiple service types?

This question is informed by the Scientific Approach and background analysis detailed in the Project Approach document.

## 3. Project Objectives

As specified in the Project Agreement (pp. 2–3) , the project will deliver a platform that:

- Produces more reliable, interpretable, and standardised demand forecasts
- Models service capacity using configurable templates and staffing rules
- Simulates operational outcomes under varying assumptions
- Enables service managers to explore “what-if” scenarios through an interactive dashboard
- Supports BI and Performance Improvement teams with reusable modelling components

## 4. Key Features
### 4.1 Automated Data Engineering Pipeline
- SQL extraction + reproducible transformation steps
- Column name normalisation to `snake_case` for consistent downstream use
- Data validation / quality checks
- Output datasets written to `data/patient_data.csv` and `data/staffing_data.csv` for the dashboard and analysis modules

### 4.2 Forecasting Engine
Current dashboard implementation uses:

- Exponential Smoothing (ETS / Holt-Winters), with optional 12-month forecasts and 95% confidence intervals
- Archived SARIMA utilities are retained under `src/forecasting/archived_sarima/` for reference/offline use

Planned: broader model benchmarking and selection (e.g., ML approaches) once evaluation requirements are finalised.

### 4.3 Capacity Modelling (reference-only in current MVP)

Due to workforce extraction constraints and service mapping misalignment (service lines vs cost centres / org structures), the project does not currently have the inputs needed for a time-aligned capacity model that cleanly merges with demand.

Current capability focuses on capacity as **context** rather than a merged model:

- Staffing is treated as a **current snapshot** (no historical staffing time series).
- Capacity outputs are presented as a **reference view** in the dashboard (staff mix, staff by service line, staff vs latest caseload) to support triangulation discussions.
- Demand-to-capacity evaluation is handled via **demand-based benchmarking** (percentile-derived targets and derived demand ratios) rather than a merged staffing model.

Future enhancement (dependent on improved workforce history + mapping):

- Service templates and configurable capacity parameters (availability, rosters, DNAs, constraints)
- Time-indexed capacity baselines to enable proper demand–capacity joins and scenarios

### 4.4 Dash Application

Interactive web dashboard built with Plotly Dash and styled to the NHS Design System colour palette.

- **Four tabs:** About This Dashboard (methodology guide), Overview, Demand Analysis, and Capacity Analysis
- **Filters:** month-based date-range slider, multi-select service dropdown, and a Clinician Patient Facing Time percentile selector (50–100%, default 60%) that drives demand benchmarking targets
- **Headline KPI cards:** Referrals, Waiters, Caseload, Contacts, Discharges, Total Staff, plus derived metrics (Demand Ratio, Sustainable Caseload, Net Caseload Flow, Caseload Throughput, Clearance Time)
- **Per-chart ETS forecast toggles:** optional 12-month Holt-Winters overlay with 95% confidence interval on each time-series chart
- **Demand table:** service-line summary with Clock Stop Target, granular demand ratios, and contacts-per-caseload fields
- **Capacity reference view:** staff-group and service-line bar charts, staff-vs-caseload scatter with trendline, and staffing pivot table

For full UI behaviour, data expectations, and tab-by-tab detail see `src/dashboard/README.md`.

### 4.5 Data Limitations, Triangulation, and Capacity Approach

During delivery two practical constraints were identified that materially shape the modelling approach:

- **No historical workforce time series available:** due to extraction limitations, historical staffing/workforce data was not obtainable at the required grain. The staffing dataset available to this project is therefore a **current snapshot only** (no month-by-month staffing history).
- **Staff vs patient “service line” misalignment:** patient activity and workforce data do not naturally align by service line because service definitions are represented differently across systems (e.g., **service lines vs cost centres / organisational hierarchies**). This project is a strong demonstration of why naive joins can be misleading.

This mismatch has been documented and shared with senior stakeholders to support future improvements in data triangulation and service mapping.

As a result:

- A “naturally merging” staff capacity model (that joins cleanly to demand at the same service-line grain over time) is **not feasible** with the currently available workforce extract.
- The **Capacity** area of the dashboard is implemented as a **reference view** (staff mix, staff by service line, and staff vs latest caseload), providing context rather than a fully time-aligned capacity model.
- The project therefore focuses on **demand-based benchmarking** using **demand percentiles**:
    - the dashboard computes a service-line **Clock Stop Target** as a chosen percentile of historical monthly referrals within the selected filter window
    - derived demand ratios then allow demand to be compared against that target, while the staffing snapshot provides contextual capacity signals (e.g., staff-per-100-caseload at latest period)

## 5. Repository Structure
```plaintext
NHFT-Demand-Capacity-Platform/
├── docs/
│   ├── Ethical and Legal Assessment.pdf
│   ├── Project Agreement.pdf
│   ├── Project Approach.pdf
│   ├── Project Plan.pdf
│   ├── Project Methodology.pdf
│   └── DPIA Assessment.pdf
│
├── data/
│   ├── patient_data.csv
│   ├── staffing_data.csv
│   └── last_refreshed.json
│
├── src/
│   ├── analysis/
│   │   ├── patient_summary_statistics.py
│   │   └── staffing_summary_statistics.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── data_handler.py
│   │   └── README.md
│   │
│   ├── data_engineering/
│   │   ├── connect.py
│   │   ├── master_data_ingestion.py
│   │   ├── patient_data_ingestion.py
│   │   ├── staffing_data_ingestion.py
│   │   └── README.md
│   │
│   ├── forecasting/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── forecast.py
│   │   ├── ets.py
│   │   ├── ets_report.py
│   │   └── archived_sarima/
│   │       ├── __init__.py
│   │       ├── sarima.py
│   │       └── sarima_report.py
│
│   ├── sql/
│   │   ├── capacity_data_extraction_query.sql
│   │   └── demand_data_extraction_query.sql
│
│   └── tests/
│       ├── test_patient_data_ingestion.py
│       ├── test_sql_connection.py
│       └── test_staffing_data_ingestion.py
│
├── .gitignore
├── config.ini
├── README.md
└── requirements.txt
```

The structure follows the Design and Structure outlined in the Project Approach (pp. 4–5) and the Project Plan (Section 6) .

## 6. Methodology

The project uses a Kanban software development methodology, aligned with NHFT BI team workflows and academic justification (Project Methodology, p. 1–2) .

Why Kanban?
- Supports continuous refinement across modelling, data engineering, and dashboard development
- Allows dynamic prioritisation based on stakeholder feedback
- Integrates well with Microsoft Planner and weekly check-ins

## 7. Data Governance, Ethics, and Legal Compliance

The project complies with:
- UK GDPR
- Data Protection Act 2018
- Caldicott Principles
- NHFT Information Governance Standards

Key principles include:
- Pseudonymisation and access restriction
- No identifiable data stored locally
- Secure SQL access within NHFT network
- Audit trails and documentation via Git & OneNote

These are detailed in the Ethical and Legal Assessment (pp. 1–5) .

A Data Protection Impact Assessment (DPIA) will be completed before processing begins.

## 8. Installation
Clone the repository:

```bash
git clone https://github.com/Hafeji1992/NHFT-Demand-Capacity-Platform
cd NHFT-Demand-Capacity-Platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note: This project must run inside the NHFT secure environment due to data governance constraints.**

## 9. Running the Dashboard

1) (Optional) Refresh extracts (writes to `data/`):

```bash
python src/data_engineering/master_data_ingestion.py
```

2) Launch the Dash application:

```bash
python src/dashboard/app.py
```

The app currently includes:
- Filtered demand metrics + KPIs
- Demand and capacity summary tables
- Optional ETS forecast overlays on selected charts
- Percentile-based demand targets and derived demand ratios for service-line benchmarking

## 10. Testing

Unit tests are currently included for:
- Data pipeline integrity (ingestion)
- SQL connectivity checks

Run tests using:

```bash
pytest src/tests/
```

## 11. Stakeholders

As defined in the Project Plan (Section 2, p. 1) :
- Project Lead: Yahya Hafeji
- BI Manager: Accountable for data governance
- Performance Improvement Manager: Project oversight
- Service Managers: Pilot feedback
- University Supervisor: Academic guidance

## 12. Project Status

| Workstream / Phase                      | Status        | Target Completion | Completion Date | Hours Spent | Notes |
|----------------------------------------|--------------|-------------------|-----------------|-------------|--------|
| Project Documentation & Governance     | ✅ Completed | Early Dec 2025    | 08/12/2025      | 42          | Project Agreement, Plan, Methodology, Ethics & Legal all completed |
| Data Access & Infrastructure Setup     | ✅ Completed | Mid Dec 2025      | 15/12/2025      | 15          | SQL access, governance approvals, secure Python environment |
| Data Engineering & Automation Pipeline | ✅ Completed | Early Jan 2026    | 22/12/2025      | 30          | Direct-query pipeline, validation scripts, staffing & demand ingestion |
| Forecasting Module (MVP)               | ✅ Completed  | Early Feb 2026    | 09/02/2026     | 97.5         | ETS forecasting utilities (dashboard default) + SARIMA retained under archived utilities |
| Capacity & Simulation Module           | ✅ Completed (Re-scoped) | Mid Feb 2026      | 09/02/2026      | 30           | Pivoted (WC 09/02/2026): simulation/queueing de-scoped due to no workforce time series + service-line mapping misalignment; delivered reference-based capacity view (snapshot staffing mix/allocations) in dashboard |
| Dash Application Development           | ✅ Completed   | End Feb 2026      | 19/02/2026    | 15           | Interactive dashboard (Overview/Demand/Capacity tabs, filters, percentile-based targets, derived tables, ETS forecast toggles; staffing snapshot used as reference) |
| Testing, Validation & Refinement       | ✅ Completed  | End Feb 2026      | 09/03/2026     | 22.5           | Forecast evaluation, stakeholder testing |
| Documentation & Final Presentation Prep| ⏳ Upcoming  | June 2026         | —               | 7.5           | Written portfolio, slides, video pitch |

Milestones taken from the Project Plan timeline (Section 6, pp. 3–4) .

**Target Hours:**  
The total time available for this project is approximately **six working weeks**, equivalent to **~225 hours** (based on a standard 37.5-hour working week). The hours spent column will be updated as the project progresses.

## 13. License

This project is developed for internal NHFT use and academic assessment.
All data remains property of NHFT; code ownership follows apprenticeship agreement terms.
Open-source libraries are used in accordance with their licences.

(As outlined in the Ethical & Legal Assessment, pp. 2–3)

## 14. Acknowledgements

Thanks to:
- NHFT Business Intelligence Service
- University of Nottingham (Data Science Apprenticeship Program)

## 15. Project Flow Diagram

Note: this diagram represents the target architecture across workstreams.

```mermaid
flowchart TD

    subgraph Data_Sources["Data Sources"]
        Referrals[(Referrals)]
        Appointments[(Appointments)]
        Workforce[(Workforce)]
        Metadata[(Service metadata)]
        Calendar[(Calendar & holidays)]
    end

    subgraph Data_Engineering_Pipeline["Data Engineering Pipeline"]
        SQLQuery[SQL extraction]
        Validation[Validation & profiling]
        Transform[Feature engineering]
        Pseudonymise[Pseudonymisation]
        Curated[(Curated / master dataset)]
    end

    subgraph Demand_Modelling["Demand Modelling"]
        Refs[Referrals]
        Waiters[Waiters]
        Caseload[Caseload]
        Contacts[Contacts]
        Discharges[Discharges]
        DemandTS[(Demand time series)]
    end

    subgraph Capacity_Modelling["Capacity Modelling"]
        Templates[Service templates]
        Staffing[Staffing & rosters]
        Constraints[Resource constraints]
        CapacitySnap[(Capacity snapshot / view)]
    end

    subgraph Forecasting_Engine["Forecasting Engine"]
        ModelSelect["Model selection (AIC)"]
        ETS["ETS (dashboard default)"]
        SARIMA[Archived SARIMA]
        ForecastOut[(Forecast outputs)]
    end

    subgraph Dashboard_App["Dashboard App"]
        UI[Interactive UI]
        Viz[Demand & capacity visualisations]
    end

    Data_Sources --> SQLQuery
    SQLQuery --> Validation --> Transform --> Pseudonymise --> Curated

    Curated --> Demand_Modelling
    Curated --> Capacity_Modelling

    Demand_Modelling --> DemandTS --> ModelSelect
    ModelSelect --> ETS --> ForecastOut
    ModelSelect -. "optional" .-> SARIMA
    SARIMA -.-> ForecastOut

    ForecastOut --> Viz
    Capacity_Modelling --> CapacitySnap --> Viz
    Viz --> UI
```
