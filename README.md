# NHFT Demand Capacity Platform

A configurable Demand and Capacity Modelling Platform for NHFT, featuring automated data pipelines, statistical forecasting, capacity modelling, simulation, and an interactive Dash interface. Developed for the ZDAT3001 Work-Based Project to support data-driven operational planning.


## 1. Overview

This project develops a Demand and Capacity Modelling Platform for Northamptonshire Healthcare NHS Foundation Trust (NHFT). It forms part of the ZDAT3001 Year 3 Work-Based Project and aims to address a key organisational need: improving forecasting accuracy, capacity planning, and operational decision-making across 500+ community, mental health, and specialist services.

The platform integrates:

- Automated data ingestion and validation
- Statistical and machine learning forecasting (ARIMA, ETS, Gradient Boosting)
- Capacity modelling using service templates and staffing parameters
- Queueing theory and discrete-event simulation using SimPy
- An interactive Dash application for scenario exploration

## 2. Problem Statement

NHFT service leads currently rely on manual spreadsheets, retrospective reporting, or ad-hoc forecasting methods. This limits proactive resource planning and obscures developing bottlenecks.

The project asks:

How can NHFT accurately forecast service demand and optimise capacity allocation using existing data sources, ensuring consistency, fairness, and operational efficiency across multiple service types?

This question is informed by the Scientific Approach and background analysis detailed in the Project Approach document .

## 3. Project Objectives

As specified in the Project Agreement (pp. 2–3) , the project will deliver a platform that:

- Produces more reliable, interpretable, and standardised demand forecasts
- Models service capacity using configurable templates and staffing rules
- Simulates operational outcomes under varying assumptions
- Enables service managers to explore “what-if” scenarios through an interactive dashboard
- Supports BI and Performance Improvement teams with reusable modelling components

## 4. Key Features
### 4.1 Automated Data Engineering Pipeline
- Direct SQL Server querying (no local file storage)
- Data validation, profiling, and pseudonymisation
- Reproducible transformation steps following governance standards

### 4.2 Forecasting Engine
Implements multiple models including:

- ARIMA
- Exponential Smoothing (ETS)
- Gradient Boosting
- Model selection using MAE/RMSE and fairness metrics

### 4.3 Capacity Modelling

- Service templates (community, outpatient, mental health)
- Configurable staff availability, rosters, DNA rates, and constraints

### 4.4 Queueing & Simulation Module

- M/M/s, M/G/s queueing systems
- Discrete-event simulation using SimPy
- Scenario analysis for staffing, demand variation, and breach risk detection

### 4.5 Dash Application

- Interactive visual dashboards
- Tabs for forecasts, capacity utilisation, and simulation outputs
- Scenario controls (sliders, toggles, drop-downs)
- Accessible, interpretable design for operational managers

## 5. Repository Structure

```plaintext
nhft-demand-capacity-modelling-platform/
├── docs/
│   ├── Ethical-and-Legal-Assessment.pdf
│   ├── Project-Agreement.pdf
│   ├── Project-Approach.pdf
│   ├── Project-Plan.pdf
│   ├── Project-Methodology.pdf
│   └── DPIA/
│
├── src/
│   ├── data_engineering/
│   │   ├── sql_queries/
│   │   ├── validation/
│   │   └── pipeline.py
│   │
│   ├── forecasting/
│   │   ├── arima.py
│   │   ├── ets.py
│   │   ├── xgboost_forecast.py
│   │   └── model_selection.py
│   │
│   ├── simulation/
│   │   ├── queue_models.py
│   │   ├── simpy_engine.py
│   │   └── scenarios.py
│   │
│   └── dashboard/
│       ├── app.py
│       ├── layout/
│       └── callbacks/
│
├── notebooks/
│   ├── exploratory-data-analysis.ipynb
│   ├── forecasting-validation.ipynb
│   └── simulation-prototypes.ipynb
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_forecasting.py
│   └── test_simulation.py
│
├── requirements.txt
├── .gitignore
└── README.md
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
- git clone <https://github.com/Hafeji1992/NHFT-Demand-Capacity-Platform>
- cd nhft-demand-capacity-modelling-platform

Install dependencies:
pip install -r requirements.txt

**Note: This project must run inside the NHFT secure environment due to data governance constraints.**

## 9. Running the Dashboard

To launch the Dash application: python src/dashboard/app.py

The app includes: 
- Forecast visualisation
- Capacity scenarios
- Simulation outputs
- Service profiling tools

## 10. Testing

Unit tests are included for:
- Data pipeline integrity
- Forecasting accuracy
- Simulation stability

Run tests using:

pytest tests/

## 11. Stakeholders

As defined in the Project Plan (Section 2, p. 1) :
- Project Lead: Yahya Hafeji
- BI Manager: Accountable for data governance
- Performance Improvement Manager: Project oversight
- Service Managers: Pilot feedback
- University Supervisor: Academic guidance

## 12. Project Status

| Workstream / Phase                         | Status        | Target Completion | Notes |
|--------------------------------------------|--------------|-------------------|--------|
| Project Documentation & Governance         | ✅ Completed | Early Dec 2025    | Project Agreement, Plan, Methodology, Ethics & Legal all completed |
| Data Access & Infrastructure Setup         | ⏳ Pending   | Mid Dec 2025      | SQL access, governance approvals, secure Python environment |
| Data Engineering & Automation Pipeline     | ⏳ Upcoming  | Early Jan 2026    | Direct-query pipeline and validation scripts |
| Forecasting Module (MVP)                   | ⏳ Upcoming  | Early Feb 2026    | ARIMA/ETS/XGB models and selection logic |
| Capacity & Simulation Module               | ⏳ Upcoming  | Mid Feb 2026      | Queueing and simulation with SimPy |
| Dash Application Development               | ⏳ Upcoming  | End Feb 2026      | Interactive dashboard with scenarios |
| Testing, Validation & Refinement           | ⏳ Upcoming  | End Feb 2026      | Forecast evaluation, stakeholder testing |
| Documentation & Final Presentation Prep    | ⏳ Upcoming  | June 2026         | Written portfolio, slides, video pitch |


Milestones taken from the Project Plan timeline (Section 6, pp. 3–4) .

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

```mermaid
flowchart TD

    subgraph Data_Sources
        Referrals[(Referrals Data)]
        Appointments[(Appointments Data)]
        Workforce[(Workforce Data)]
        Metadata[(Service Metadata)]
        Calendar[(Calendar & Holidays)]
    end

    subgraph Data_Engineering_Pipeline
        SQLQuery[SQL Extraction]
        Validation[Data Validation & Profiling]
        Transform[Feature Engineering]
        Pseudonymise[Pseudonymisation]
    end

    subgraph Forecasting_Engine
        ARIMA[ARIMA Model]
        ETS[ETS Model]
        XGB[Gradient Boosting]
        ModelSelect[Model Selection]
    end

    subgraph Demand_Modelling
        Refs[Referrals]
        Waiters[Waiters]
        Caseload[Caseload]
        Contacts[Contacts]
        Discharges[Discharges]
    end

    subgraph Capacity_Modelling
        Templates[Service Templates]
        Staffing[Staffing & Rosters]
        Constraints[Resource Constraints]
    end

    subgraph Simulation
        Queues[Queueing Models]
        SimPyEngine[SimPy Engine]
        Scenarios[Scenario Analysis]
    end

    subgraph Dashboard_App
        UI[Interactive UI]
        ForecastViz[Forecast Visualisations]
        SimViz[Simulation Outputs]
        ScenarioControls[Scenario Controls]
    end

    Data_Sources --> Data_Engineering_Pipeline
    Data_Engineering_Pipeline --> Demand_Modelling
    Data_Engineering_Pipeline --> Capacity_Modelling
    Demand_Modelling --> Forecasting_Engine 
    Capacity_Modelling --> Forecasting_Engine 
    Forecasting_Engine --> Simulation
    Simulation --> Dashboard_App
    ```
