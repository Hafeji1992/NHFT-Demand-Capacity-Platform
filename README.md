# NHFT Demand & Capacity Modelling Platform

A configurable Demand and Capacity Modelling Platform for NHS community and mental health services, featuring automated SQL data pipelines, statistical forecasting, capacity modelling, simulation, and an interactive Dash interface.

## Features

- **Automated SQL Data Pipelines**: Extract, transform, and load healthcare data from SQL databases
- **Time-Series Forecasting**: Multiple forecasting models including:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - ETS (Exponential Smoothing State Space Model)
  - Machine Learning (Gradient Boosting, Random Forest)
  - Ensemble methods combining multiple models
- **Capacity Modelling**: Calculate utilisation, FTE requirements, and wait times
- **Queue Simulation**: Discrete event simulation with Monte Carlo analysis
- **Interactive Dash Web Application**: Visualise forecasts, scenarios, and utilisation metrics

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/Hafeji1992/NHFT-Demand-Capacity-Platform.git
cd NHFT-Demand-Capacity-Platform

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Run the Web Application

```bash
# Using the entry point
dcp-run

# Or directly
python -m demand_capacity_platform.web_app.app
```

Then open your browser at http://localhost:8050

### Using the Python API

```python
from demand_capacity_platform.data_pipelines import DataLoader
from demand_capacity_platform.forecasting import EnsembleForecaster
from demand_capacity_platform.simulation import CapacityModel, QueueSimulation

# Generate sample data
loader = DataLoader()
referrals = loader.generate_sample_referrals(n=1000, seed=42)

# Prepare time series
ts = loader.prepare_timeseries(referrals, "referral_date")

# Run forecast
forecaster = EnsembleForecaster()
result = forecaster.fit_predict(ts, steps=30)
print(f"Forecast model: {result.model_name}")
print(f"MAE: {result.fit_metrics['MAE']:.2f}")

# Capacity analysis
capacity = loader.generate_sample_capacity()
model = CapacityModel()
metrics = model.analyse_capacity(referrals, capacity, "Adult Mental Health")
print(f"Utilisation: {metrics.utilisation_rate:.1%}")

# Queue simulation
sim = QueueSimulation(daily_capacity=20, seed=42)
sim_result = sim.run(days=365, daily_arrivals=18)
print(f"Average wait time: {sim_result.average_wait_time:.1f} days")
```

## Project Structure

```
NHFT-Demand-Capacity-Platform/
├── src/
│   └── demand_capacity_platform/
│       ├── __init__.py
│       ├── data_pipelines/          # Data ingestion and ETL
│       │   ├── __init__.py
│       │   ├── sql_pipeline.py      # SQL database connectivity
│       │   └── data_loader.py       # Data loading utilities
│       ├── forecasting/             # Time-series forecasting
│       │   ├── __init__.py
│       │   ├── base.py              # Base forecaster class
│       │   ├── arima.py             # ARIMA implementation
│       │   ├── ets.py               # ETS implementation
│       │   ├── ml_forecaster.py     # ML-based forecasting
│       │   └── ensemble.py          # Ensemble methods
│       ├── simulation/              # Capacity and queue simulation
│       │   ├── __init__.py
│       │   ├── capacity_model.py    # Capacity analysis
│       │   └── queue_simulation.py  # Queue simulation
│       └── web_app/                 # Dash web application
│           ├── __init__.py
│           ├── app.py               # Main application
│           ├── layouts.py           # UI layouts
│           └── callbacks.py         # Interactive callbacks
├── tests/                           # Test suite
│   ├── conftest.py                  # Test fixtures
│   ├── data_pipelines/
│   ├── forecasting/
│   ├── simulation/
│   └── web_app/
├── pyproject.toml                   # Project configuration
├── requirements.txt                 # Dependencies
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=demand_capacity_platform --cov-report=html

# Run specific test module
pytest tests/forecasting/
```

## Configuration

The platform supports configuration through YAML files. Create a `config.yaml`:

```yaml
database:
  connection_string: "postgresql://user:pass@localhost/nhft"

forecasting:
  default_model: ensemble
  horizon_days: 30
  confidence_level: 0.95

capacity:
  working_hours_per_day: 7.5
  working_days_per_week: 5
  productivity_factor: 0.8

simulation:
  n_simulations: 100
  seed: 42
```

## API Reference

### Data Pipelines

#### `SQLDataPipeline`
```python
pipeline = SQLDataPipeline("sqlite:///data.db")
df = pipeline.execute_query("SELECT * FROM referrals")
```

#### `DataLoader`
```python
loader = DataLoader()
referrals = loader.generate_sample_referrals(n=1000)
ts = loader.prepare_timeseries(referrals, "referral_date")
```

### Forecasting

#### Forecasters
- `ARIMAForecaster`: ARIMA models with automatic order selection
- `ETSForecaster`: Exponential smoothing with trend/seasonal components
- `MLForecaster`: Machine learning with lag features
- `EnsembleForecaster`: Weighted combination of multiple models

```python
forecaster = EnsembleForecaster()
result = forecaster.fit_predict(time_series, steps=30)
print(result.forecast)
print(result.fit_metrics)
```

### Simulation

#### `CapacityModel`
```python
model = CapacityModel()
metrics = model.analyse_capacity(demand_df, capacity_df, "Service Name")
print(f"Utilisation: {metrics.utilisation_rate:.1%}")
```

#### `QueueSimulation`
```python
sim = QueueSimulation(daily_capacity=20)
result = sim.run(days=365, daily_arrivals=18)
print(f"Average wait: {result.average_wait_time:.1f} days")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed for NHFT as part of the ZDAT3001 Work-Based Project
- Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/)
- Statistical models powered by [statsmodels](https://www.statsmodels.org/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
