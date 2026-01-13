# Forecasting (SARIMA/SARIMAX)

This folder contains the time-series forecasting utilities for the NHFT Demand–Capacity Platform.

The goal is to generate short-to-medium horizon planning forecasts (typically **6-month** and **12-month**) from historical monthly aggregates and to provide **uncertainty bounds** (95% confidence intervals) that can be displayed on the dashboard.

## What we are doing

At a high level, the pipeline is:

1. **Aggregate to monthly time series** (one value per month)
   - Converts row-level data into a regularly spaced monthly series.
   - Fills missing months (if any) so the model sees a consistent calendar.

2. **Fit a Seasonal ARIMA model (SARIMA via SARIMAX)**
   - SARIMAX is used as the implementation because it supports SARIMA-style models and provides standard forecast intervals.
   - Both are modelled:
     - **Short-term autocorrelation** (recent months influence next month)
     - **Seasonality** (monthly data often has an annual pattern; period = 12)

3. **Forecast forward and produce 95% confidence intervals**
   - Outputs are returned in a simple DataFrame format that is easy to plot in Plotly/Dash.

## Why SARIMA

SARIMA/SARIMAX was chosen for this use case because it matches the constraints of the project:

- **Works well with monthly data and annual seasonality**: healthcare operational metrics commonly show seasonality (winter pressures, holiday effects, etc.). SARIMA explicitly represents this with seasonal parameters and period $s=12$.
- **Interpretable and defensible**: ARIMA-family models have a long track record in operational forecasting, and stakeholders can understand the idea of trend, seasonality and noise.
- **Produces statistically grounded prediction intervals**: SARIMAX provides model-based confidence intervals that are straightforward to communicate.
- **Suitable for limited history**: The data pulled for this project has 12–24 monthly points. Many ML models (e.g., XGBoost/LSTM) typically require more data and feature engineering to be stable.

Alternatives and trade-offs:

- **Holt-Winters / ETS**: strong baseline for seasonal data, but in practice but a model that can handle different autocorrelation structures and provides a more general framework when series behave differently is required.
- **Machine learning (e.g., random forests, gradient boosting)**: usually needs engineered calendar/lag features and enough history to avoid overfitting; interval estimation is also more involved.

## How parameters are chosen (method and justification)

SARIMA is typically written as:

$$\text{ARIMA}(p, d, q) \times (P, D, Q)_s$$

Where:

- $(p, d, q)$ control short-term autoregression, differencing, and moving-average behaviour.
- $(P, D, Q)$ control seasonal effects.
- $s$ is the seasonal period (for monthly data, $s=12$).

### Seasonal period ($s$)

-  **$s=12$** is used because the input series is monthly and this allows an annual seasonal pattern.

### Differencing ($d$ and $D$)

- Differencing helps make the series closer to stationary.
- Seasonal differencing ($D$) is powerful but **data-hungry**: it effectively reduces usable history and can destabilize estimation with short series.

Practical rule used in the dashboard integration:

- If there is **at least ~24 months** of history, a seasonal component with differencing (depending on the chosen spec) can be created.
- If there is **12–23 months**, we still allow seasonality ($s=12$) but seasonal differencing is disabled ($D=0$) to keep the model estimable.

This is a pragmatic compromise: it preserves the ability to capture repeating annual structure without overfitting or losing too many degrees of freedom.

### Orders ($p,q,P,Q$) and model selection

Two modes are supported:

1. **Fixed, fast specification (recommended for dashboards)**
   - Uses a small, stable SARIMA specification that is quick to fit and tends to behave well for many operational series.
   - This is chosen for responsiveness and to avoid large per-refresh compute.

2. **Optional lightweight auto-selection (AIC search) for offline tuning**
   - `small_grid_search_aic(...)` tries a small grid of candidate orders and selects the model with the lowest AIC.
   - AIC (Akaike Information Criterion) provides a balance between fit quality and model complexity:
     - lower AIC = better trade-off between goodness-of-fit and overfitting risk.

The search grid is intentionally kept small to reduce runtime and to avoid “over-optimising” on very short histories.

### Frequency handling (monthly)

- Series are coerced to a regular monthly frequency (month-end), using pandas’ **`ME`** frequency alias.
- This ensures the model and the plotting layer agree about what a “month” timestamp represents.

### Confidence intervals (95%)

- Forecast intervals are generated from the SARIMAX results object.
- The default confidence level is **0.95**, which is standard for planning.
- Intervals reflect model uncertainty under SARIMA assumptions (not “worst case” bounds).

## Outputs

The main helper returns a DataFrame with history and forecast in one table, typically including:

- `y` (observed)
- `yhat` (forecast mean)
- `yhat_lower`, `yhat_upper` (confidence interval)

This is designed to plug directly into Plotly traces (solid actuals, dashed forecast, and a shaded interval band).

## Key files

- `preprocessing.py`
  - `build_monthly_series(df, value_col, date_col="period_end")`
  - `coerce_monthly_series(y, freq="ME")`
- `sarima.py`
  - `SarimaForecaster` (fit + forecast with intervals)
  - `SarimaSpec` (stores `(p,d,q)` and `(P,D,Q,s)`)
  - `small_grid_search_aic(...)` (optional lightweight order search)
- `forecast.py`
  - `ForecastConfig` (forecast horizon, confidence level, frequency)
  - `make_forecast_frame(y, config=...)` (history + future in one DataFrame)

## Quick example

```python
from forecasting.preprocessing import build_monthly_series
from forecasting.forecast import make_forecast_frame, ForecastConfig

# df must have: period_end (datetime-like), referrals (numeric)
y = build_monthly_series(df, "referrals", date_col="period_end")

# 6-month forecast with 95% CI
frame_6 = make_forecast_frame(
    y,
    config=ForecastConfig(months_ahead=6, conf_level=0.95, freq="ME"),
)

# 12-month forecast with 95% CI
frame_12 = make_forecast_frame(
    y,
    config=ForecastConfig(months_ahead=12, conf_level=0.95, freq="ME"),
)
```

## Console report (business case walkthrough)

To generate a console-based report that includes:

- ADF test output
- AIC grid search (model selection) + justification
- Model fit summary
- Multi-step forecast with 95% confidence intervals

Run:

```bash
python src/forecasting/sarima_console_report.py --metric referrals
```

Optional examples:

```bash
python src/forecasting/sarima_console_report.py --metric waiters --months-ahead 6
python src/forecasting/sarima_console_report.py --csv data/staffing_data.csv --metric staff --date-col period_end
python src/forecasting/sarima_console_report.py --metric referrals --show-warnings
```

## Important caveats

- Forecasts are only as good as the historical signal; sudden operational changes (policy, service redesign, data definition changes) may not be captured.
- Confidence intervals are model-based and assume the SARIMA structure is an adequate approximation.
- If the series is extremely short or mostly flat/zero, SARIMA may not converge or may produce wide intervals; the dashboard should handle this gracefully.
