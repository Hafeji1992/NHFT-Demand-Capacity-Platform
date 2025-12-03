"""
Dash Application Callbacks Module.

Defines the callback functions for the web application.
"""

import logging
from typing import Optional

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from demand_capacity_platform.data_pipelines import DataLoader
from demand_capacity_platform.forecasting import (
    ARIMAForecaster,
    EnsembleForecaster,
    ETSForecaster,
    MLForecaster,
)
from demand_capacity_platform.simulation import CapacityModel, QueueSimulation

logger = logging.getLogger(__name__)


def register_callbacks(app):
    """Register all callbacks for the Dash application."""

    @app.callback(
        [
            Output("demand-data-store", "data"),
            Output("capacity-data-store", "data"),
        ],
        Input("service-dropdown", "value"),
    )
    def load_data(service: str) -> tuple[dict, dict]:
        """Load demand and capacity data for the selected service."""
        loader = DataLoader()

        # Generate sample data
        demand_df = loader.generate_sample_referrals(n=2000, seed=42)
        capacity_df = loader.generate_sample_capacity()

        return demand_df.to_dict("records"), capacity_df.to_dict("records")

    @app.callback(
        [
            Output("kpi-demand", "children"),
            Output("kpi-capacity", "children"),
            Output("kpi-utilisation", "children"),
            Output("kpi-wait-time", "children"),
        ],
        [
            Input("demand-data-store", "data"),
            Input("capacity-data-store", "data"),
            Input("service-dropdown", "value"),
            Input("demand-change", "value"),
            Input("capacity-change", "value"),
        ],
    )
    def update_kpis(
        demand_data: list,
        capacity_data: list,
        service: str,
        demand_change: float,
        capacity_change: float,
    ) -> tuple[str, str, str, str]:
        """Update KPI cards based on selected service and scenarios."""
        if not demand_data or not capacity_data:
            raise PreventUpdate

        demand_df = pd.DataFrame(demand_data)
        capacity_df = pd.DataFrame(capacity_data)

        # Filter for selected service
        service_demand = demand_df[demand_df["service"] == service]
        service_capacity = capacity_df[capacity_df["service"] == service].iloc[0]

        # Calculate weekly demand
        service_demand["referral_date"] = pd.to_datetime(service_demand["referral_date"])
        weekly_demand = (
            service_demand.groupby(pd.Grouper(key="referral_date", freq="W"))
            .size()
            .mean()
        )

        # Apply demand change
        weekly_demand = weekly_demand * (1 + demand_change / 100)

        # Calculate capacity
        model = CapacityModel()
        weekly_capacity = model.calculate_weekly_capacity(
            service_capacity["staff_fte"],
            service_capacity["avg_appointment_mins"],
        )

        # Apply capacity change
        weekly_capacity = weekly_capacity * (1 + capacity_change / 100)

        # Calculate metrics
        utilisation = weekly_demand / weekly_capacity if weekly_capacity > 0 else 1.0
        wait_time = model.estimate_wait_time(weekly_demand * 2, weekly_capacity, weekly_demand)

        return (
            f"{weekly_demand:.0f}",
            f"{weekly_capacity:.0f}",
            f"{utilisation:.0%}",
            f"{wait_time:.1f}",
        )

    @app.callback(
        Output("chart-content", "children"),
        [
            Input("chart-tabs", "active_tab"),
            Input("run-analysis-btn", "n_clicks"),
        ],
        [
            State("demand-data-store", "data"),
            State("capacity-data-store", "data"),
            State("service-dropdown", "value"),
            State("forecast-horizon", "value"),
            State("forecast-model", "value"),
            State("demand-change", "value"),
            State("capacity-change", "value"),
        ],
    )
    def update_chart(
        active_tab: str,
        n_clicks: Optional[int],
        demand_data: list,
        capacity_data: list,
        service: str,
        forecast_horizon: int,
        forecast_model: str,
        demand_change: float,
        capacity_change: float,
    ) -> dcc.Graph:
        """Update the main chart based on selected tab and parameters."""
        if not demand_data:
            return html.P("No data available. Please select a service.")

        demand_df = pd.DataFrame(demand_data)
        capacity_df = pd.DataFrame(capacity_data)

        if active_tab == "tab-forecast":
            return create_forecast_chart(
                demand_df, service, forecast_horizon, forecast_model
            )
        elif active_tab == "tab-capacity":
            return create_capacity_chart(demand_df, capacity_df, service)
        elif active_tab == "tab-simulation":
            return create_simulation_chart(demand_df, capacity_df, service)
        elif active_tab == "tab-scenarios":
            return create_scenario_chart(
                demand_df, capacity_df, service, demand_change, capacity_change
            )

        return html.P("Select a tab to view analysis.")

    @app.callback(
        Output("model-metrics", "children"),
        [
            Input("run-analysis-btn", "n_clicks"),
            Input("chart-tabs", "active_tab"),
        ],
        [
            State("demand-data-store", "data"),
            State("service-dropdown", "value"),
            State("forecast-model", "value"),
        ],
    )
    def update_model_metrics(
        n_clicks: Optional[int],
        active_tab: str,
        demand_data: list,
        service: str,
        forecast_model: str,
    ) -> html.Div:
        """Update model performance metrics."""
        if not demand_data or active_tab != "tab-forecast":
            return html.P("Run analysis to see model metrics.")

        demand_df = pd.DataFrame(demand_data)
        service_demand = demand_df[demand_df["service"] == service].copy()

        if len(service_demand) < 30:
            return html.P("Insufficient data for model metrics.")

        # Prepare time series
        loader = DataLoader()
        ts = loader.prepare_timeseries(service_demand, "referral_date")

        if len(ts) < 30:
            return html.P("Insufficient time series data.")

        # Get forecaster
        forecaster = get_forecaster(forecast_model)

        try:
            forecaster.fit(ts)
            result = forecaster.predict(30)
            metrics = result.fit_metrics or {}

            return html.Div(
                [
                    html.P([html.Strong("Model: "), result.model_name]),
                    html.P([html.Strong("MAE: "), f"{metrics.get('MAE', 0):.2f}"]),
                    html.P([html.Strong("RMSE: "), f"{metrics.get('RMSE', 0):.2f}"]),
                    html.P([html.Strong("MAPE: "), f"{metrics.get('MAPE', 0):.1f}%"]),
                ]
            )
        except Exception as e:
            return html.P(f"Error calculating metrics: {str(e)}")

    @app.callback(
        Output("recommendations", "children"),
        [
            Input("kpi-utilisation", "children"),
            Input("kpi-wait-time", "children"),
        ],
        State("service-dropdown", "value"),
    )
    def update_recommendations(
        utilisation: str,
        wait_time: str,
        service: str,
    ) -> html.Div:
        """Generate recommendations based on current metrics."""
        try:
            util_value = float(utilisation.replace("%", "")) / 100
            wait_value = float(wait_time)
        except (ValueError, AttributeError):
            return html.P("Run analysis to see recommendations.")

        recommendations = []

        if util_value > 0.95:
            recommendations.append(
                dbc.Alert(
                    [
                        html.Strong("High Utilisation: "),
                        f"At {util_value:.0%} utilisation, consider increasing capacity.",
                    ],
                    color="danger",
                )
            )
        elif util_value > 0.85:
            recommendations.append(
                dbc.Alert(
                    [
                        html.Strong("Optimal Range: "),
                        f"Utilisation at {util_value:.0%} is within target range.",
                    ],
                    color="success",
                )
            )
        else:
            recommendations.append(
                dbc.Alert(
                    [
                        html.Strong("Low Utilisation: "),
                        f"At {util_value:.0%}, there may be excess capacity.",
                    ],
                    color="warning",
                )
            )

        if wait_value > 28:
            recommendations.append(
                dbc.Alert(
                    [
                        html.Strong("Long Wait Times: "),
                        f"Average wait of {wait_value:.0f} days exceeds 4-week target.",
                    ],
                    color="danger",
                )
            )
        elif wait_value > 14:
            recommendations.append(
                dbc.Alert(
                    [
                        html.Strong("Wait Times: "),
                        f"Average wait of {wait_value:.0f} days is within acceptable range.",
                    ],
                    color="info",
                )
            )

        return html.Div(recommendations) if recommendations else html.P("No recommendations.")


def get_forecaster(model_name: str):
    """Get the appropriate forecaster based on model name."""
    forecasters = {
        "arima": ARIMAForecaster(auto=True),
        "ets": ETSForecaster(auto=True),
        "ml": MLForecaster(),
        "ensemble": EnsembleForecaster(),
    }
    return forecasters.get(model_name, EnsembleForecaster())


def create_forecast_chart(
    demand_df: pd.DataFrame,
    service: str,
    horizon: int,
    model_name: str,
) -> dcc.Graph:
    """Create the demand forecast chart."""
    service_demand = demand_df[demand_df["service"] == service].copy()

    if len(service_demand) < 30:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="Insufficient data for forecasting",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        )

    # Prepare time series
    loader = DataLoader()
    ts = loader.prepare_timeseries(service_demand, "referral_date")

    # Ensure minimum length
    if len(ts) < 30:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="Insufficient time series data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        )

    # Run forecast
    forecaster = get_forecaster(model_name)

    try:
        result = forecaster.fit_predict(ts, steps=horizon)
    except Exception as e:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text=f"Forecasting error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        )

    # Create figure
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts.values,
            mode="lines",
            name="Historical",
            line=dict(color="blue"),
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast.values,
            mode="lines",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        )
    )

    # Confidence interval
    if result.lower_bound is not None and result.upper_bound is not None:
        fig.add_trace(
            go.Scatter(
                x=list(result.forecast.index) + list(result.forecast.index[::-1]),
                y=list(result.upper_bound.values) + list(result.lower_bound.values[::-1]),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color="rgba(255, 0, 0, 0)"),
                name=f"{result.confidence_level:.0%} CI",
            )
        )

    fig.update_layout(
        title=f"Demand Forecast - {service} ({result.model_name})",
        xaxis_title="Date",
        yaxis_title="Daily Referrals",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig)


def create_capacity_chart(
    demand_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    service: str,
) -> dcc.Graph:
    """Create the capacity analysis chart."""
    # Calculate weekly demand and capacity
    service_demand = demand_df[demand_df["service"] == service].copy()
    service_capacity = capacity_df[capacity_df["service"] == service].iloc[0]

    service_demand["referral_date"] = pd.to_datetime(service_demand["referral_date"])

    weekly = (
        service_demand.groupby(pd.Grouper(key="referral_date", freq="W"))
        .size()
        .reset_index(name="demand")
    )

    # Calculate capacity
    model = CapacityModel()
    weekly_capacity = model.calculate_weekly_capacity(
        service_capacity["staff_fte"],
        service_capacity["avg_appointment_mins"],
    )

    fig = go.Figure()

    # Demand bars
    fig.add_trace(
        go.Bar(
            x=weekly["referral_date"],
            y=weekly["demand"],
            name="Weekly Demand",
            marker_color="steelblue",
        )
    )

    # Capacity line
    fig.add_trace(
        go.Scatter(
            x=weekly["referral_date"],
            y=[weekly_capacity] * len(weekly),
            mode="lines",
            name="Weekly Capacity",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Demand vs Capacity - {service}",
        xaxis_title="Week",
        yaxis_title="Appointments",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig)


def create_simulation_chart(
    demand_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    service: str,
) -> dcc.Graph:
    """Create the queue simulation chart."""
    service_demand = demand_df[demand_df["service"] == service].copy()
    service_capacity = capacity_df[capacity_df["service"] == service].iloc[0]

    # Calculate daily arrivals and capacity
    service_demand["referral_date"] = pd.to_datetime(service_demand["referral_date"])
    daily_arrivals = service_demand.groupby("referral_date").size().mean()

    model = CapacityModel()
    weekly_capacity = model.calculate_weekly_capacity(
        service_capacity["staff_fte"],
        service_capacity["avg_appointment_mins"],
    )
    daily_capacity = int(weekly_capacity / 5)

    # Run simulation
    sim = QueueSimulation(daily_capacity=daily_capacity, seed=42)
    result = sim.run(days=365, daily_arrivals=daily_arrivals, service_name=service)

    daily_stats = result.daily_stats

    fig = go.Figure()

    # Queue length over time
    fig.add_trace(
        go.Scatter(
            x=daily_stats["day"],
            y=daily_stats["queue_length"],
            mode="lines",
            name="Queue Length",
            line=dict(color="orange"),
        )
    )

    # Add rolling average
    rolling_avg = daily_stats["queue_length"].rolling(window=7).mean()
    fig.add_trace(
        go.Scatter(
            x=daily_stats["day"],
            y=rolling_avg,
            mode="lines",
            name="7-day Average",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title=f"Queue Simulation - {service} (Avg Wait: {result.average_wait_time:.1f} days)",
        xaxis_title="Simulation Day",
        yaxis_title="Queue Length",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig)


def create_scenario_chart(
    demand_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    service: str,
    demand_change: float,
    capacity_change: float,
) -> dcc.Graph:
    """Create the scenario comparison chart."""
    service_demand = demand_df[demand_df["service"] == service].copy()
    service_capacity = capacity_df[capacity_df["service"] == service].iloc[0]

    # Calculate base metrics
    service_demand["referral_date"] = pd.to_datetime(service_demand["referral_date"])
    daily_arrivals = service_demand.groupby("referral_date").size().mean()

    model = CapacityModel()
    weekly_capacity = model.calculate_weekly_capacity(
        service_capacity["staff_fte"],
        service_capacity["avg_appointment_mins"],
    )
    daily_capacity = int(weekly_capacity / 5)

    # Run scenarios
    sim = QueueSimulation(daily_capacity=daily_capacity, seed=42)

    scenarios = sim.run_scenarios(
        base_arrivals=daily_arrivals,
        base_capacity=daily_capacity,
        days=180,
        demand_scenarios=[0.8, 0.9, 1.0, 1.1, 1.2],
        capacity_scenarios=[1.0],
        service_name=service,
    )

    # Create heatmap data
    fig = px.bar(
        scenarios,
        x="demand_multiplier",
        y="avg_wait_time",
        title=f"Impact of Demand Changes - {service}",
        labels={
            "demand_multiplier": "Demand Multiplier",
            "avg_wait_time": "Average Wait Time (days)",
        },
        color="avg_wait_time",
        color_continuous_scale="RdYlGn_r",
    )

    # Add current scenario marker
    current_mult = 1 + demand_change / 100
    fig.add_vline(
        x=current_mult,
        line_dash="dash",
        line_color="blue",
        annotation_text="Current",
    )

    fig.update_layout(
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig)
