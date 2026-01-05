"""
NHFT Demand-Capacity Dashboard Application
===========================================
Interactive Dash application for visualising demand and capacity data.
"""

import logging
from pathlib import Path
from typing import Optional

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data_handler import get_data_handler, DataHandler

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Initialise Data Handler
# ---------------------------------------------------------------------
try:
    data_handler = get_data_handler()
    logger.info("✅ Data handler initialised successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialise data handler: {e}")
    data_handler = None

# ---------------------------------------------------------------------
# Initialise Dash App
# ---------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="NHFT Demand-Capacity Dashboard",
    suppress_callback_exceptions=True,
)


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def create_metric_card(
    title: str, value: str, icon: str = "📊", color: str = "primary"
):
    """Create a metric card component."""
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4(
                        [html.Span(icon, className="me-2"), title],
                        className="card-title",
                    ),
                    html.H2(
                        value,
                        className="text-center mt-3",
                        style={"color": f"var(--bs-{color})"},
                    ),
                ]
            )
        ],
        className="mb-3 shadow-sm",
    )


def create_filter_section():
    """Create the filter controls section."""
    if data_handler is None:
        return html.Div("Data not available", className="alert alert-danger")

    # Get unique service lines with provider codes
    if data_handler.patient_df is not None:
        service_line_data = (
            data_handler.patient_df[["providercodecurrent", "service_line"]]
            .drop_duplicates()
            .sort_values(["providercodecurrent", "service_line"])
        )
        service_line_options = [
            {
                "label": f"{row['providercodecurrent']} - {row['service_line']}",
                "value": f"{row['providercodecurrent']}|{row['service_line']}",
            }
            for _, row in service_line_data.iterrows()
        ]
    else:
        service_line_options = []

    min_date, max_date = data_handler.get_date_range()

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label("Date Range:", className="fw-bold"),
                                width="auto",
                            ),
                            dbc.Col(
                                dcc.DatePickerRange(
                                    id="date-range-picker",
                                    start_date=min_date,
                                    end_date=max_date,
                                    display_format="YYYY-MM-DD",
                                ),
                                width=True,
                            ),
                        ],
                        className="align-items-center g-2",
                    )
                ],
                width=5,
            ),
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Label("Service Lines:", className="fw-bold"),
                                width="auto",
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="service-line-dropdown",
                                    options=service_line_options,
                                    multi=True,
                                    placeholder="Select service lines (all if none selected)",
                                ),
                                width=True,
                            ),
                        ],
                        className="align-items-center g-2",
                    )
                ],
                width=5,
            ),
            dbc.Col(
                [
                    dbc.Button(
                        "Apply Filters",
                        id="apply-filters-btn",
                        color="primary",
                        className="w-100",
                    ),
                ],
                width=2,
            ),
        ],
        className="mb-4 justify-content-center",
    )


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "🏥 NHFT Demand & Capacity Platform",
                            className="text-center my-4",
                        ),
                        html.Hr(),
                    ]
                )
            ]
        ),
        # Filters Row
        create_filter_section(),
        # Summary Statistics Row
        html.Div(id="summary-stats-row"),
        # Main Content
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Tabs for different views
                        dcc.Tabs(
                            id="main-tabs",
                            value="overview-tab",
                            children=[
                                dcc.Tab(label="📈 Overview", value="overview-tab"),
                                dcc.Tab(label="👥 Demand Analysis", value="demand-tab"),
                                dcc.Tab(
                                    label="💼 Capacity Analysis", value="capacity-tab"
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Tab Content
                        html.Div(id="tab-content"),
                    ],
                    width=12,
                ),
            ]
        ),
        # Store filtered data
        dcc.Store(id="filtered-data-store"),
        # Footer
        html.Div(
            id="footer-content",
            className="footer mt-5 pt-4 pb-3 border-top",
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------
@app.callback(
    Output("filtered-data-store", "data"),
    [Input("apply-filters-btn", "n_clicks")],
    [
        State("date-range-picker", "start_date"),
        State("date-range-picker", "end_date"),
        State("service-line-dropdown", "value"),
    ],
)
def filter_data(n_clicks, start_date, end_date, service_lines):
    """Filter data based on user selections."""
    if data_handler is None or data_handler.patient_df is None:
        return None

    df = data_handler.patient_df.copy()

    # Apply filters
    if start_date and end_date:
        df = data_handler.filter_by_date_range(df, start_date, end_date)

    if service_lines:
        # Parse the combined provider|service_line values
        providers = []
        service_line_names = []
        for item in service_lines:
            provider, service_line = item.split("|")
            providers.append(provider)
            service_line_names.append(service_line)

        # Filter by matching both provider and service line
        df = df[
            df.apply(
                lambda row: f"{row['providercodecurrent']}|{row['service_line']}"
                in service_lines,
                axis=1,
            )
        ]

    # Store as JSON
    return df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("summary-stats-row", "children"), [Input("filtered-data-store", "data")]
)
def update_summary_stats(data_json):
    """Update summary statistics cards for the selected date range."""
    if data_json is None or data_handler is None:
        return html.Div("No data available", className="alert alert-warning")

    df = pd.read_json(data_json, orient="split")

    if df.empty:
        return html.Div(
            "No data matches the selected filters", className="alert alert-info"
        )

    # Ensure periodend is datetime
    df["periodend"] = pd.to_datetime(df["periodend"])

    # Calculate metrics across the entire selected date range
    total_referrals = int(df["referrals"].sum())
    total_waiters = int(df["waiters"].sum())
    total_caseload = int(df["caseload"].sum())
    total_contacts = int(df["totalcontacts"].sum())
    total_discharges = int(df["dischargesfromcaseload"].sum())

    return dbc.Row(
        [
            dbc.Col(
                create_metric_card(
                    "Referrals", f"{total_referrals:,}", "📥", "primary"
                ),
                width=2,
            ),
            dbc.Col(
                create_metric_card("Waiters", f"{total_waiters:,}", "⏳", "warning"),
                width=2,
            ),
            dbc.Col(
                create_metric_card("Caseload", f"{total_caseload:,}", "👥", "info"),
                width=2,
            ),
            dbc.Col(
                create_metric_card("Contacts", f"{total_contacts:,}", "📞", "success"),
                width=2,
            ),
            dbc.Col(
                create_metric_card(
                    "Discharges", f"{total_discharges:,}", "✅", "secondary"
                ),
                width=2,
            ),
        ],
        className="mb-4 justify-content-center",
    )


@app.callback(
    Output("footer-content", "children"), [Input("filtered-data-store", "data")]
)
def update_footer(data_json):
    """Update footer with data last refreshed date."""
    if data_json is None or data_handler is None:
        latest_date = "N/A"
    else:
        df = pd.read_json(data_json, orient="split")
        if not df.empty:
            df["periodend"] = pd.to_datetime(df["periodend"])
            latest_date = df["periodend"].max().strftime("%d %B %Y")
        else:
            latest_date = "N/A"

    return html.P(
        [
            "© 2025 Northamptonshire Healthcare Foundation Trust | ",
            "BI Development Team | ",
            "Data Science Project | ",
            html.Strong(f"Data Last Refreshed: {latest_date}"),
        ],
        className="text-center text-muted mb-0",
    )


@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "value"), Input("filtered-data-store", "data")],
)
def update_tab_content(active_tab, data_json):
    """Update content based on selected tab."""
    if data_json is None:
        return html.Div(
            "No data available. Click 'Apply Filters' to load data.",
            className="alert alert-info",
        )

    df = pd.read_json(data_json, orient="split")
    df["periodend"] = pd.to_datetime(df["periodend"])

    if df.empty:
        return html.Div(
            "No data matches the selected filters", className="alert alert-warning"
        )

    if active_tab == "overview-tab":
        return create_overview_tab(df)
    elif active_tab == "demand-tab":
        return create_demand_tab(df)
    elif active_tab == "capacity-tab":
        return create_capacity_tab(df)

    return html.Div("Invalid tab selection")


def create_overview_tab(df):
    """Create overview tab content."""
    # Time series of key metrics
    df_sorted = df.sort_values("periodend")

    # Aggregate by month
    monthly = (
        df_sorted.groupby("year_month")
        .agg(
            {
                "referrals": "sum",
                "waiters": "sum",
                "caseload": "sum",
                "totalcontacts": "sum",
            }
        )
        .reset_index()
    )

    # Create multi-line chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["referrals"],
            name="Referrals",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["waiters"],
            name="Waiters",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["caseload"],
            name="Caseload",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title="Key Metrics Over Time",
        xaxis_title="Period",
        yaxis_title="Count",
        hovermode="x unified",
        height=500,
    )

    return html.Div(
        [
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H4("Data Summary", className="mt-4"),
            dbc.Table.from_dataframe(
                df.groupby("service_line")
                .agg(
                    {
                        "referrals": "sum",
                        "waiters": "sum",
                        "caseload": "sum",
                        "totalcontacts": "sum",
                    }
                )
                .reset_index()
                .round(0),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
            ),
        ]
    )


def create_demand_tab(df):
    """Create demand analysis tab content - stub for future development."""
    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("👥 Demand Analysis", className="alert-heading"),
                    html.P(
                        "This section is currently under development. "
                        "Future features will include:"
                    ),
                    html.Ul(
                        [
                            html.Li("Referrals vs Discharges trends"),
                            html.Li("Waiting list analysis"),
                            html.Li("Demand forecasting"),
                            html.Li("Service capacity utilisation"),
                        ]
                    ),
                ],
                color="info",
                className="mt-4",
            )
        ]
    )


def create_capacity_tab(df):
    """Create capacity analysis tab content - stub for future development."""
    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("💼 Capacity Analysis", className="alert-heading"),
                    html.P(
                        "This section is currently under development. "
                        "Future features will include:"
                    ),
                    html.Ul(
                        [
                            html.Li("Staff capacity by service line"),
                            html.Li("Staff to caseload ratios"),
                            html.Li("Workforce planning insights"),
                            html.Li("Productivity metrics"),
                        ]
                    ),
                ],
                color="info",
                className="mt-4",
            )
        ]
    )


# ---------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting NHFT Demand-Capacity Dashboard...")
    logger.info("Dashboard will be available at: http://127.0.0.1:8050/")

    app.run(debug=True, host="127.0.0.1", port=8050)
