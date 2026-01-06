"""
NHFT Demand-Capacity Dashboard Application
===========================================
Interactive Dash application for visualising demand and capacity data.
"""

import logging
from pathlib import Path
from typing import Optional
from io import StringIO

import dash
from dash import dcc, html, Input, Output, State, dash_table
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

    # Build dropdown options from patient data.
    # UX: show the full service_line label (user-friendly)
    # Behaviour: filter by providercodecurrent (filters both patient + staffing reliably)
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
                                html.Label("Service:", className="fw-bold"),
                                width="auto",
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="service-line-dropdown",
                                    options=service_line_options,
                                    multi=True,
                                    placeholder="Select service(s) (all if none selected)",
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
        # Users select service labels, but we filter by provider code only.
        selected_providers = sorted({item.split("|")[0] for item in service_lines})
        df = df[df["providercodecurrent"].astype(str).isin(selected_providers)]

    # Store as JSON
    return df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("summary-stats-row", "children"), [Input("filtered-data-store", "data")]
)
def update_summary_stats(data_json):
    """Update summary statistics cards for the selected date range."""
    if data_json is None or data_handler is None:
        return html.Div("No data available", className="alert alert-warning")

    df = pd.read_json(StringIO(data_json), orient="split")

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
        df = pd.read_json(StringIO(data_json), orient="split")

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

    df = pd.read_json(StringIO(data_json), orient="split")

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

    # -----------------------------
    # Key Metrics (Time series chart)
    # -----------------------------
    df_sorted = df.sort_values("periodend")

    monthly = (
        df_sorted.groupby("year_month")
        .agg(
            {
                "referrals": "sum",
                "waiters": "sum",
                "caseload": "sum",
                "clockstopactuals": "sum",
            }
        )
        .reset_index()
    )

    fig_timeseries = go.Figure()

    fig_timeseries.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["referrals"],
            name="Referrals",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig_timeseries.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["waiters"],
            name="Waiters",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig_timeseries.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["caseload"],
            name="Caseload",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig_timeseries.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["clockstopactuals"],
            name="Clock Stop Actuals",
            mode="lines+markers",
            line=dict(width=2),
        )
    )

    fig_timeseries.update_layout(
        title="Key Metrics Over Time",
        xaxis_title="Period",
        yaxis_title="Count",
        hovermode="x unified",
        height=500,
    )

    # -----------------------------
    # 18-week waiters split (stacked bar)
    # -----------------------------
    waiters_18wk = df_sorted.groupby("year_month", as_index=False).agg(
        waitersunder18weeks=("waitersunder18weeks", "sum"),
        waiters18plusweeks=("waiters18plusweeks", "sum"),
    )

    fig_18wk = go.Figure()

    fig_18wk.add_trace(
        go.Bar(
            x=waiters_18wk["year_month"],
            y=waiters_18wk["waitersunder18weeks"],
            name="Under 18 Weeks",
            marker_color="#636EFA",  # Plotly blue
        )
    )

    fig_18wk.add_trace(
        go.Bar(
            x=waiters_18wk["year_month"],
            y=waiters_18wk["waiters18plusweeks"],
            name="18+ Weeks",
            marker_color="#EF553B",  # Plotly red
        )
    )

    fig_18wk.update_layout(
        title="Waiting List Breakdown: Under vs Over 18 Weeks",
        xaxis_title="Period",
        yaxis_title="Number of Waiters",
        barmode="stack",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    waiters_chart = dbc.Row(
        [
            dbc.Col(
                dcc.Graph(figure=fig_18wk),
                width=12,
            ),
        ],
        className="mb-4",
    )

    # -----------------------------
    # Layout
    # -----------------------------
    return html.Div(
        [
            # Main time series chart
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_timeseries), width=12),
                ],
                className="mb-4",
            ),
            # Stacked 18-week bar chart
            waiters_chart,
        ]
    )


def create_service_line_summary_table(df: pd.DataFrame):
    """Create the patient metrics summary table by provider code + service line."""
    exclude_cols = {
        # Keep these as identifier columns rather than aggregated numeric metrics
        "providercodecurrent",
        "service_line",
        "periodend",
        "year",
        "month",
        "quarter",
    }

    summary_columns = [
        col
        for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    summary_df = (
        df.groupby(["providercodecurrent", "service_line"])[summary_columns]
        .sum()
        .reset_index()
        .round(0)
        .sort_values(["providercodecurrent", "service_line"])
    )

    return dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in summary_df.columns],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "10px",
        },
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
    )


def create_staffing_pivot_table(
    patient_df: pd.DataFrame,
    staffing_df: Optional[pd.DataFrame],
):
    """Create a staffing pivot table: rows=provider+service_line, cols=staff_group, values=staff."""
    if staffing_df is None or staffing_df.empty:
        return html.Div(
            "No staffing data available.",
            className="alert alert-warning",
        )

    required_cols = {"providercodecurrent", "service_line", "staff_group", "staff"}
    if not required_cols.issubset(set(staffing_df.columns)):
        missing = sorted(required_cols - set(staffing_df.columns))
        return html.Div(
            f"Staffing data is missing required columns: {', '.join(missing)}",
            className="alert alert-danger",
        )

    selected_providers: list[str] = []
    if (
        patient_df is not None
        and not patient_df.empty
        and "providercodecurrent" in patient_df.columns
    ):
        selected_providers = (
            patient_df["providercodecurrent"].dropna().astype(str).unique().tolist()
        )

    staffing_filtered = staffing_df.copy()
    if selected_providers:
        staffing_filtered = staffing_filtered[
            staffing_filtered["providercodecurrent"]
            .astype(str)
            .isin(selected_providers)
        ]

    if staffing_filtered.empty:
        return html.Div(
            "No staffing rows match the selected provider(s).",
            className="alert alert-info",
        )

    staffing_filtered["staff"] = pd.to_numeric(
        staffing_filtered["staff"], errors="coerce"
    ).fillna(0)

    pivot_df = (
        staffing_filtered.groupby(
            ["providercodecurrent", "service_line", "staff_group"], as_index=False
        )
        .agg(staff=("staff", "sum"))
        .pivot(
            index=["providercodecurrent", "service_line"],
            columns="staff_group",
            values="staff",
        )
        .fillna(0)
        .reset_index()
    )

    # Keep columns stable-ish: identifiers first, then alphabetical staff groups
    id_cols = ["providercodecurrent", "service_line"]
    non_index_cols = [c for c in pivot_df.columns if c not in id_cols]
    pivot_df = pivot_df[id_cols + sorted(non_index_cols, key=lambda x: str(x))]

    pivot_df = pivot_df.sort_values(["providercodecurrent", "service_line"])

    # Render integers when possible
    for col in non_index_cols:
        try:
            pivot_df[col] = pivot_df[col].round(0).astype(int)
        except Exception:
            pivot_df[col] = pivot_df[col].round(0)

    return dash_table.DataTable(
        data=pivot_df.to_dict("records"),
        columns=[{"name": str(i), "id": str(i)} for i in pivot_df.columns],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "minWidth": "120px",
            "maxWidth": "240px",
            "whiteSpace": "normal",
        },
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
    )


def create_demand_tab(df):
    """Create demand analysis tab content."""

    summary_table = create_service_line_summary_table(df)

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
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Patients Data by Service Line", className="mb-3"),
                            summary_table,
                        ],
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
        ]
    )


def create_capacity_tab(df):
    """Create capacity analysis tab content - stub for future development."""
    if data_handler is None:
        return html.Div("Data not available", className="alert alert-danger")

    staffing_table = create_staffing_pivot_table(df, data_handler.staffing_df)

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
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Staffing by Service Line", className="mb-3"),
                            staffing_table,
                        ],
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
        ]
    )


# ---------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting NHFT Demand-Capacity Dashboard...")
    logger.info("Dashboard will be available at: http://127.0.0.1:8050/")

    app.run(debug=True, host="127.0.0.1", port=8050)
