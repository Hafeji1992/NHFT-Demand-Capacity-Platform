"""NHFT Demand-Capacity Dashboard Application
=============================================
Interactive Dash application for visualising demand and capacity data.
"""

import sys
import logging
from collections import OrderedDict
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

# Allow imports from sibling folders under `src/` when running as:
#   python src/dashboard/app.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting.forecast import ForecastConfig, make_forecast_frame
from forecasting.sarima import SarimaSpec, small_grid_search_aic


# ---------------------------------------------------------------------
# Forecast Caching (keep dashboard interactions responsive)
# ---------------------------------------------------------------------
# Dash callbacks can fire frequently (legend toggles, filter changes). Keeping a
# small in-memory cache of selected model specs and forecast frames keyed by the
# input series signature.
_FORECAST_CACHE_MAX = 64
_SPEC_CACHE_MAX = 128
_forecast_frame_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_best_spec_cache: "OrderedDict[tuple, SarimaSpec]" = OrderedDict()


def _lru_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _lru_set(cache: OrderedDict, key, value, *, maxsize: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(maxsize):
        cache.popitem(last=False)


def _series_signature(y: pd.Series) -> tuple:
    """Return a stable, hashable signature for a monthly time series."""
    y = y.sort_index()
    # Include length + endpoints and a fast content hash for low collision risk.
    content_hash = int(pd.util.hash_pandas_object(y, index=True).sum())
    start = int(y.index[0].value) if len(y) else 0
    end = int(y.index[-1].value) if len(y) else 0
    return (int(len(y)), start, end, content_hash)


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Initialise Data Handler
# ---------------------------------------------------------------------
try:
    data_handler = get_data_handler()
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
def prettify_column_name(column_id: str) -> str:
    """Format column headers into user-friendly table headers.

    Rules:
    - Converts snake_case / kebab-case into spaced words.
    - Title-cases human-entered headers too (e.g., "referral contacts" -> "Referral Contacts").
    - Uppercases 3-letter alphabetic tokens to preserve acronyms (e.g., ftf -> FTF).
    - Preserves already-uppercase tokens (e.g., NHS, WTE).
    """

    raw = str(column_id)
    normalised = raw.replace("_", " ").replace("-", " ")
    parts: list[str] = [p for p in normalised.split() if p]

    pretty_parts: list[str] = []
    for part in parts:
        if part.isdigit():
            pretty_parts.append(part)
            continue

        if part.isalpha():
            if part.isupper():
                pretty_parts.append(part)
                continue

            if len(part) == 3:
                pretty_parts.append(part.upper())
                continue

            pretty_parts.append(part[:1].upper() + part[1:].lower())
            continue

        # For tokens with punctuation (e.g., "A/B"), leave unchanged.
        pretty_parts.append(part)

    return " ".join(pretty_parts)


def create_metric_card(
    title: str, value: str, icon: str = "📊", color: str = "primary"
):
    """Create a small KPI/metric card for the dashboard.

    Args:
        title: Card title label.
        value: Pre-formatted value to display (e.g., "1,234").
        icon: Emoji/icon prefix for the title.
        color: Bootstrap theme colour name (e.g., "primary", "warning").

    Returns:
        A Dash Bootstrap Components Card.
    """
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


def _rgba(color: str, alpha: float) -> str:
    """Convert a Plotly colour string (hex or rgb) to an rgba string."""
    c = str(color).strip()
    a = max(0.0, min(1.0, float(alpha)))

    if c.startswith("#") and len(c) in {4, 7}:
        if len(c) == 4:
            r = int(c[1] * 2, 16)
            g = int(c[2] * 2, 16)
            b = int(c[3] * 2, 16)
        else:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{a})"

    if c.lower().startswith("rgb(") and c.endswith(")"):
        inner = c[c.find("(") + 1 : -1]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) == 3:
            return f"rgba({parts[0]},{parts[1]},{parts[2]},{a})"

    # Fallback: keep a neutral band if we can't parse
    return f"rgba(0,0,0,{a})"


def create_filter_section():
    """Create the top-of-page filter controls.

    The dropdown displays labels as "ProviderCodeCurrent - Service Line" from the
    patient dataset, but filtering logic uses provider codes so it can be applied
    consistently across both patient and staffing datasets.

    Returns:
        A Dash layout component (dbc.Row) containing date + service filters.
    """
    if data_handler is None:
        return html.Div("Data not available", className="alert alert-danger")

    # Build dropdown options from patient data.
    # UX: show the full service_line label (user-friendly)
    # Behaviour: filter by provider_code_current (filters both patient + staffing reliably)
    if data_handler.patient_df is not None:
        service_line_data = (
            data_handler.patient_df[["provider_code_current", "service_line"]]
            .drop_duplicates()
            .sort_values(["provider_code_current", "service_line"])
        )
        service_line_options = [
            {
                "label": f"{row['provider_code_current']} - {row['service_line']}",
                "value": f"{row['provider_code_current']}|{row['service_line']}",
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
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Switch(
                                        id="forecast-toggle",
                                        label="Forecast (SARIMA, 95% CI)",
                                        value=False,
                                    ),
                                    width="auto",
                                    style={
                                        # Align this control with the chart legend region
                                        "marginLeft": "auto",
                                        "marginRight": "200px",
                                    },
                                )
                            ],
                            className="mb-2",
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
    """Filter patient data based on the UI controls.

    Args:
        n_clicks: Number of clicks on the "Apply Filters" button.
        start_date: Start date (YYYY-MM-DD) from the date picker.
        end_date: End date (YYYY-MM-DD) from the date picker.
        service_lines: List of selected dropdown values in the form
            "provider_code_current|service_line".

    Returns:
        JSON-encoded filtered patient DataFrame (orient="split"), or None if data
        is unavailable.
    """
    if data_handler is None or data_handler.patient_df is None:
        return None

    df = data_handler.patient_df.copy()

    # Apply filters
    if start_date and end_date:
        df = data_handler.filter_by_date_range(df, start_date, end_date)

    if service_lines:
        # Users select service labels, but we filter by provider code only.
        selected_providers = sorted({item.split("|")[0] for item in service_lines})
        df = df[df["provider_code_current"].astype(str).isin(selected_providers)]

    # Store as JSON
    return df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("summary-stats-row", "children"), [Input("filtered-data-store", "data")]
)
def update_summary_stats(data_json):
    """Update the KPI cards row based on the filtered dataset.

    Args:
        data_json: JSON-encoded DataFrame (orient="split") from dcc.Store.

    Returns:
        A Dash component representing the KPI cards row, or an alert div when no
        data is available.
    """
    if data_json is None or data_handler is None:
        return html.Div("No data available", className="alert alert-warning")

    df = pd.read_json(StringIO(data_json), orient="split")

    if df.empty:
        return html.Div(
            "No data matches the selected filters", className="alert alert-info"
        )

    # Ensure period_end is datetime
    df["period_end"] = pd.to_datetime(df["period_end"])

    # Calculate metrics across the entire selected date range
    total_referrals = int(df["referrals"].sum())
    total_waiters = int(df["waiters"].sum())
    total_caseload = int(df["caseload"].sum())
    total_contacts = int(df["total_contacts"].sum())
    total_discharges = int(df["discharges_from_caseload"].sum())

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
    """Update footer text, including the latest reporting period end date.

    Args:
        data_json: JSON-encoded DataFrame (orient="split") from dcc.Store.

    Returns:
        A Dash HTML paragraph element with the latest reporting period end date.
    """
    if data_json is None or data_handler is None:
        latest_period_end = "N/A"
    else:
        df = pd.read_json(StringIO(data_json), orient="split")

        if not df.empty:
            df["period_end"] = pd.to_datetime(df["period_end"])
            latest_period_end = df["period_end"].max().strftime("%d %B %Y")
        else:
            latest_period_end = "N/A"

    return html.P(
        [
            "© 2025 Northamptonshire Healthcare NHS Foundation Trust | ",
            "BI Development Team | ",
            html.Strong(f"Latest Reporting Period End: {latest_period_end}"),
        ],
        className="text-center text-muted mb-0",
    )


@app.callback(
    Output("tab-content", "children"),
    [
        Input("main-tabs", "value"),
        Input("filtered-data-store", "data"),
    ],
)
def update_tab_content(active_tab, data_json):
    """Render the selected tab content.

    Args:
        active_tab: Selected tab ID ("overview-tab", "demand-tab", "capacity-tab").
        data_json: JSON-encoded DataFrame (orient="split") from dcc.Store.

    Returns:
        A Dash component containing the tab content (charts/tables), or an alert
        div if data is missing/empty.
    """
    if data_json is None:
        return html.Div(
            "No data available. Click 'Apply Filters' to load data.",
            className="alert alert-info",
        )

    df = pd.read_json(StringIO(data_json), orient="split")

    df["period_end"] = pd.to_datetime(df["period_end"])

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


@app.callback(
    Output("forecast-toggle", "disabled"),
    [Input("main-tabs", "value")],
)
def disable_forecast_toggle_when_not_overview(active_tab: str) -> bool:
    return active_tab != "overview-tab"


def create_overview_tab(df):
    """Create the Overview tab layout.

    Args:
        df: Filtered patient dataset.

    Returns:
        A Dash layout component containing the overview charts.
    """

    # -----------------------------
    # Key Metrics (Time series chart)
    # -----------------------------
    df_sorted = df.sort_values("period_end")

    monthly = (
        df_sorted.groupby("year_month")
        .agg(
            {
                "referrals": "sum",
                "waiters": "sum",
                "caseload": "sum",
                "discharges_from_caseload": "sum",
                "total_contacts": "sum",
                "clock_stop_actuals": "sum",
            }
        )
        .reset_index()
    )

    # Use a real datetime x-axis at month-end so forecast + actuals join smoothly.
    # Keep month bucket labels in hover/ticks via formatting.
    try:
        monthly_x = pd.PeriodIndex(
            monthly["year_month"].astype(str), freq="M"
        ).to_timestamp("M")
    except Exception:
        monthly_x = monthly["year_month"].astype(str)

    fig_timeseries = go.Figure()

    # Use Plotly's default qualitative palette for consistent, accessible colours
    palette = px.colors.qualitative.Plotly
    series = [
        ("Referrals", "referrals"),
        ("Waiters", "waiters"),
        ("Caseload", "caseload"),
        ("Contacts", "total_contacts"),
        ("Discharges", "discharges_from_caseload"),
        ("Clock Stop Actuals", "clock_stop_actuals"),
    ]

    for i, (label, col) in enumerate(series):
        if col not in monthly.columns:
            continue

        # Only show Referrals + Clock Stop Actuals by default; keep others selectable.
        default_visible = col in {"referrals", "clock_stop_actuals"}
        color = palette[i % len(palette)]

        # Line trace
        fig_timeseries.add_trace(
            go.Scatter(
                x=monthly_x,
                y=monthly[col],
                name=label,
                legendgroup=col,
                showlegend=False,
                mode="lines",
                visible=True if default_visible else "legendonly",
                line=dict(width=3, shape="spline", color=color),
                # With hovermode='x unified', keep the date in the unified header (shown once)
                hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
            )
        )

        # Marker trace (legend entry shown as circle only)
        fig_timeseries.add_trace(
            go.Scatter(
                x=monthly_x,
                y=monthly[col],
                name=label,
                legendgroup=col,
                showlegend=True,
                mode="markers",
                visible=True if default_visible else "legendonly",
                marker=dict(size=9, color=color),
                hoverinfo="skip",
            )
        )

    fig_timeseries.update_layout(
        title=dict(
            text="Key Metrics Over Time",
            x=0.5,
            xanchor="center",
            font=dict(size=22),
        ),
        xaxis_title="Period",
        yaxis_title="Number of Patients",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(
            title=dict(text="Key metrics (click to hide/show)"),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            itemsizing="constant",
        ),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        legend_groupclick="togglegroup",
        margin=dict(l=40, r=200, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig_timeseries.update_xaxes(
        showgrid=False,
        tickangle=-30,
        ticks="outside",
        ticklen=6,
        dtick="M1",
        tickformat="%B %Y",
        hoverformat="%B %Y",
    )
    fig_timeseries.update_yaxes(
        tickformat=",",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    # -----------------------------
    # 18-week waiters split (stacked bar)
    # -----------------------------
    waiters_18wk = df_sorted.groupby("year_month", as_index=False).agg(
        waiters_under_18_weeks=("waiters_under_18_weeks", "sum"),
        waiters_over_18_weeks=("waiters_over_18_weeks", "sum"),
    )

    fig_18wk = go.Figure()

    fig_18wk.add_trace(
        go.Bar(
            x=waiters_18wk["year_month"],
            y=waiters_18wk["waiters_under_18_weeks"],
            name="Under 18 Weeks",
            marker_color="#636EFA",  # Plotly blue
            hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
        )
    )

    fig_18wk.add_trace(
        go.Bar(
            x=waiters_18wk["year_month"],
            y=waiters_18wk["waiters_over_18_weeks"],
            name="18+ Weeks",
            marker_color="#EF553B",  # Plotly red
            hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
        )
    )

    fig_18wk.update_layout(
        title=dict(
            text="Waiting List Breakdown: Under vs Over 18 Weeks",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        xaxis_title="Period",
        yaxis_title="Number of Waiters",
        barmode="stack",
        height=400,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            title=dict(text="Waiters split (click to hide/show)"),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            itemsizing="constant",
        ),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        margin=dict(l=40, r=200, t=70, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig_18wk.update_xaxes(
        showgrid=False,
        tickangle=-30,
        ticks="outside",
        ticklen=6,
    )
    fig_18wk.update_yaxes(
        tickformat=",",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
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
                    dbc.Col(
                        dcc.Graph(id="key-metrics-timeseries", figure=fig_timeseries),
                        width=12,
                    ),
                ],
                className="mb-4",
            ),
            # Stacked 18-week bar chart
            waiters_chart,
        ]
    )


@app.callback(
    Output("key-metrics-timeseries", "figure"),
    [
        Input("filtered-data-store", "data"),
        Input("forecast-toggle", "value"),
        Input("key-metrics-timeseries", "restyleData"),
    ],
    [State("key-metrics-timeseries", "figure")],
)
def update_key_metrics_timeseries(data_json, forecast_on, _restyle_data, current_fig):
    """Update the key metrics time series chart.

    - Keeps current legend selections (via `current_fig`).
    - When forecast is enabled, computes SARIMA forecasts ONLY for metrics that
      are currently visible (not 'legendonly').
    """

    if data_json is None:
        return go.Figure()

    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()

    df["period_end"] = pd.to_datetime(df["period_end"])
    df_sorted = df.sort_values("period_end")

    monthly = (
        df_sorted.groupby("year_month")
        .agg(
            {
                "referrals": "sum",
                "waiters": "sum",
                "caseload": "sum",
                "discharges_from_caseload": "sum",
                "total_contacts": "sum",
                "clock_stop_actuals": "sum",
            }
        )
        .reset_index()
    )

    # Use real datetime points (month-end) for both actuals and forecasts.
    try:
        x_actual = pd.PeriodIndex(
            monthly["year_month"].astype(str), freq="M"
        ).to_timestamp("M")
    except Exception:
        x_actual = monthly["year_month"].astype(str)

    # Extract current legend visibility state (so toggling forecast doesn't reset selections)
    group_visibility = {}
    if isinstance(current_fig, dict):
        for tr in current_fig.get("data", []) or []:
            if tr.get("showlegend") is True:
                group = tr.get("legendgroup")
                if group:
                    group_visibility[group] = tr.get("visible", True)

    # Build a monthly datetime index for forecasting (month-end).
    monthly_dt_index = x_actual if isinstance(x_actual, pd.DatetimeIndex) else None

    fig = go.Figure()

    palette = px.colors.qualitative.Plotly
    series = [
        ("Referrals", "referrals"),
        ("Waiters", "waiters"),
        ("Caseload", "caseload"),
        ("Contacts", "total_contacts"),
        ("Discharges", "discharges_from_caseload"),
        ("Clock Stop Actuals", "clock_stop_actuals"),
    ]

    forecast_added = False
    forecast_errors: list[str] = []

    for i, (label, col) in enumerate(series):
        if col not in monthly.columns:
            continue

        default_visible = col in {"referrals", "clock_stop_actuals"}
        visible = group_visibility.get(col, True if default_visible else "legendonly")

        color = palette[i % len(palette)]

        # Actuals line
        fig.add_trace(
            go.Scatter(
                x=x_actual,
                y=monthly[col],
                name=label,
                legendgroup=col,
                showlegend=False,
                mode="lines",
                visible=visible,
                line=dict(width=3, shape="spline", color=color),
                hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
            )
        )

        # Legend marker
        fig.add_trace(
            go.Scatter(
                x=x_actual,
                y=monthly[col],
                name=label,
                legendgroup=col,
                showlegend=True,
                mode="markers",
                visible=visible,
                marker=dict(size=9, color=color),
                hoverinfo="skip",
            )
        )

        # Forecast overlays: only compute if forecast is ON and this metric is visible
        metric_is_visible = visible not in (False, "legendonly")
        if bool(forecast_on) and metric_is_visible and monthly_dt_index is not None:
            try:
                y = pd.Series(
                    pd.to_numeric(monthly[col], errors="coerce").astype(float).values,
                    index=monthly_dt_index,
                ).dropna()

                # Auto-select the best model *for this metric + current filtered series*
                # using a small AIC grid search. Cache both selected specs and forecasts
                # so legend toggles don't re-fit models.
                if len(y) >= 12 and y.nunique() >= 2:
                    sig = _series_signature(y)

                    spec_key = (col, sig)
                    spec = _lru_get(_best_spec_cache, spec_key)
                    if spec is None:
                        y_len = int(len(y))
                        # Keep the grid small for responsiveness.
                        p = (0, 1) if y_len < 24 else (0, 1, 2)
                        q = (0, 1) if y_len < 24 else (0, 1, 2)
                        # With fewer than ~2 seasonal cycles, avoid seasonal differencing.
                        D = (0,) if y_len < 24 else (0, 1)

                        spec = small_grid_search_aic(
                            y,
                            seasonal_period=12,
                            p=p,
                            d=(0, 1),
                            q=q,
                            P=(0, 1),
                            D=D,
                            Q=(0, 1),
                            trend="n",
                        )
                        _lru_set(
                            _best_spec_cache, spec_key, spec, maxsize=_SPEC_CACHE_MAX
                        )

                    fc_key = (col, sig, 12, 0.95, "ME")
                    future = _lru_get(_forecast_frame_cache, fc_key)
                    if future is None:
                        frame = make_forecast_frame(
                            y,
                            config=ForecastConfig(
                                months_ahead=12,
                                conf_level=0.95,
                                freq="ME",
                                auto_select=False,
                            ),
                            spec=spec,
                        )
                        future = frame[frame["is_forecast"]].copy()
                        _lru_set(
                            _forecast_frame_cache,
                            fc_key,
                            future,
                            maxsize=_FORECAST_CACHE_MAX,
                        )
                    else:
                        future = future.copy()

                    if not future.empty:
                        band_color = _rgba(color, 0.12)

                        # Plotly will not draw a segment between two separate traces.
                        # To make the forecast visually "join" the historical series,
                        # prepend the last actual point to the dashed forecast trace.
                        last_x = y.index.max()
                        last_y = float(y.iloc[-1])
                        x_fc = pd.concat(
                            [
                                pd.Series([last_x]),
                                future["period_end"].reset_index(drop=True),
                            ],
                            ignore_index=True,
                        )
                        yhat_fc = pd.concat(
                            [
                                pd.Series([last_y]),
                                future["yhat"].reset_index(drop=True),
                            ],
                            ignore_index=True,
                        )

                        # CI band (upper then lower with fill)
                        fig.add_trace(
                            go.Scatter(
                                x=future["period_end"],
                                y=future["yhat_upper"],
                                name=f"{label} (95% CI)",
                                legendgroup=col,
                                showlegend=False,
                                mode="lines",
                                line=dict(width=0),
                                hoverinfo="skip",
                                visible=True,
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=future["period_end"],
                                y=future["yhat_lower"],
                                name=f"{label} (95% CI)",
                                legendgroup=col,
                                showlegend=False,
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor=band_color,
                                hoverinfo="skip",
                                visible=True,
                            )
                        )

                        # Forecast dashed line
                        fig.add_trace(
                            go.Scatter(
                                x=x_fc,
                                y=yhat_fc,
                                name=f"{label} (Forecast)",
                                legendgroup=col,
                                showlegend=False,
                                mode="lines",
                                visible=True,
                                line=dict(width=2, dash="dash", color=color),
                                hovertemplate=f"{label} (Forecast): <b>%{{y:,.0f}}</b><extra></extra>",
                            )
                        )

                        forecast_added = True
                else:
                    forecast_errors.append(
                        f"{label}: insufficient history (need ≥12 months)"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Forecast skipped for '{col}': {e}")
                forecast_errors.append(f"{label}: {type(e).__name__}")

    fig.update_layout(
        title=dict(
            text="Key Metrics Over Time",
            x=0.5,
            xanchor="center",
            font=dict(size=22),
        ),
        xaxis_title="Period",
        yaxis_title="Number of Patients",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(
            title=dict(text="Key metrics (click to hide/show)"),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            itemsizing="constant",
        ),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        legend_groupclick="togglegroup",
        margin=dict(l=40, r=200, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        showgrid=False,
        tickangle=-30,
        ticks="outside",
        ticklen=6,
        dtick="M1",
        tickformat="%B %Y",
        hoverformat="%B %Y",
    )
    fig.update_yaxes(
        tickformat=",",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    # Make the toggle feel responsive even when no forecast could be generated.
    if bool(forecast_on) and not forecast_added:
        months_available = int(pd.Series(monthly["year_month"]).nunique())
        details = (
            "; ".join(sorted(set(forecast_errors))[:3])
            if forecast_errors
            else "No eligible metrics visible"
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            text=(
                f"Forecast ON, but no forecasts were generated. "
                f"Months available: {months_available}. {details}"
            ),
            showarrow=False,
            font=dict(size=12, color="rgba(0,0,0,0.65)"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        )

    return fig


def create_service_line_summary_table(df: pd.DataFrame):
    """Create the patient summary DataTable by provider code and service line.

    Args:
        df: Filtered patient dataset.

    Returns:
        A Dash DataTable where rows are (provider_code_current, service_line) and
        numeric metrics are aggregated by sum.
    """
    exclude_cols = {
        # Keep these as identifier columns rather than aggregated numeric metrics
        "provider_code_current",
        "service_line",
        "period_end",
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
        df.groupby(["provider_code_current", "service_line"])[summary_columns]
        .sum()
        .reset_index()
        .sort_values(["provider_code_current", "service_line"])
    )

    # -----------------------------------------------------------------
    # Derived planning ratios (computed from aggregated totals)
    # -----------------------------------------------------------------
    ratio_cols = [
        "referral_clock_stop_ratio",
        "referral_discharged_no_clock_stop_ratio",
        "demand_ratio",
        "total_contacts_per_caseload",
        "ftf_contacts_per_caseload",
        "clearance_time_months",
    ]

    def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
        denom_num = pd.to_numeric(denom, errors="coerce")
        numer_num = pd.to_numeric(numer, errors="coerce")
        return numer_num.divide(denom_num.where(denom_num != 0))

    # Ensure base columns exist (if not present in df, keep derived as NA)
    base_needed = {
        "referrals",
        "clock_stop_actuals",
        "discharges_no_clock_stop",
        "caseload",
        "total_caseload_contacts",
        "ftf_caseload_contacts",
    }
    for base in base_needed:
        if base not in summary_df.columns:
            summary_df[base] = pd.NA

    # "Current" values for clearance time are taken from the latest available period
    # per provider/service line within the filtered dataset.
    current_caseload = pd.Series(pd.NA, index=summary_df.index)
    current_clock_stops = pd.Series(pd.NA, index=summary_df.index)
    if (
        "period_end" in df.columns
        and "caseload" in df.columns
        and "clock_stop_actuals" in df.columns
    ):
        latest = (
            df.copy()
            .assign(period_end=pd.to_datetime(df["period_end"], errors="coerce"))
            .sort_values("period_end")
            .groupby(["provider_code_current", "service_line"], dropna=False)
            .tail(1)
        )

        latest = latest[["provider_code_current", "service_line", "caseload", "clock_stop_actuals"]]
        latest["caseload"] = pd.to_numeric(latest["caseload"], errors="coerce")
        latest["clock_stop_actuals"] = pd.to_numeric(
            latest["clock_stop_actuals"], errors="coerce"
        )

        merged_latest = summary_df[["provider_code_current", "service_line"]].merge(
            latest,
            on=["provider_code_current", "service_line"],
            how="left",
        )
        current_caseload = merged_latest["caseload"]
        current_clock_stops = merged_latest["clock_stop_actuals"]

    summary_df["referral_clock_stop_ratio"] = _safe_div(
        summary_df["clock_stop_actuals"], summary_df["referrals"]
    )
    summary_df["referral_discharged_no_clock_stop_ratio"] = _safe_div(
        summary_df["discharges_no_clock_stop"], summary_df["referrals"]
    )

    # As requested: Demand Ratio = Referral Clock Stop Ratio + Referral Discharged No Clock Stop Ratio
    summary_df["demand_ratio"] = pd.to_numeric(
        summary_df["referral_clock_stop_ratio"], errors="coerce"
    ) + pd.to_numeric(
        summary_df["referral_discharged_no_clock_stop_ratio"], errors="coerce"
    )

    summary_df["total_contacts_per_caseload"] = _safe_div(
        summary_df["total_caseload_contacts"], summary_df["caseload"]
    )
    summary_df["ftf_contacts_per_caseload"] = _safe_div(
        summary_df["ftf_caseload_contacts"], summary_df["caseload"]
    )

    # Clearance time (months): how long to clear the *current* caseload if no new
    # referrals arrived, at the current clock stop rate.
    summary_df["clearance_time_months"] = _safe_div(current_caseload, current_clock_stops)

    # Preserve existing behaviour for aggregated numeric metrics (rounded to 0),
    # but keep derived ratios with meaningful decimals.
    numeric_cols = [
        c
        for c in summary_df.columns
        if c not in {"provider_code_current", "service_line"}
    ]
    for c in numeric_cols:
        summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")

    for c in [c for c in numeric_cols if c not in ratio_cols]:
        try:
            summary_df[c] = summary_df[c].round(0)
        except Exception:
            pass

    for c in ratio_cols:
        try:
            summary_df[c] = summary_df[c].round(4)
        except Exception:
            pass

    # Clearance time reads better with fewer decimals
    try:
        summary_df["clearance_time_months"] = pd.to_numeric(
            summary_df["clearance_time_months"], errors="coerce"
        ).round(2)
    except Exception:
        pass

    # Keep ratio columns near related activity measures if present
    preferred_order = [
        "provider_code_current",
        "service_line",
        "referrals",
        "clock_stop_actuals",
        "discharges_no_clock_stop",
        "referral_clock_stop_ratio",
        "referral_discharged_no_clock_stop_ratio",
        "demand_ratio",
        "total_contacts",
        "ftf_contacts",
        "caseload",
        "clearance_time_months",
        "total_caseload_contacts",
        "ftf_caseload_contacts",
        "total_contacts_per_caseload",
        "ftf_contacts_per_caseload",
    ]
    existing = [c for c in preferred_order if c in summary_df.columns]
    remaining = [c for c in summary_df.columns if c not in existing]
    summary_df = summary_df[existing + remaining]

    return dash_table.DataTable(
        id="patient-summary-table",
        data=summary_df.to_dict("records"),
        columns=[
            {"name": prettify_column_name(i), "id": i} for i in summary_df.columns
        ],
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 2},
        sort_action="native",
        style_table={
            "width": "100%",
            "minWidth": "100%",
            "overflowX": "auto",
            "maxHeight": "520px",
            "overflowY": "auto",
        },
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "minWidth": "120px",
            "maxWidth": "240px",
            "whiteSpace": "normal",
        },
        style_cell_conditional=[
            {
                "if": {"column_id": "service_line"},
                "minWidth": "180px",
                "maxWidth": "320px",
            }
        ],
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
    )


def create_staffing_pivot_table(
    patient_df: pd.DataFrame,
    staffing_df: Optional[pd.DataFrame],
):
    """Create a staffing pivot DataTable by provider/service line.

    Args:
        patient_df: Filtered patient dataset (used to determine selected providers).
        staffing_df: Staffing dataset (typically data_handler.staffing_df).

    Returns:
        A Dash DataTable pivoted to one row per (provider_code_current, service_line)
        and one column per staff_group, or an alert div if data is missing.
    """
    if staffing_df is None or staffing_df.empty:
        return html.Div(
            "No staffing data available.",
            className="alert alert-warning",
        )

    required_cols = {"provider_code_current", "service_line", "staff_group", "staff"}
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
        and "provider_code_current" in patient_df.columns
    ):
        selected_providers = (
            patient_df["provider_code_current"].dropna().astype(str).unique().tolist()
        )

    staffing_filtered = staffing_df.copy()
    if selected_providers:
        staffing_filtered = staffing_filtered[
            staffing_filtered["provider_code_current"]
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
            ["provider_code_current", "service_line", "staff_group"], as_index=False
        )
        .agg(staff=("staff", "sum"))
        .pivot(
            index=["provider_code_current", "service_line"],
            columns="staff_group",
            values="staff",
        )
        .fillna(0)
        .reset_index()
    )

    # Keep columns stable-ish: identifiers first, then alphabetical staff groups
    id_cols = ["provider_code_current", "service_line"]
    non_index_cols = [c for c in pivot_df.columns if c not in id_cols]
    pivot_df = pivot_df[id_cols + sorted(non_index_cols, key=lambda x: str(x))]

    pivot_df = pivot_df.sort_values(["provider_code_current", "service_line"])

    # Render integers when possible
    for col in non_index_cols:
        try:
            pivot_df[col] = pivot_df[col].round(0).astype(int)
        except Exception:
            pivot_df[col] = pivot_df[col].round(0)

    return dash_table.DataTable(
        id="staffing-pivot-table",
        data=pivot_df.to_dict("records"),
        columns=[
            {
                "name": (
                    prettify_column_name(str(i))
                    if str(i) in {"provider_code_current", "service_line"}
                    else str(i)
                ),
                "id": str(i),
            }
            for i in pivot_df.columns
        ],
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 2},
        sort_action="native",
        style_table={
            "width": "100%",
            "minWidth": "100%",
            "overflowX": "auto",
            "maxHeight": "520px",
            "overflowY": "auto",
        },
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "minWidth": "120px",
            "maxWidth": "240px",
            "whiteSpace": "normal",
        },
        style_cell_conditional=[
            {
                "if": {"column_id": "service_line"},
                "minWidth": "180px",
                "maxWidth": "320px",
            }
        ],
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
    )


def create_demand_tab(df):
    """Create the Demand tab layout.

    Args:
        df: Filtered patient dataset.

    Returns:
        A Dash layout component containing the demand alert and patient summary table.
    """

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
    """Create the Capacity tab layout.

    Args:
        df: Filtered patient dataset (used to select provider(s)).

    Returns:
        A Dash layout component containing the capacity alert and staffing pivot table.
    """
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

    # Flask's debug reloader starts the app twice (parent + child). Disable it to
    # prevent duplicate startup logs while keeping debug mode features.
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)
