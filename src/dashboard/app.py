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

from forecasting.forecast import ForecastConfig, make_ets_forecast_frame
from forecasting.ets import EtsSpec, small_grid_search_aic_ets


# ---------------------------------------------------------------------
# Forecast Caching (keep dashboard interactions responsive)
# ---------------------------------------------------------------------
# Dash callbacks can fire frequently (legend toggles, filter changes). Keeping a
# small in-memory cache of selected model specs and forecast frames keyed by the
# input series signature.
#
# NOTE: If the design requiresETS to re-run on every callback (no caching), set this to False.
ETS_CACHE_ENABLED = False
_FORECAST_CACHE_MAX = 64
_SPEC_CACHE_MAX = 128
_forecast_frame_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_best_spec_cache: "OrderedDict[tuple, EtsSpec]" = OrderedDict()


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

    # Build a month-level slider index from patient period_end.
    if (
        data_handler.patient_df is not None
        and "period_end" in data_handler.patient_df.columns
    ):
        period_end = pd.to_datetime(
            data_handler.patient_df["period_end"], errors="coerce"
        )
        month_ends = (
            period_end.dropna()
            .dt.to_period("M")
            .dt.to_timestamp("M")
            .drop_duplicates()
            .sort_values()
        )
        slider_dates = [d.strftime("%Y-%m-%d") for d in month_ends.tolist()]
    else:
        slider_dates = []

    # Default range: from the first available month to the latest month
    default_start_idx = 0
    default_end_idx = max(0, len(slider_dates) - 1)

    # Slider marks (keep it readable): show April each year + endpoints
    marks: dict[int, str] = {}
    if slider_dates:
        dt = pd.to_datetime(pd.Series(slider_dates), errors="coerce")
        for idx, d in enumerate(dt.tolist()):
            if d is None or pd.isna(d):
                continue
            if idx in {0, default_end_idx} or d.month == 4:
                marks[idx] = d.strftime("%b %Y")

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
                                html.Div(
                                    [
                                        dcc.Store(
                                            id="date-slider-dates",
                                            data=slider_dates,
                                        ),
                                        html.Div(
                                            id="date-range-slider-label",
                                            className="text-muted small mb-1",
                                        ),
                                        dcc.RangeSlider(
                                            id="date-range-slider",
                                            min=0,
                                            max=max(0, len(slider_dates) - 1),
                                            step=1,
                                            value=[default_start_idx, default_end_idx],
                                            marks=marks,
                                            allowCross=False,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": False,
                                            },
                                        ),
                                    ]
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


def _monthly_series_from_filtered_df(df: pd.DataFrame, metric: str) -> pd.Series:
    """Build a month-end indexed series for a single metric from the filtered dataset."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df_sorted = df.sort_values("period_end")
    if "year_month" not in df_sorted.columns:
        # Fallback if upstream preprocessing ever changes
        df_sorted["year_month"] = (
            pd.to_datetime(df_sorted["period_end"]).dt.to_period("M").astype(str)
        )

    monthly = (
        df_sorted.groupby("year_month")
        .agg({metric: "sum"})
        .reset_index()
        .sort_values("year_month")
    )

    # Month-end DatetimeIndex for forecasting + plotting
    x = pd.PeriodIndex(monthly["year_month"].astype(str), freq="M").to_timestamp("M")
    y = pd.Series(
        pd.to_numeric(monthly[metric], errors="coerce").astype(float).values,
        index=pd.DatetimeIndex(x),
        name=metric,
    ).dropna()
    return y


def _metric_timeseries_figure(
    *,
    df: pd.DataFrame,
    metric: str,
    label: str,
    color: str,
    forecast_on: bool,
    monthly_ticks: bool = True,
) -> go.Figure:
    """Create a single-metric time series figure with optional ETS forecast."""
    y = _monthly_series_from_filtered_df(df, metric)
    fig = go.Figure()

    if y.empty:
        fig.update_layout(
            title=dict(text=label, x=0.5, xanchor="center"),
            template="plotly_white",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Actuals (line + markers)
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y.values,
            name=label,
            mode="lines",
            line=dict(width=3, shape="spline", color=color),
            hovertemplate=f"{label}: <b>%{{y:,}}</b><extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y.values,
            name=label,
            mode="markers",
            marker=dict(size=8, color=color),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if bool(forecast_on):
        try:
            if len(y) >= 12 and y.nunique() >= 2:
                # Use the filtered series signature as a cache key so model
                # selection runs only once per metric per filter state.
                #
                # Your dataset currently spans ~30 months, so we keep the search
                # space intentionally small to stay responsive.
                y_fit = y
                if len(y_fit) > 36:
                    y_fit = y_fit.iloc[-36:]

                sig = _series_signature(y_fit)

                # AIC-based spec selection (optionally cached)
                spec_key = ("ets_aic_v1", metric, sig)
                spec = (
                    _lru_get(_best_spec_cache, spec_key) if ETS_CACHE_ENABLED else None
                )
                if spec is None:
                    y_len = int(len(y_fit))
                    # Guardrails for short monthly series (~30 months): ETS with
                    # additive seasonality is usually OK, but keep it optional.
                    seasonal_opts = ("add", None) if y_len >= 24 else (None,)

                    spec = small_grid_search_aic_ets(
                        y_fit,
                        seasonal_period=12,
                        trend=("add", None),
                        seasonal=seasonal_opts,
                        damped_trend=(True, False),
                    )
                    if ETS_CACHE_ENABLED:
                        _lru_set(
                            _best_spec_cache, spec_key, spec, maxsize=_SPEC_CACHE_MAX
                        )

                # Include the selected spec in the forecast cache key to avoid
                # collisions if the selection settings change.
                fc_key = (
                    metric,
                    sig,
                    spec.trend,
                    spec.seasonal,
                    spec.seasonal_periods,
                    spec.damped_trend,
                    12,
                    0.95,
                    "ME",
                )
                future = (
                    _lru_get(_forecast_frame_cache, fc_key)
                    if ETS_CACHE_ENABLED
                    else None
                )
                if future is None:
                    frame = make_ets_forecast_frame(
                        y_fit,
                        config=ForecastConfig(
                            months_ahead=12,
                            conf_level=0.95,
                            freq="ME",
                            auto_select=False,
                        ),
                        spec=spec,
                    )
                    future = frame[frame["is_forecast"]].copy()
                    if ETS_CACHE_ENABLED:
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

                    # CI band
                    fig.add_trace(
                        go.Scatter(
                            x=future["period_end"],
                            y=future["yhat_upper"],
                            mode="lines",
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{label} (95% CI)",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=future["period_end"],
                            y=future["yhat_lower"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=band_color,
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{label} (95% CI)",
                        )
                    )

                    # Forecast dashed line joined to last actual point
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

                    fig.add_trace(
                        go.Scatter(
                            x=x_fc,
                            y=yhat_fc,
                            mode="lines",
                            line=dict(width=2, dash="dash", color=color),
                            hovertemplate=f"{label} (Forecast): <b>%{{y:,.0f}}</b><extra></extra>",
                            showlegend=False,
                            name=f"{label} (Forecast)",
                        )
                    )
            else:
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.99,
                    xanchor="left",
                    yanchor="top",
                    text="Forecast needs ≥12 months of history",
                    showarrow=False,
                    font=dict(size=12, color="rgba(0,0,0,0.65)"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                )
        except Exception as e:
            logger.warning(f"⚠️ Forecast skipped for '{metric}': {e}")
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                text=f"Forecast error: {type(e).__name__}",
                showarrow=False,
                font=dict(size=12, color="rgba(0,0,0,0.65)"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
            )

    fig.update_layout(
        title=dict(text=label, x=0.5, xanchor="center", font=dict(size=20)),
        xaxis_title="Period",
        yaxis_title="Count",
        hovermode="x unified",
        height=420,
        template="plotly_white",
        margin=dict(l=40, r=30, t=70, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=-30,
        ticks="outside",
        ticklen=6,
        tickformat="%b %Y",
        hoverformat="%b %Y",
    )
    if bool(monthly_ticks):
        fig.update_xaxes(dtick="M1")
    fig.update_yaxes(
        tickformat=",",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    return fig


def _overview_key_metrics_figure(
    *,
    df: pd.DataFrame,
    forecast_on: bool,
    visible_metrics: Optional[set[str]] = None,
) -> go.Figure:
    """Create the Overview multi-metric time series figure with optional ETS forecasts.

    If `visible_metrics` is provided, ETS forecasts are computed only for those
    metrics (legend-visible series). The same set also drives which traces are
    shown vs `legendonly` so legend state survives rerenders.
    """
    if df is None or df.empty:
        return go.Figure()

    df_sorted = df.sort_values("period_end")
    if "year_month" not in df_sorted.columns:
        df_sorted = df_sorted.copy()
        df_sorted["year_month"] = (
            pd.to_datetime(df_sorted["period_end"]).dt.to_period("M").astype(str)
        )

    monthly = (
        df_sorted.groupby("year_month")
        .agg(
            {
                "referrals": "sum",
                "waiters": "sum",
                "caseload": "sum",
                "discharges_with_clock_stop": "sum",
                "total_contacts": "sum",
                "clock_stop_actuals": "sum",
            }
        )
        .reset_index()
    )

    try:
        monthly_x = pd.PeriodIndex(
            monthly["year_month"].astype(str), freq="M"
        ).to_timestamp("M")
    except Exception:
        monthly_x = monthly["year_month"].astype(str)

    fig = go.Figure()

    palette = px.colors.qualitative.Plotly
    series = [
        ("Referrals", "referrals"),
        ("Waiters", "waiters"),
        ("Caseload", "caseload"),
        ("Contacts", "total_contacts"),
        ("Discharges", "discharges_with_clock_stop"),
        ("Clock Stop Actuals", "clock_stop_actuals"),
    ]

    default_visible = {"referrals", "clock_stop_actuals"}
    effective_visible = (
        visible_metrics if visible_metrics is not None else default_visible
    )

    # Actuals (line + marker legend entry per metric)
    for i, (label, metric) in enumerate(series):
        if metric not in monthly.columns:
            continue

        color = palette[i % len(palette)]
        is_visible = metric in effective_visible
        trace_visible = True if is_visible else "legendonly"

        # Line trace
        fig.add_trace(
            go.Scatter(
                x=monthly_x,
                y=monthly[metric],
                name=label,
                legendgroup=metric,
                showlegend=False,
                mode="lines",
                visible=trace_visible,
                line=dict(width=3, shape="spline", color=color),
                hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
                meta={"metric": metric},
            )
        )

        # Marker trace (legend entry)
        fig.add_trace(
            go.Scatter(
                x=monthly_x,
                y=monthly[metric],
                name=label,
                legendgroup=metric,
                showlegend=True,
                mode="markers",
                visible=trace_visible,
                marker=dict(size=9, color=color),
                hoverinfo="skip",
                meta={"metric": metric},
            )
        )

    # Optional forecasts (runs only when this visual's toggle is on)
    if bool(forecast_on):
        skipped = 0
        added_any = False
        forecast_metrics = effective_visible

        for i, (label, metric) in enumerate(series):
            if metric not in df_sorted.columns:
                continue
            if metric not in forecast_metrics:
                continue

            y = _monthly_series_from_filtered_df(df_sorted, metric)
            if len(y) < 12 or y.nunique() < 2:
                skipped += 1
                continue

            try:
                y_fit = y.iloc[-36:] if len(y) > 36 else y
                sig = _series_signature(y_fit)

                spec_key = ("ets_aic_v1", metric, sig)
                spec = (
                    _lru_get(_best_spec_cache, spec_key) if ETS_CACHE_ENABLED else None
                )
                if spec is None:
                    y_len = int(len(y_fit))
                    seasonal_opts = ("add", None) if y_len >= 24 else (None,)
                    spec = small_grid_search_aic_ets(
                        y_fit,
                        seasonal_period=12,
                        trend=("add", None),
                        seasonal=seasonal_opts,
                        damped_trend=(True, False),
                    )
                    if ETS_CACHE_ENABLED:
                        _lru_set(
                            _best_spec_cache, spec_key, spec, maxsize=_SPEC_CACHE_MAX
                        )

                fc_key = (
                    "overview_key_metrics",
                    metric,
                    sig,
                    spec.trend,
                    spec.seasonal,
                    spec.seasonal_periods,
                    spec.damped_trend,
                    12,
                    0.95,
                    "ME",
                )
                future = (
                    _lru_get(_forecast_frame_cache, fc_key)
                    if ETS_CACHE_ENABLED
                    else None
                )
                if future is None:
                    frame = make_ets_forecast_frame(
                        y_fit,
                        config=ForecastConfig(
                            months_ahead=12,
                            conf_level=0.95,
                            freq="ME",
                            auto_select=False,
                        ),
                        spec=spec,
                    )
                    future = frame[frame["is_forecast"]].copy()
                    if ETS_CACHE_ENABLED:
                        _lru_set(
                            _forecast_frame_cache,
                            fc_key,
                            future,
                            maxsize=_FORECAST_CACHE_MAX,
                        )
                else:
                    future = future.copy()

                if future.empty:
                    skipped += 1
                    continue

                color = palette[i % len(palette)]
                band_color = _rgba(color, 0.10)

                # CI band
                fig.add_trace(
                    go.Scatter(
                        x=future["period_end"],
                        y=future["yhat_upper"],
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{label} (95% CI)",
                        legendgroup=metric,
                        meta={"metric": metric},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=future["period_end"],
                        y=future["yhat_lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=band_color,
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{label} (95% CI)",
                        legendgroup=metric,
                        meta={"metric": metric},
                    )
                )

                # Forecast dashed line joined to last actual point
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

                fig.add_trace(
                    go.Scatter(
                        x=x_fc,
                        y=yhat_fc,
                        mode="lines",
                        line=dict(width=2, dash="dash", color=color),
                        hovertemplate=f"{label} (Forecast): <b>%{{y:,.0f}}</b><extra></extra>",
                        showlegend=False,
                        name=f"{label} (Forecast)",
                        legendgroup=metric,
                        meta={"metric": metric},
                    )
                )
                added_any = True
            except Exception as e:
                logger.warning(f"⚠️ Overview forecast skipped for '{metric}': {e}")
                skipped += 1

        if not added_any:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                text="Forecast needs ≥12 months of history",
                showarrow=False,
                font=dict(size=12, color="rgba(0,0,0,0.65)"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
            )
        elif skipped:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                text=f"Forecast skipped for {skipped} series (insufficient history)",
                showarrow=False,
                font=dict(size=12, color="rgba(0,0,0,0.65)"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
            )

    fig.update_layout(
        title=dict(
            text="Key Metrics Over Time",
            x=0.5,
            xanchor="center",
            font=dict(size=22),
        ),
        xaxis_title="Period",
        yaxis_title="Count",
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

    return fig


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
        State("date-range-slider", "value"),
        State("date-slider-dates", "data"),
        State("service-line-dropdown", "value"),
    ],
)
def filter_data(n_clicks, slider_range, slider_dates, service_lines):
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

    # Apply date filter from slider indices
    if (
        slider_dates
        and isinstance(slider_range, (list, tuple))
        and len(slider_range) == 2
    ):
        try:
            start_idx = int(slider_range[0])
            end_idx = int(slider_range[1])
            start_idx = max(0, min(start_idx, len(slider_dates) - 1))
            end_idx = max(0, min(end_idx, len(slider_dates) - 1))
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            start_date = slider_dates[start_idx]
            end_date = slider_dates[end_idx]
            df = data_handler.filter_by_date_range(df, start_date, end_date)
        except Exception:
            pass

    if service_lines:
        # Users select service labels, but we filter by provider code only.
        selected_providers = sorted({item.split("|")[0] for item in service_lines})
        df = df[df["provider_code_current"].astype(str).isin(selected_providers)]

    # Store as JSON
    return df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("date-range-slider-label", "children"),
    [Input("date-range-slider", "value")],
    [State("date-slider-dates", "data")],
)
def update_date_slider_label(slider_range, slider_dates):
    if (
        not slider_dates
        or not isinstance(slider_range, (list, tuple))
        or len(slider_range) != 2
    ):
        return "Date range: N/A"
    try:
        start_idx = int(slider_range[0])
        end_idx = int(slider_range[1])
        start_idx = max(0, min(start_idx, len(slider_dates) - 1))
        end_idx = max(0, min(end_idx, len(slider_dates) - 1))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        start_dt = pd.to_datetime(slider_dates[start_idx], errors="coerce")
        end_dt = pd.to_datetime(slider_dates[end_idx], errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            return "Date range: N/A"
        return f"Date range: {start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}"
    except Exception:
        return "Date range: N/A"


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
    total_discharges = int(df["discharges_with_clock_stop"].sum())

    # Capacity metric (staff is not time-indexed in the current dataset)
    total_staff_value: Optional[int] = None
    if data_handler is not None and data_handler.staffing_df is not None:
        staffing_filtered, staffing_error = _capacity_filter_staffing(
            df, data_handler.staffing_df
        )
        if staffing_filtered is not None and staffing_error is None:
            try:
                total_staff_value = int(round(float(staffing_filtered["staff"].sum())))
            except Exception:
                total_staff_value = None

    total_staff_display = (
        f"{total_staff_value:,}" if total_staff_value is not None else "N/A"
    )

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
            dbc.Col(
                create_metric_card(
                    "Total Staff",
                    total_staff_display,
                    "🧑‍⚕️",
                    "dark",
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


def create_overview_tab(df):
    """Create the Overview tab layout.

    Args:
        df: Filtered patient dataset.

    Returns:
        A Dash layout component containing the overview charts.
    """

    df_sorted = df.sort_values("period_end")
    if "year_month" not in df_sorted.columns:
        df_sorted = df_sorted.copy()
        df_sorted["year_month"] = (
            pd.to_datetime(df_sorted["period_end"]).dt.to_period("M").astype(str)
        )

    default_visible_metrics = ["referrals", "clock_stop_actuals"]

    # Key metrics combined time-series (legend selectable) + local forecast toggle
    key_metrics_chart = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5("Key Metrics Over Time", className="mb-0"),
                            width=True,
                        ),
                        dbc.Col(
                            dbc.Switch(
                                id="overview-keymetrics-forecast-toggle",
                                label="Forecast (ETS, 95% CI)",
                                value=False,
                            ),
                            width="auto",
                        ),
                    ],
                    align="center",
                )
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="overview-keymetrics-timeseries",
                    figure=_overview_key_metrics_figure(
                        df=df_sorted,
                        forecast_on=False,
                        visible_metrics=set(default_visible_metrics),
                    ),
                ),
                className="p-2",
            ),
        ],
        className="mb-4 shadow-sm",
    )

    # -----------------------------
    # Capacity: Staff vs Caseload (latest)
    # -----------------------------
    staff_vs_caseload_card = None
    if data_handler is not None:
        staffing_filtered, staffing_error = _capacity_filter_staffing(
            df, data_handler.staffing_df
        )
        if staffing_filtered is not None and staffing_error is None:
            fig_staff_vs_case = _capacity_staff_vs_caseload_latest_figure(
                df, staffing_filtered
            )
            if fig_staff_vs_case is not None:
                staff_vs_caseload_card = dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Staff vs Caseload (Latest)", className="mb-0")
                        ),
                        dbc.CardBody(
                            dcc.Graph(
                                id="overview-staff-vs-caseload",
                                figure=fig_staff_vs_case,
                            ),
                            className="p-2",
                        ),
                    ],
                    className="mb-4 shadow-sm",
                )

    # -----------------------------
    # 18-week waiters split (stacked bar)
    # -----------------------------
    fig_18wk = _waiters_18wk_breakdown_figure(df_sorted, monthly_ticks=True)
    waiters_chart = dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    "Waiting List Breakdown (Under vs Over 18 Weeks)",
                    className="mb-0",
                )
            ),
            dbc.CardBody(
                dcc.Graph(id="overview-waiters-breakdown", figure=fig_18wk),
                className="p-2",
            ),
        ],
        className="mb-4 shadow-sm",
    )

    # -----------------------------
    # Layout
    # -----------------------------
    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("📈 Overview", className="alert-heading"),
                    html.P(
                        "Use the filters above to change the view. In charts, you can click legend items to show/hide metrics. "
                        "Forecast toggles apply per chart (Overview and Demand Analysis)."
                    ),
                    html.P(
                        "Forecasting note: ETS / exponential smoothing (Holt-Winters) works best with enough history to learn seasonality. "
                        "For monthly data, adjust the date selector to 2-3 years (~24-36 points) when forecasting."
                    ),
                ],
                color="info",
                className="mt-4",
            ),
            dcc.Store(
                id="overview-keymetrics-visible-metrics",
                data=default_visible_metrics,
            ),
            key_metrics_chart,
            staff_vs_caseload_card,
            waiters_chart,
        ]
    )


def _waiters_18wk_breakdown_figure(
    df_sorted: pd.DataFrame,
    *,
    monthly_ticks: bool = False,
) -> go.Figure:
    """Build the stacked bar chart for under/over 18-week waiters by month."""
    base = df_sorted.copy()
    if "year_month" not in base.columns:
        base["year_month"] = (
            pd.to_datetime(base["period_end"]).dt.to_period("M").astype(str)
        )

    waiters_18wk = base.groupby("year_month", as_index=False).agg(
        waiters_under_18_weeks=("waiters_under_18_weeks", "sum"),
        waiters_over_18_weeks=("waiters_over_18_weeks", "sum"),
    )

    # Use a real date x-axis so we can control tick density.
    x_dates = pd.PeriodIndex(waiters_18wk["year_month"].astype(str), freq="M").to_timestamp(
        "M"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=waiters_18wk["waiters_under_18_weeks"],
            name="Under 18 Weeks",
            marker_color="#636EFA",
            hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=waiters_18wk["waiters_over_18_weeks"],
            name="18+ Weeks",
            marker_color="#EF553B",
            hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=None,
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
    fig.update_xaxes(
        showgrid=False,
        tickangle=-30,
        ticks="outside",
        ticklen=6,
        tickformat="%b %Y",
        hoverformat="%b %Y",
    )
    if bool(monthly_ticks):
        fig.update_xaxes(dtick="M1")
    fig.update_yaxes(
        tickformat=",",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    return fig


@app.callback(
    Output("overview-keymetrics-timeseries", "figure"),
    [
        Input("filtered-data-store", "data"),
        Input("overview-keymetrics-forecast-toggle", "value"),
        Input("overview-keymetrics-visible-metrics", "data"),
    ],
)
def update_overview_keymetrics_timeseries(data_json, forecast_on, visible_metrics):
    if data_json is None:
        return go.Figure()
    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"])
    visible_set = set(visible_metrics or ["referrals", "clock_stop_actuals"])
    return _overview_key_metrics_figure(
        df=df,
        forecast_on=bool(forecast_on),
        visible_metrics=visible_set,
    )


@app.callback(
    Output("overview-keymetrics-visible-metrics", "data"),
    Input("overview-keymetrics-timeseries", "restyleData"),
    State("overview-keymetrics-timeseries", "figure"),
    State("overview-keymetrics-visible-metrics", "data"),
    prevent_initial_call=True,
)
def update_overview_keymetrics_visible_metrics(restyle_data, fig, current_visible):
    """Maintain a list of legend-visible metric keys for the Overview chart.

    Plotly legend interactions emit `restyleData` with trace indices and new
    `visible` values. We store visibility at the *metric* level using trace
    `meta.metric` so we can recompute forecasts only for currently visible
    series.
    """
    if not restyle_data or not fig:
        return dash.no_update

    if not isinstance(restyle_data, (list, tuple)) or len(restyle_data) != 2:
        return dash.no_update

    updates, trace_indices = restyle_data
    if not isinstance(updates, dict) or "visible" not in updates:
        return dash.no_update

    visible_update = updates.get("visible")
    if not isinstance(trace_indices, list):
        trace_indices = [trace_indices]

    visible_set = set(current_visible or [])

    for pos, trace_idx in enumerate(trace_indices):
        try:
            trace = (fig.get("data") or [])[int(trace_idx)]
            metric = (trace.get("meta") or {}).get("metric")
        except Exception:
            metric = None

        if not metric:
            continue

        if isinstance(visible_update, list) and pos < len(visible_update):
            new_vis = visible_update[pos]
        else:
            new_vis = visible_update

        if new_vis in ("legendonly", False):
            visible_set.discard(metric)
        else:
            visible_set.add(metric)

    return sorted(visible_set)


@app.callback(
    Output("referrals-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("referrals-forecast-toggle", "value")],
)
def update_referrals_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"])
    return _metric_timeseries_figure(
        df=df,
        metric="referrals",
        label="Referrals",
        color="#636EFA",
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("waiters-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("waiters-forecast-toggle", "value")],
)
def update_waiters_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"])
    return _metric_timeseries_figure(
        df=df,
        metric="waiters",
        label="Waiters",
        color="#EF553B",
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("caseload-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("caseload-forecast-toggle", "value")],
)
def update_caseload_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"])
    return _metric_timeseries_figure(
        df=df,
        metric="caseload",
        label="Caseload",
        color="#00CC96",
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("contacts-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("contacts-forecast-toggle", "value")],
)
def update_contacts_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"])
    return _metric_timeseries_figure(
        df=df,
        metric="total_contacts",
        label="Contacts",
        color="#AB63FA",
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("discharges-timeseries", "figure"),
    [
        Input("filtered-data-store", "data"),
        Input("discharges-forecast-toggle", "value"),
    ],
)
def update_discharges_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"])
    return _metric_timeseries_figure(
        df=df,
        metric="discharges_with_clock_stop",
        label="Discharges",
        color="#FFA15A",
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


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
        .round(0)
        .sort_values(["provider_code_current", "service_line"])
    )

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


def _capacity_filter_staffing(
    patient_df: pd.DataFrame,
    staffing_df: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[html.Div]]:
    """Filter staffing dataset to the providers currently selected in patient_df."""

    if staffing_df is None or staffing_df.empty:
        return None, html.Div(
            "No staffing data available.",
            className="alert alert-warning",
        )

    required_cols = {"provider_code_current", "service_line", "staff_group", "staff"}
    if not required_cols.issubset(set(staffing_df.columns)):
        missing = sorted(required_cols - set(staffing_df.columns))
        return None, html.Div(
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
    staffing_filtered["provider_code_current"] = staffing_filtered[
        "provider_code_current"
    ].astype(str)

    if selected_providers:
        staffing_filtered = staffing_filtered[
            staffing_filtered["provider_code_current"].isin(
                [str(p) for p in selected_providers]
            )
        ]

    staffing_filtered["staff"] = pd.to_numeric(
        staffing_filtered["staff"], errors="coerce"
    ).fillna(0)

    if staffing_filtered.empty:
        return None, html.Div(
            "No staffing rows match the selected provider(s).",
            className="alert alert-info",
        )

    return staffing_filtered, None


def _capacity_style_figure(fig: go.Figure) -> go.Figure:
    """Apply a consistent, dashboard-friendly Plotly style."""
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_yaxes(
        tickformat=",",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    return fig


def _capacity_staff_mix_by_group_figure(staffing_filtered: pd.DataFrame) -> go.Figure:
    by_group = (
        staffing_filtered.groupby("staff_group", as_index=False)
        .agg(staff=("staff", "sum"))
        .sort_values("staff", ascending=True)
    )
    fig = px.bar(
        by_group,
        x="staff",
        y="staff_group",
        orientation="h",
        title="Total Staff by Staff Group",
        labels={"staff": "Staff (count)", "staff_group": "Staff group"},
    )
    fig.update_traces(hovertemplate="%{y}<br>Staff: %{x:,}<extra></extra>")
    return _capacity_style_figure(fig)


def _capacity_staff_vs_caseload_latest_figure(
    patient_df: pd.DataFrame,
    staffing_filtered: pd.DataFrame,
) -> Optional[go.Figure]:
    if patient_df is None or patient_df.empty or "caseload" not in patient_df.columns:
        return None

    required_patient_cols = {
        "provider_code_current",
        "service_line",
        "period_end",
        "caseload",
    }
    if not required_patient_cols.issubset(set(patient_df.columns)):
        return None

    df = patient_df.copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    latest_period_end = df["period_end"].max()
    if pd.isna(latest_period_end):
        return None

    latest = df[df["period_end"] == latest_period_end].copy()
    latest["provider_code_current"] = latest["provider_code_current"].astype(str)

    demand_latest = latest.groupby(
        ["provider_code_current", "service_line"], as_index=False
    ).agg(caseload=("caseload", "sum"))
    demand_latest["caseload"] = pd.to_numeric(
        demand_latest["caseload"], errors="coerce"
    ).fillna(0)

    staff_totals = staffing_filtered.groupby(
        ["provider_code_current", "service_line"], as_index=False
    ).agg(staff=("staff", "sum"))

    merged = demand_latest.merge(
        staff_totals,
        on=["provider_code_current", "service_line"],
        how="inner",
    )

    if merged.empty:
        return None

    merged["service_label"] = (
        merged["provider_code_current"].astype(str)
        + " - "
        + merged["service_line"].astype(str)
    )
    merged["staff_per_100_caseload"] = merged.apply(
        lambda r: (
            (r["staff"] / r["caseload"] * 100) if float(r["caseload"]) > 0 else None
        ),
        axis=1,
    )

    # Keep a clean analytic frame for the trendline
    plot_df = merged.copy()
    plot_df["caseload"] = pd.to_numeric(plot_df["caseload"], errors="coerce")
    plot_df["staff"] = pd.to_numeric(plot_df["staff"], errors="coerce")
    plot_df = plot_df.dropna(subset=["caseload", "staff"])

    fig = px.scatter(
        plot_df,
        x="caseload",
        y="staff",
        color="staff_per_100_caseload",
        size="caseload",
        size_max=40,
        trendline="ols",
        hover_name="service_label",
        hover_data={
            "caseload": ":,",
            "staff": ":,",
            "staff_per_100_caseload": ":.2f",
            "service_label": False,
        },
        title=f"Staff vs Caseload (Latest period: {latest_period_end.strftime('%b %Y')})",
        labels={
            "caseload": "Caseload (latest)",
            "staff": "Staff (total)",
            "staff_per_100_caseload": "Staff per 100 caseload",
        },
    )
    fig.update_traces(
        marker=dict(opacity=0.75, line=dict(width=1, color="rgba(0,0,0,0.15)"))
    )
    fig.update_xaxes(showgrid=False, tickformat=",")
    fig.update_yaxes(tickformat=",")
    return _capacity_style_figure(fig)


def create_demand_tab(df):
    """Create the Demand tab layout.

    Args:
        df: Filtered patient dataset.

    Returns:
        A Dash layout component containing the demand alert and patient summary table.
    """

    df_sorted = df.sort_values("period_end")
    if "year_month" not in df_sorted.columns:
        df_sorted = df_sorted.copy()
        df_sorted["year_month"] = (
            pd.to_datetime(df_sorted["period_end"]).dt.to_period("M").astype(str)
        )

    fig_18wk = _waiters_18wk_breakdown_figure(df_sorted, monthly_ticks=False)
    waiters_breakdown_card = dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    "Waiting List Breakdown (Under vs Over 18 Weeks)",
                    className="mb-0",
                )
            ),
            dbc.CardBody(
                dcc.Graph(id="demand-waiters-breakdown", figure=fig_18wk),
                className="p-2",
            ),
        ],
        className="mb-4 shadow-sm",
    )

    summary_table = create_service_line_summary_table(df)

    def metric_card(title: str, graph_id: str, toggle_id: str):
        return dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(html.H5(title, className="mb-0"), width=True),
                            dbc.Col(
                                dbc.Switch(
                                    id=toggle_id,
                                    label="Forecast (ETS, 95% CI)",
                                    value=False,
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                    )
                ),
                dbc.CardBody(
                    dcc.Graph(id=graph_id, figure=go.Figure()),
                    className="p-2",
                ),
            ],
            className="mb-4 shadow-sm",
        )

    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("👥 Demand Analysis", className="alert-heading"),
                    html.P(
                        "Use the filters above to change the view. Each chart has its own forecast toggle (ETS with 95% CI)."
                    ),
                    html.P(
                        "Forecasting note: ETS / exponential smoothing (Holt-Winters) works best with enough history to learn seasonality. "
                        "For monthly data, adjust the date selector to 2-3 years (~24-36 points) when forecasting."
                    ),
                ],
                color="info",
                className="mt-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        metric_card(
                            "Referrals",
                            "referrals-timeseries",
                            "referrals-forecast-toggle",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        metric_card(
                            "Waiters", "waiters-timeseries", "waiters-forecast-toggle"
                        ),
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        metric_card(
                            "Caseload",
                            "caseload-timeseries",
                            "caseload-forecast-toggle",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        metric_card(
                            "Contacts",
                            "contacts-timeseries",
                            "contacts-forecast-toggle",
                        ),
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        metric_card(
                            "Discharges",
                            "discharges-timeseries",
                            "discharges-forecast-toggle",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        waiters_breakdown_card,
                        width=6,
                    ),
                ]
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

    staffing_filtered, staffing_error = _capacity_filter_staffing(
        df, data_handler.staffing_df
    )

    staffing_table = create_staffing_pivot_table(df, data_handler.staffing_df)

    capacity_graphs = []
    if staffing_filtered is not None and staffing_error is None:
        fig_by_group = _capacity_staff_mix_by_group_figure(staffing_filtered)
        fig_staff_vs_case = _capacity_staff_vs_caseload_latest_figure(
            df, staffing_filtered
        )

        def graph_card(title: str, fig: go.Figure, graph_id: str):
            return dbc.Card(
                [
                    dbc.CardHeader(html.H5(title, className="mb-0")),
                    dbc.CardBody(dcc.Graph(id=graph_id, figure=fig), className="p-2"),
                ],
                className="mb-4 shadow-sm",
            )

        capacity_graphs.append(
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Staff by Group",
                            fig_by_group,
                            "capacity-staff-by-group",
                        ),
                        width=12,
                    ),
                ]
            )
        )

        if fig_staff_vs_case is not None:
            capacity_graphs.append(
                dbc.Row(
                    [
                        dbc.Col(
                            graph_card(
                                "Staff vs Caseload",
                                fig_staff_vs_case,
                                "capacity-staff-vs-caseload",
                            ),
                            width=12,
                        )
                    ]
                )
            )

    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("💼 Capacity Analysis", className="alert-heading"),
                    html.P(
                        "Capacity visuals are based on the currently selected provider(s). "
                        "Use the filters above to change the view."
                    ),
                    staffing_error if staffing_error is not None else None,
                ],
                color="info",
                className="mt-4",
            ),
            *capacity_graphs,
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
