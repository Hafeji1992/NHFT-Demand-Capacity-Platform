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
import numpy as np

# Ensure the dashboard folder is importable regardless of how the file is executed.
# This supports launches like VS Code "Run Python File", `python src/dashboard/app.py`,
# and `runpy.run_path(...)` where the script directory may not be on sys.path.
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

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
# If the design requiresETS to re-run on every callback (no caching), set this to False.
ETS_CACHE_ENABLED = False
_FORECAST_CACHE_MAX = 64
_SPEC_CACHE_MAX = 128
_forecast_frame_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_best_spec_cache: "OrderedDict[tuple, EtsSpec]" = OrderedDict()


# Forecast cache helper used to keep chart callbacks responsive.
def _lru_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


# Forecast cache helper used to keep chart callbacks responsive.
def _lru_set(cache: OrderedDict, key, value, *, maxsize: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(maxsize):
        cache.popitem(last=False)


# Forecast cache helper used to keep chart callbacks responsive.
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
import os
from flask import send_from_directory

# --- NHS Design System Colour Palette ---
NHS_BLUE = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_BRIGHT_BLUE = "#0072CE"
NHS_LIGHT_BLUE = "#41B6E6"
NHS_AQUA_GREEN = "#00A499"
NHS_GREEN = "#007F3B"
NHS_LIGHT_GREEN = "#78BE20"
NHS_YELLOW = "#FFB81C"
NHS_ORANGE = "#ED8B00"
NHS_RED = "#DA291C"
NHS_DARK_RED = "#8A1538"
NHS_PINK = "#AE2573"
NHS_PURPLE = "#330072"
NHS_BLACK = "#231F20"
NHS_DARK_GREY = "#425563"
NHS_MID_GREY = "#768692"
NHS_PALE_GREY = "#E8EDEE"
NHS_WHITE = "#FFFFFF"

# Plotly chart colour cycle using NHS palette
NHS_CHART_COLOURS = [
    NHS_BLUE,
    NHS_RED,
    NHS_AQUA_GREEN,
    NHS_ORANGE,
    NHS_PURPLE,
    NHS_LIGHT_BLUE,
    NHS_GREEN,
    NHS_PINK,
    NHS_YELLOW,
    NHS_DARK_BLUE,
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="NHFT Demand & Capacity Dashboard",
    suppress_callback_exceptions=True,
)

# Configure Flask to serve images folder
images_dir = os.path.join(os.path.dirname(__file__), "../../images")


@app.server.route("/images/<filename>")
def serve_images(filename):
    return send_from_directory(images_dir, filename)


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
# Convert technical column names into user-friendly labels.
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


# Format raw values into display-ready KPI text.
def _format_provider_code_current(series: pd.Series) -> pd.Series:
    """Normalise provider codes to a 3-digit string (e.g. 6 -> '006').

    Important:
        `pd.read_json(..., orient='split')` will often coerce numeric-like strings
        such as "006" into integers (6), losing leading zeros. This helper
        re-applies canonical formatting after any JSON roundtrip.
    """

    if series is None:
        return series

    s = series.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)

    is_digits = s.str.fullmatch(r"\d+", na=False)
    within_3 = s.str.len().fillna(0).astype(int) <= 3
    to_pad = is_digits & within_3
    return s.mask(to_pad, s.str.zfill(3))


# Read and normalize data from app state storage.
def _read_filtered_store_frame(data_json: str) -> pd.DataFrame:
    """Read the filtered patient frame from dcc.Store and normalise key dtypes."""

    df = pd.read_json(StringIO(data_json), orient="split")

    if "provider_code_current" in df.columns:
        df["provider_code_current"] = _format_provider_code_current(
            df["provider_code_current"]
        )

    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    return df


# Build and return a dashboard layout section or component.
# Map logical colour names to NHS CSS modifier classes.
_NHS_METRIC_CARD_CLASS = {
    "primary": "",
    "success": "nhs-metric-green",
    "warning": "nhs-metric-yellow",
    "info": "nhs-metric-aqua",
    "danger": "nhs-metric-red",
    "secondary": "nhs-metric-dark",
    "dark": "nhs-metric-purple",
}


def create_metric_card(
    title: str,
    value: str,
    icon: str = "📊",
    color: str = "primary",
    description: str = None,
):
    """Create a small KPI/metric card for the dashboard.

    Args:
        title: Card title label.
        value: Pre-formatted value to display (e.g., "1,234").
        icon: Emoji/icon prefix for the title.
        color: Bootstrap theme colour name (e.g., "primary", "warning").
        description: Optional description text to display below the value.

    Returns:
        A Dash Bootstrap Components Card.
    """
    variant_cls = _NHS_METRIC_CARD_CLASS.get(color, "")
    card_content = [
        html.H4(
            [html.Span(icon, className="me-2"), title],
            className="card-title",
        ),
        html.H2(
            value,
            className="metric-value",
        ),
    ]

    if description:
        card_content.append(
            html.P(
                description,
                className="metric-description",
            )
        )

    return dbc.Card(
        [dbc.CardBody(card_content)],
        className=f"nhs-metric-card mb-3 {variant_cls}".strip(),
    )


# Internal helper for shared dashboard logic.
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


# Format raw values into display-ready KPI text.
def _format_int_card_value(value: Optional[float]) -> str:
    """Format a numeric KPI value for display, or return 'N/A'."""

    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "N/A"


# Format raw values into display-ready KPI text.
def _format_ratio_card_value(value: Optional[float], *, decimals: int = 3) -> str:
    """Format a ratio KPI value for display, or return 'N/A'."""

    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    try:
        x = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(x):
        return "N/A"

    d = max(0, int(decimals))
    return f"{x:.{d}f}"


# Format raw values into display-ready KPI text.
def _format_percent_card_value(value: Optional[float], *, decimals: int = 1) -> str:
    """Format a proportion (0..1) as a percentage for KPI display, or 'N/A'."""

    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    try:
        x = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(x):
        return "N/A"

    d = max(0, int(decimals))
    return f"{x * 100:.{d}f}%"


# Format raw values into display-ready KPI text.
def _format_signed_int_card_value(value: Optional[float]) -> str:
    """Format a signed integer KPI value for display, or return 'N/A'."""

    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    try:
        x = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(x):
        return "N/A"

    n = int(round(x))
    if n > 0:
        return f"+{n:,}"
    return f"{n:,}"


# Compute a derived KPI from the current filtered dataset.
def _compute_net_caseload_flow_latest(df: pd.DataFrame) -> Optional[float]:
    """Compute latest-month net flow = inflow - outflow.

    Inflow:
      - referrals (sum)
    Outflow (first available):
      - discharges_with_clock_stop + discharges_no_clock_stop (if both exist)
      - discharges_with_clock_stop (if only this exists)
      - clock_stop_actuals (fallback if discharge fields are absent)
    """

    if df is None or df.empty:
        return None

    base = df.copy()
    if "period_end" not in base.columns:
        return None

    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base = base.dropna(subset=["period_end"])
    if base.empty:
        return None

    base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")
    latest_period_end = base["period_end"].max()
    if pd.isna(latest_period_end):
        return None

    if "referrals" not in base.columns:
        return None

    latest = base[base["period_end"] == latest_period_end].copy()
    if latest.empty:
        return None

    inflow = pd.to_numeric(latest["referrals"], errors="coerce").sum(skipna=True)

    has_with = "discharges_with_clock_stop" in latest.columns
    has_no = "discharges_no_clock_stop" in latest.columns
    if has_with and has_no:
        outflow = pd.to_numeric(
            latest["discharges_with_clock_stop"], errors="coerce"
        ).sum(skipna=True) + pd.to_numeric(
            latest["discharges_no_clock_stop"], errors="coerce"
        ).sum(
            skipna=True
        )
    elif has_with:
        outflow = pd.to_numeric(
            latest["discharges_with_clock_stop"], errors="coerce"
        ).sum(skipna=True)
    elif "clock_stop_actuals" in latest.columns:
        outflow = pd.to_numeric(latest["clock_stop_actuals"], errors="coerce").sum(
            skipna=True
        )
    else:
        return None

    if not np.isfinite(inflow) or not np.isfinite(outflow):
        return None

    return float(inflow) - float(outflow)


# Compute a derived KPI from the current filtered dataset.
def _compute_caseload_throughput_rate_latest(df: pd.DataFrame) -> Optional[float]:
    """Compute latest-month caseload throughput rate = outflow / caseload.

    Interprets outflow in the same way as Net Caseload Flow:
      - discharges_with_clock_stop + discharges_no_clock_stop (preferred)
      - discharges_with_clock_stop
      - clock_stop_actuals (fallback)
    """

    if df is None or df.empty:
        return None

    base = df.copy()
    if "period_end" not in base.columns:
        return None

    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base = base.dropna(subset=["period_end"])
    if base.empty:
        return None

    base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")
    latest_period_end = base["period_end"].max()
    if pd.isna(latest_period_end):
        return None

    if "caseload" not in base.columns:
        return None

    latest = base[base["period_end"] == latest_period_end].copy()
    if latest.empty:
        return None

    caseload = pd.to_numeric(latest["caseload"], errors="coerce").sum(skipna=True)
    if not np.isfinite(caseload) or float(caseload) <= 0:
        return None

    has_with = "discharges_with_clock_stop" in latest.columns
    has_no = "discharges_no_clock_stop" in latest.columns
    if has_with and has_no:
        outflow = pd.to_numeric(
            latest["discharges_with_clock_stop"], errors="coerce"
        ).sum(skipna=True) + pd.to_numeric(
            latest["discharges_no_clock_stop"], errors="coerce"
        ).sum(
            skipna=True
        )
    elif has_with:
        outflow = pd.to_numeric(
            latest["discharges_with_clock_stop"], errors="coerce"
        ).sum(skipna=True)
    elif "clock_stop_actuals" in latest.columns:
        outflow = pd.to_numeric(latest["clock_stop_actuals"], errors="coerce").sum(
            skipna=True
        )
    else:
        return None

    if not np.isfinite(outflow) or float(outflow) < 0:
        return None

    return float(outflow) / float(caseload)


# Compute a derived KPI from the current filtered dataset.
def _compute_clearance_time_weeks_latest(
    df: pd.DataFrame,
    *,
    weeks_in_month: float = 4.3,
    waiters_col: str = "waiters",
    first_contacts_col: str = "clock_stop_actuals",
) -> Optional[float]:
    """Compute the latest-month waiting list clearance time in weeks.

    Purpose:
        Estimate how many weeks it would take to clear the current waiting list
        at the current (latest-month) rate of first contacts.

    Formula:
        clearance_weeks = Waiters / (LatestMonthFirstContacts / 4.3)
    """

    if df is None or df.empty:
        return None

    base = df.copy()
    if "period_end" not in base.columns:
        return None

    if waiters_col not in base.columns or first_contacts_col not in base.columns:
        return None

    try:
        wim = float(weeks_in_month)
    except Exception:
        wim = 4.3
    if not np.isfinite(wim) or wim <= 0:
        return None

    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base = base.dropna(subset=["period_end"])
    if base.empty:
        return None

    base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")
    latest_period_end = base["period_end"].max()
    if pd.isna(latest_period_end):
        return None

    latest = base[base["period_end"] == latest_period_end].copy()
    if latest.empty:
        return None

    waiters = pd.to_numeric(latest[waiters_col], errors="coerce").sum(skipna=True)
    first_contacts = pd.to_numeric(latest[first_contacts_col], errors="coerce").sum(
        skipna=True
    )

    if not np.isfinite(waiters) or float(waiters) < 0:
        return None
    if not np.isfinite(first_contacts) or float(first_contacts) <= 0:
        return None

    weekly_contact_rate = float(first_contacts) / float(wim)
    if not np.isfinite(weekly_contact_rate) or weekly_contact_rate <= 0:
        return None

    clearance_weeks = float(waiters) / float(weekly_contact_rate)
    if not np.isfinite(clearance_weeks) or clearance_weeks < 0:
        return None
    return clearance_weeks


# Compute a derived KPI from the current filtered dataset.
def _compute_demand_ratio_latest(
    df: pd.DataFrame,
    *,
    demand_percentile: Optional[int | float] = 60,
) -> Optional[float]:
    """Compute latest-month Demand Ratio using the dashboard's target framework."""

    if df is None or df.empty:
        return None

    base = df.copy()
    if "period_end" not in base.columns:
        return None

    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base = base.dropna(subset=["period_end"])
    if base.empty:
        return None

    base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")
    latest_period_end = base["period_end"].max()
    if pd.isna(latest_period_end):
        return None

    required = {
        "service_line",
        "referrals",
        "clock_stop_actuals",
        "discharges_no_clock_stop",
    }
    if not required.issubset(set(base.columns)):
        return None

    try:
        p = float(demand_percentile) if demand_percentile is not None else 60.0
    except Exception:
        p = 60.0
    p = max(50.0, min(100.0, p))

    # Compute clock stop target per service line from historical monthly referrals.
    ref_base = base[["service_line", "period_end", "referrals"]].copy()
    ref_base["referrals"] = pd.to_numeric(ref_base["referrals"], errors="coerce")
    monthly_referrals = (
        ref_base.dropna(subset=["service_line", "period_end", "referrals"])
        .groupby(["service_line", "period_end"], as_index=False)
        .agg(referrals=("referrals", "sum"))
    )
    if monthly_referrals.empty:
        return None

    targets = monthly_referrals.groupby("service_line")["referrals"].quantile(p / 100.0)
    targets_map = targets.to_dict()

    latest = base[base["period_end"] == latest_period_end].copy()
    if latest.empty:
        return None

    latest["clock_stop_actuals"] = pd.to_numeric(
        latest["clock_stop_actuals"], errors="coerce"
    )
    latest["discharges_no_clock_stop"] = pd.to_numeric(
        latest["discharges_no_clock_stop"], errors="coerce"
    )

    by_service = latest.groupby("service_line", as_index=False).agg(
        clock_stop_actuals=("clock_stop_actuals", "sum"),
        discharges_no_clock_stop=("discharges_no_clock_stop", "sum"),
    )
    by_service["clock_stop_target"] = by_service["service_line"].map(targets_map)
    by_service["clock_stop_target"] = pd.to_numeric(
        by_service["clock_stop_target"], errors="coerce"
    )
    by_service = by_service.dropna(subset=["clock_stop_target"])
    by_service = by_service[by_service["clock_stop_target"] > 0]
    if by_service.empty:
        return None

    # Aggregate at trust-filter level: ratio of totals (not mean of service ratios)
    # so larger services are weighted by their target volumes.
    numer = (
        pd.to_numeric(by_service["clock_stop_actuals"], errors="coerce").fillna(0)
        + pd.to_numeric(by_service["discharges_no_clock_stop"], errors="coerce").fillna(
            0
        )
    ).sum()
    denom = pd.to_numeric(by_service["clock_stop_target"], errors="coerce").sum()

    if not np.isfinite(numer) or not np.isfinite(denom) or float(denom) <= 0:
        return None

    out = float(numer) / float(denom)
    if not np.isfinite(out):
        return None
    return out


# Compute a derived KPI from the current filtered dataset.
def _compute_sustainable_caseload_latest(
    df: pd.DataFrame,
    *,
    demand_percentile: Optional[int | float] = 60,
) -> Optional[float]:
    """Estimate sustainable caseload (latest month) assuming Demand Ratio = 1.0.

    Uses the same Demand Ratio framework as the Demand table:
      - clock_stop_target is a percentile of historical monthly referrals per service line
        within the currently filtered dataset.
      - demand_ratio is derived from clock_stop_actuals and discharges_no_clock_stop
        relative to that target.

    The card returns a single value:
      sustainable_caseload = latest_month_caseload / latest_month_demand_ratio
    where latest_month_demand_ratio is computed as:
      (sum(clock_stop_actuals) + sum(discharges_no_clock_stop)) / sum(clock_stop_target)
    across service lines with valid targets.
    """

    if df is None or df.empty:
        return None

    base = df.copy()
    if "period_end" not in base.columns:
        return None

    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base = base.dropna(subset=["period_end"])
    if base.empty:
        return None

    base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")
    latest_period_end = base["period_end"].max()
    if pd.isna(latest_period_end):
        return None

    # Inputs required to compute target + demand ratio.
    required = {
        "service_line",
        "referrals",
        "clock_stop_actuals",
        "discharges_no_clock_stop",
        "caseload",
    }
    if not required.issubset(set(base.columns)):
        return None

    try:
        p = float(demand_percentile) if demand_percentile is not None else 60.0
    except Exception:
        p = 60.0
    p = max(50.0, min(100.0, p))

    # Compute clock stop target per service line from historical monthly referrals.
    ref_base = base[["service_line", "period_end", "referrals"]].copy()
    ref_base["referrals"] = pd.to_numeric(ref_base["referrals"], errors="coerce")
    monthly_referrals = (
        ref_base.dropna(subset=["service_line", "period_end", "referrals"])
        .groupby(["service_line", "period_end"], as_index=False)
        .agg(referrals=("referrals", "sum"))
    )
    if monthly_referrals.empty:
        return None

    targets = monthly_referrals.groupby("service_line")["referrals"].quantile(p / 100.0)
    targets_map = targets.to_dict()

    # Latest month aggregates.
    latest = base[base["period_end"] == latest_period_end].copy()
    if latest.empty:
        return None

    latest["caseload"] = pd.to_numeric(latest["caseload"], errors="coerce")
    latest_caseload = latest["caseload"].sum(skipna=True)

    latest["clock_stop_actuals"] = pd.to_numeric(
        latest["clock_stop_actuals"], errors="coerce"
    )
    latest["discharges_no_clock_stop"] = pd.to_numeric(
        latest["discharges_no_clock_stop"], errors="coerce"
    )

    by_service = latest.groupby("service_line", as_index=False).agg(
        clock_stop_actuals=("clock_stop_actuals", "sum"),
        discharges_no_clock_stop=("discharges_no_clock_stop", "sum"),
    )
    by_service["clock_stop_target"] = by_service["service_line"].map(targets_map)
    by_service["clock_stop_target"] = pd.to_numeric(
        by_service["clock_stop_target"], errors="coerce"
    )

    # Keep only valid denominators.
    by_service = by_service.dropna(subset=["clock_stop_target"])
    by_service = by_service[by_service["clock_stop_target"] > 0]
    if by_service.empty:
        return None

    numer = (
        pd.to_numeric(by_service["clock_stop_actuals"], errors="coerce").fillna(0)
        + pd.to_numeric(by_service["discharges_no_clock_stop"], errors="coerce").fillna(
            0
        )
    ).sum()
    denom = pd.to_numeric(by_service["clock_stop_target"], errors="coerce").sum()
    if not np.isfinite(numer) or not np.isfinite(denom) or float(denom) <= 0:
        return None

    # Invert the latest demand ratio to estimate caseload level at DR=1.0:
    # sustainable_caseload = current_caseload / demand_ratio_latest.
    demand_ratio_latest = float(numer) / float(denom)
    if not np.isfinite(demand_ratio_latest) or demand_ratio_latest <= 0:
        return None

    if not np.isfinite(latest_caseload) or float(latest_caseload) < 0:
        return None

    return float(latest_caseload) / float(demand_ratio_latest)


# Build and return a dashboard layout section or component.
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

    demand_percentile_options = [
        {"label": f"{p}%", "value": p} for p in range(50, 101, 5)
    ]

    return html.Div(
        dbc.Row(
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
                                                value=[
                                                    default_start_idx,
                                                    default_end_idx,
                                                ],
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
                    width=4,
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
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Label(
                                        "Clinician Patient Facing Time:",
                                        className="fw-bold",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="demand-percentile-dropdown",
                                        options=demand_percentile_options,
                                        value=60,
                                        clearable=False,
                                    ),
                                    width=True,
                                ),
                            ],
                            className="align-items-center g-2",
                        )
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            "Apply Filters",
                            id="apply-filters-btn",
                            className="nhs-btn-primary w-100",
                        ),
                    ],
                    width=2,
                ),
            ],
            className="justify-content-center",
        ),
        className="nhs-filter-bar mb-4",
    )


# Build a monthly-indexed time series for charting and forecasts.
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


# Build a metric-specific chart, optionally with forecast overlay.
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
        xaxis_title="Period",
        yaxis_title="Count",
        hovermode="x unified",
        height=420,
        template="plotly_white",
        margin=dict(l=40, r=30, t=20, b=50),
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


# Helper that prepares the Overview tab visual output.
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

    palette = NHS_CHART_COLOURS
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
        margin=dict(l=40, r=200, t=20, b=50),
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
app.layout = html.Div(
    [
        # NHS Header Banner
        html.Div(
            dbc.Container(
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1("NHFT Demand & Capacity Platform"),
                            width=True,
                        ),
                        dbc.Col(
                            html.Img(
                                src="/images/NHFT.png",
                                className="nhs-logo-img",
                            ),
                            width="auto",
                            className="d-flex align-items-center",
                        ),
                    ],
                    className="align-items-center",
                ),
                fluid=True,
            ),
            className="nhs-header",
        ),
        # Body
        dbc.Container(
            [
                # Filters Row
                html.Div(className="mt-4"),
                create_filter_section(),
                # Summary Statistics Row
                html.Div(id="summary-stats-row"),
                # Main Content
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Tabs(
                                    id="main-tabs",
                                    value="overview-tab",
                                    children=[
                                        dcc.Tab(
                                            label="ℹ️  About This Dashboard",
                                            value="about-tab",
                                        ),
                                        dcc.Tab(
                                            label="📈  Overview",
                                            value="overview-tab",
                                        ),
                                        dcc.Tab(
                                            label="👥  Demand Analysis",
                                            value="demand-tab",
                                        ),
                                        dcc.Tab(
                                            label="💼  Capacity Analysis",
                                            value="capacity-tab",
                                        ),
                                    ],
                                    className="nhs-tabs mb-3",
                                ),
                                html.Div(id="tab-content"),
                            ],
                            width=12,
                        ),
                    ]
                ),
                # Store filtered data
                dcc.Store(id="filtered-data-store"),
            ],
            fluid=True,
        ),
        # NHS Footer
        html.Div(
            dbc.Container(
                html.Div(id="footer-content"),
                fluid=True,
            ),
            className="nhs-footer",
        ),
    ]
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
# Internal helper for shared dashboard logic.
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
        # Filter by the selected (provider_code_current, service_line) pairs.
        selected_pairs = set()
        for item in service_lines:
            try:
                provider, service = str(item).split("|", 1)
                selected_pairs.add((provider, service))
            except Exception:
                continue

        if selected_pairs:
            provider_series = df["provider_code_current"].astype(str)
            service_series = df["service_line"].astype(str)
            mask = [
                (p, s) in selected_pairs
                for p, s in zip(provider_series, service_series)
            ]
            df = df[pd.Series(mask, index=df.index)]

    # Ensure provider codes remain canonical before writing to the store.
    if "provider_code_current" in df.columns:
        df["provider_code_current"] = _format_provider_code_current(
            df["provider_code_current"]
        )

    # Store as JSON
    return df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("date-range-slider-label", "children"),
    [Input("date-range-slider", "value")],
    [State("date-slider-dates", "data")],
)
# Callback handler that updates UI output from current inputs.
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
# Callback handler that updates UI output from current inputs.
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

    df = _read_filtered_store_frame(data_json)

    if df.empty:
        return html.Div(
            "No data matches the selected filters", className="alert alert-info"
        )

    # Ensure period_end is datetime
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

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
# Callback handler that updates UI output from current inputs.
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
        df = _read_filtered_store_frame(data_json)

        if not df.empty and "period_end" in df.columns:
            latest_period_end = df["period_end"].max().strftime("%d %B %Y")
        else:
            latest_period_end = "N/A"

    return html.P(
        [
            "© 2025 Northamptonshire Healthcare NHS Foundation Trust  |  ",
            "BI Development Team  |  ",
            html.Strong(f"Latest Reporting Period End: {latest_period_end}"),
        ],
        className="text-center mb-0",
    )


@app.callback(
    Output("tab-content", "children"),
    [
        Input("main-tabs", "value"),
        Input("filtered-data-store", "data"),
        Input("demand-percentile-dropdown", "value"),
    ],
)
# Callback handler that updates UI output from current inputs.
def update_tab_content(active_tab, data_json, demand_percentile):
    """Render the selected tab content.

    Args:
        active_tab: Selected tab ID ("about-tab", "overview-tab", "demand-tab", "capacity-tab").
        data_json: JSON-encoded DataFrame (orient="split") from dcc.Store.

    Returns:
        A Dash component containing the tab content (charts/tables), or an alert
        div if data is missing/empty.
    """
    # Keep the About tab accessible even before filters are applied.
    if active_tab == "about-tab":
        return create_about_tab()

    if data_json is None:
        return html.Div(
            "No data available. Click 'Apply Filters' to load data.",
            className="alert alert-info",
        )

    df = _read_filtered_store_frame(data_json)

    if df.empty:
        return html.Div(
            "No data matches the selected filters", className="alert alert-warning"
        )

    if active_tab == "overview-tab":
        return create_overview_tab(df, demand_percentile=demand_percentile)
    elif active_tab == "demand-tab":
        return create_demand_tab(df, demand_percentile=demand_percentile)
    elif active_tab == "capacity-tab":
        return create_capacity_tab(df)

    return html.Div("Invalid tab selection")


# Build and return a dashboard layout section or component.
def create_overview_tab(df, *, demand_percentile: Optional[int | float] = 60):
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
        className="nhs-card mb-4",
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
                    className="nhs-card mb-4",
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
        className="nhs-card mb-4",
    )

    sustainable_caseload = _compute_sustainable_caseload_latest(
        df_sorted,
        demand_percentile=demand_percentile,
    )
    demand_ratio_latest = _compute_demand_ratio_latest(
        df_sorted,
        demand_percentile=demand_percentile,
    )
    net_flow_latest = _compute_net_caseload_flow_latest(df_sorted)
    throughput_rate_latest = _compute_caseload_throughput_rate_latest(df_sorted)
    clearance_weeks_latest = _compute_clearance_time_weeks_latest(df_sorted)
    sustainable_card_row = dbc.Row(
        [
            dbc.Col(
                create_metric_card(
                    "Demand Ratio",
                    _format_percent_card_value(demand_ratio_latest, decimals=1),
                    "⚖️",
                    "primary",
                    description="Total throughput vs target. 100% and above means demand was met last month",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Sustainable Caseload",
                    _format_int_card_value(sustainable_caseload),
                    "🌿",
                    "success",
                    description="Estimated caseload size sustainable at current throughput (based on Demand Ratio)",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Net Caseload Flow",
                    _format_signed_int_card_value(net_flow_latest),
                    "🔄",
                    "warning",
                    description="Balance of patients entering vs leaving caseload (First Contacts vs Discharges)",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Caseload Throughput",
                    _format_percent_card_value(throughput_rate_latest, decimals=1),
                    "♻️",
                    "info",
                    description="Percentage of patients discharged from caseload last month",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Clearance Time (Weeks)",
                    _format_ratio_card_value(clearance_weeks_latest, decimals=1),
                    "🧹",
                    "secondary",
                    description="Assuming referrals are closed, weeks to clear waiting list at last month's first contact rate",
                ),
            ),
        ],
        className="mb-4 justify-content-center",
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
                        "This tab provides a high-level summary of service demand and capacity. "
                        "The KPI cards show the latest month's key metrics, while the time series charts track trends over time. "
                        "Use the filters above to focus on specific services or date ranges, and click legend items to show/hide individual metrics."
                    ),
                    html.P(
                        "Forecast toggles on each chart use ETS (Holt-Winters Exponential Smoothing) to project future values with a 95% confidence interval. "
                        "For best results, include at least 2 years of data when using forecasts."
                    ),
                ],
                className="nhs-info-box mt-4",
            ),
            dcc.Store(
                id="overview-keymetrics-visible-metrics",
                data=default_visible_metrics,
            ),
            sustainable_card_row,
            key_metrics_chart,
            staff_vs_caseload_card,
            waiters_chart,
        ]
    )


# Internal helper for shared dashboard logic.
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
    x_dates = pd.PeriodIndex(
        waiters_18wk["year_month"].astype(str), freq="M"
    ).to_timestamp("M")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=waiters_18wk["waiters_under_18_weeks"],
            name="Under 18 Weeks",
            marker_color=NHS_BLUE,
            hovertemplate="%{fullData.name}: <b>%{y:,}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=waiters_18wk["waiters_over_18_weeks"],
            name="18+ Weeks",
            marker_color=NHS_RED,
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
# Callback handler that updates UI output from current inputs.
def update_overview_keymetrics_timeseries(data_json, forecast_on, visible_metrics):
    if data_json is None:
        return go.Figure()
    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
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
# Callback handler that updates UI output from current inputs.
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
# Callback handler that updates UI output from current inputs.
def update_referrals_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return _metric_timeseries_figure(
        df=df,
        metric="referrals",
        label="Referrals",
        color=NHS_BLUE,
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("waiters-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("waiters-forecast-toggle", "value")],
)
# Callback handler that updates UI output from current inputs.
def update_waiters_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return _metric_timeseries_figure(
        df=df,
        metric="waiters",
        label="Waiters",
        color=NHS_RED,
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("caseload-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("caseload-forecast-toggle", "value")],
)
# Callback handler that updates UI output from current inputs.
def update_caseload_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return _metric_timeseries_figure(
        df=df,
        metric="caseload",
        label="Caseload",
        color=NHS_AQUA_GREEN,
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


@app.callback(
    Output("contacts-timeseries", "figure"),
    [Input("filtered-data-store", "data"), Input("contacts-forecast-toggle", "value")],
)
# Callback handler that updates UI output from current inputs.
def update_contacts_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return _metric_timeseries_figure(
        df=df,
        metric="total_contacts",
        label="Contacts",
        color=NHS_PURPLE,
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
# Callback handler that updates UI output from current inputs.
def update_discharges_timeseries(data_json, forecast_on):
    if data_json is None:
        return go.Figure()
    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return go.Figure()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return _metric_timeseries_figure(
        df=df,
        metric="discharges_with_clock_stop",
        label="Discharges",
        color=NHS_ORANGE,
        forecast_on=bool(forecast_on),
        monthly_ticks=False,
    )


# Build and return a dashboard layout section or component.
def create_service_line_summary_table(df: pd.DataFrame):
    """Create the patient summary DataTable by provider code and service line.

    Args:
        df: Filtered patient dataset.

    Returns:
        A Dash DataTable where rows are (provider_code_current, service_line) and
        numeric metrics are aggregated by sum.
    """
    summary_df = _build_demand_summary_table_frame(df, demand_percentile=60)

    return dash_table.DataTable(
        id="patient-summary-table",
        data=summary_df.to_dict("records"),
        columns=[
            {
                "name": ("Period" if i == "period_end" else prettify_column_name(i)),
                "id": i,
            }
            for i in summary_df.columns
        ],
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 2},
        sort_action="native",
        sort_by=[
            {"column_id": "provider_code_current", "direction": "asc"},
            {"column_id": "period_end", "direction": "asc"},
        ],
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
            "backgroundColor": NHS_BLUE,
            "color": NHS_WHITE,
            "fontWeight": "bold",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F7F9FA"}
        ],
    )


# Internal helper for shared dashboard logic.
def _build_demand_summary_table_frame(
    df: pd.DataFrame,
    *,
    demand_percentile: Optional[int | float] = 60,
) -> pd.DataFrame:
    """Build the Demand Analysis summary table frame with derived columns.

        - Aggregates numeric metrics for each (service_line, period_end) so the table is
            split correctly by month.
        - Computes waiters under 18 weeks from waiters - waiters_over_18_weeks when
            missing.
    - Computes a clock stop target per service line based on the selected percentile
      of historical monthly referrals (within the currently filtered dataset).
    - Computes demand ratio metrics and contacts-per-caseload metrics.
    """

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["provider_code_current", "service_line", "period_end"]
        )

    try:
        p = float(demand_percentile) if demand_percentile is not None else 60.0
    except Exception:
        p = 60.0
    p = max(50.0, min(100.0, p))

    base = df.copy()
    if "period_end" in base.columns:
        base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
        # Normalise to month-end so multiple within-month dates don't create duplicates.
        base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")

    exclude_cols = {
        "provider_code_current",
        "service_line",
        "period_end",
        "year",
        "month",
        "quarter",
        "year_month",
    }

    numeric_cols: list[str] = [
        col
        for col in base.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(base[col])
    ]

    # Aggregate: sums for counts/volumes, means for "average_*" fields.
    agg: dict[str, str] = {}
    for col in numeric_cols:
        if str(col).startswith("average_"):
            agg[col] = "mean"
        else:
            agg[col] = "sum"

    group_keys = []
    if "provider_code_current" in base.columns:
        group_keys.append("provider_code_current")
    group_keys.append("service_line")
    if "period_end" in base.columns:
        group_keys.append("period_end")

    summary_df = base.groupby(group_keys).agg(agg).reset_index().sort_values(group_keys)

    # -----------------------------
    # Waiters under 18 weeks (computed if missing)
    # -----------------------------
    if {"waiters", "waiters_over_18_weeks"}.issubset(summary_df.columns):
        calc_under_18 = pd.to_numeric(
            summary_df["waiters"], errors="coerce"
        ) - pd.to_numeric(summary_df["waiters_over_18_weeks"], errors="coerce")
        if "waiters_under_18_weeks" in summary_df.columns:
            summary_df["waiters_under_18_weeks"] = pd.to_numeric(
                summary_df["waiters_under_18_weeks"], errors="coerce"
            ).fillna(calc_under_18)
        else:
            summary_df["waiters_under_18_weeks"] = calc_under_18

    # Clock Stop Target is computed once per service_line from full filtered history,
    # then repeated across monthly rows to provide a stable benchmark line.
    if {"service_line", "referrals", "period_end"}.issubset(base.columns):
        ref_base = base[["service_line", "period_end", "referrals"]].copy()
        ref_base["referrals"] = pd.to_numeric(ref_base["referrals"], errors="coerce")
        monthly_referrals = (
            ref_base.dropna(subset=["service_line", "period_end", "referrals"])
            .groupby(["service_line", "period_end"], as_index=False)
            .agg(referrals=("referrals", "sum"))
        )
        targets = monthly_referrals.groupby("service_line")["referrals"].quantile(
            p / 100.0
        )
        summary_df["clock_stop_target"] = summary_df["service_line"].map(
            targets.to_dict()
        )
    else:
        summary_df["clock_stop_target"] = None

    # -----------------------------
    # Derived ratios and metrics
    # -----------------------------
    def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
        numer_n = pd.to_numeric(numer, errors="coerce")
        denom_n = pd.to_numeric(denom, errors="coerce")
        out = numer_n / denom_n
        # Ratios are undefined when target/caseload is zero or missing.
        out = out.where(denom_n > 0)
        return out

    if "clock_stop_actuals" in summary_df.columns:
        summary_df["referral_clock_stop_ratio"] = _safe_divide(
            summary_df["clock_stop_actuals"], summary_df["clock_stop_target"]
        )
    else:
        summary_df["referral_clock_stop_ratio"] = None

    if "discharges_no_clock_stop" in summary_df.columns:
        summary_df["referral_discharged_no_clock_stop_ratio"] = _safe_divide(
            summary_df["discharges_no_clock_stop"], summary_df["clock_stop_target"]
        )
    else:
        summary_df["referral_discharged_no_clock_stop_ratio"] = None

    summary_df["demand_ratio"] = pd.to_numeric(
        summary_df["referral_clock_stop_ratio"], errors="coerce"
    ) + pd.to_numeric(
        summary_df["referral_discharged_no_clock_stop_ratio"], errors="coerce"
    )

    if {"total_caseload_contacts", "caseload"}.issubset(summary_df.columns):
        summary_df["total_contacts_per_caseload"] = _safe_divide(
            summary_df["total_caseload_contacts"], summary_df["caseload"]
        )
    else:
        summary_df["total_contacts_per_caseload"] = None

    if {"ftf_caseload_contacts", "caseload"}.issubset(summary_df.columns):
        summary_df["ftf_contacts_per_caseload"] = _safe_divide(
            summary_df["ftf_caseload_contacts"], summary_df["caseload"]
        )
    else:
        summary_df["ftf_contacts_per_caseload"] = None

    # Format/round for readability (keep ratios as decimals)
    for col in summary_df.columns:
        if col in {
            "referral_clock_stop_ratio",
            "referral_discharged_no_clock_stop_ratio",
            "demand_ratio",
            "total_contacts_per_caseload",
            "ftf_contacts_per_caseload",
        }:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").round(3)
        elif col in {"clock_stop_target"}:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").round(0)

    if "period_end" in summary_df.columns:
        summary_df["period_end"] = pd.to_datetime(
            summary_df["period_end"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    # Column order as requested
    preferred_order = [
        "provider_code_current",
        "service_line",
        "period_end",
        "referrals",
        "clock_stop_target",
        "clock_stop_actuals",
        "waiters",
        "waiters_over_18_weeks",
        "waiters_under_18_weeks",
        "caseload",
        "total_caseload_contacts",
        "ftf_caseload_contacts",
        "total_contacts_per_caseload",
        "ftf_contacts_per_caseload",
        "total_contacts",
        "ftf_contacts",
        "average_length_of_treatment",
        "average_contacts_at_discharge",
        "average_ftf_contacts_at_discharge",
        "discharges_no_clock_stop",
        "discharges_with_clock_stop",
        "referral_clock_stop_ratio",
        "referral_discharged_no_clock_stop_ratio",
        "demand_ratio",
    ]
    existing = [c for c in preferred_order if c in summary_df.columns]
    remaining = [c for c in summary_df.columns if c not in existing]
    return summary_df[existing + remaining]


@app.callback(
    [
        Output("patient-summary-table", "data"),
        Output("patient-summary-table", "columns"),
    ],
    [
        Input("filtered-data-store", "data"),
        Input("demand-percentile-dropdown", "value"),
    ],
)
# Callback handler that updates UI output from current inputs.
def update_patient_summary_table(data_json, demand_percentile):
    """Recalculate Demand Analysis table metrics when filters change."""
    if data_json is None:
        return [], []

    df = _read_filtered_store_frame(data_json)
    if df.empty:
        return [], []

    summary_df = _build_demand_summary_table_frame(
        df,
        demand_percentile=demand_percentile,
    )

    columns = [
        {
            "name": ("Period" if i == "period_end" else prettify_column_name(i)),
            "id": i,
        }
        for i in summary_df.columns
    ]
    return summary_df.to_dict("records"), columns


# Build and return a dashboard layout section or component.
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
            "backgroundColor": NHS_BLUE,
            "color": NHS_WHITE,
            "fontWeight": "bold",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F7F9FA"}
        ],
    )


# Helper used by Capacity Analysis charts and tables.
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


# Helper used by Capacity Analysis charts and tables.
def _capacity_style_figure(fig: go.Figure) -> go.Figure:
    """Apply a consistent, dashboard-friendly Plotly style."""
    fig.update_layout(
        margin=dict(l=40, r=40, t=20, b=40),
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


# Helper used by Capacity Analysis charts and tables.
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
        labels={"staff": "Staff (count)", "staff_group": "Staff group"},
    )
    fig.update_traces(hovertemplate="%{y}<br>Staff: %{x:,}<extra></extra>")
    return _capacity_style_figure(fig)


# Helper used by Capacity Analysis charts and tables.
def _capacity_staff_by_service_line_figure(
    staffing_filtered: pd.DataFrame,
) -> go.Figure:
    if staffing_filtered is None or staffing_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    if "service_line" not in staffing_filtered.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    by_service_line = (
        staffing_filtered.groupby("service_line", as_index=False)
        .agg(staff=("staff", "sum"))
        .sort_values("staff", ascending=True)
    )

    n_lines = int(by_service_line.shape[0])
    fig_height = max(460, 28 * n_lines + 120)

    fig = px.bar(
        by_service_line,
        x="staff",
        y="service_line",
        orientation="h",
        labels={"staff": "Staff (count)", "service_line": "Service Line"},
        height=fig_height,
    )
    fig.update_traces(hovertemplate="%{y}<br>Staff: %{x:,}<extra></extra>")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _capacity_style_figure(fig)


# Helper used by Capacity Analysis charts and tables.
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

    # Inner join keeps only provider/service pairs present in both patient
    # demand and staffing snapshots for like-for-like comparison.
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


# Note: Helper that builds a net caseload flow trend line chart over time.
def _demand_net_flow_trend_figure(df: pd.DataFrame) -> go.Figure:
    """Line chart showing monthly net caseload flow (referrals minus discharges) by service over time."""

    summary_df = _build_demand_summary_table_frame(df, demand_percentile=60)
    fig = go.Figure()

    required_cols = {"service_line", "referrals", "period_end"}
    if summary_df.empty or not required_cols.issubset(set(summary_df.columns)):
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Not enough data to compute net caseload flow trend",
            showarrow=False,
        )
        fig.update_layout(template="plotly_white", height=420)
        return fig

    summary_df = summary_df.copy()
    summary_df["period_end"] = pd.to_datetime(summary_df["period_end"], errors="coerce")
    summary_df["referrals"] = pd.to_numeric(summary_df["referrals"], errors="coerce")

    # Calculate total outflow (discharges)
    if "discharges_no_clock_stop" in summary_df.columns:
        discharges_no = pd.to_numeric(
            summary_df["discharges_no_clock_stop"], errors="coerce"
        ).fillna(0)
    else:
        discharges_no = pd.Series(0, index=summary_df.index, dtype=float)

    if "discharges_with_clock_stop" in summary_df.columns:
        discharges_with = pd.to_numeric(
            summary_df["discharges_with_clock_stop"], errors="coerce"
        ).fillna(0)
    elif "clock_stop_actuals" in summary_df.columns:
        discharges_with = pd.to_numeric(
            summary_df["clock_stop_actuals"], errors="coerce"
        ).fillna(0)
    else:
        discharges_with = pd.Series(0, index=summary_df.index, dtype=float)

    summary_df["outflow"] = discharges_with + discharges_no
    summary_df["net_flow"] = summary_df["referrals"] - summary_df["outflow"]

    if {"provider_code_current", "service_line"}.issubset(summary_df.columns):
        summary_df["service_label"] = (
            summary_df["provider_code_current"].astype(str)
            + " - "
            + summary_df["service_line"].astype(str)
        )
    else:
        summary_df["service_label"] = summary_df["service_line"].astype(str)

    summary_df = summary_df.dropna(subset=["period_end", "net_flow"])
    summary_df = summary_df.sort_values("period_end")

    if summary_df.empty:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid net flow data for selected filters",
            showarrow=False,
        )
        fig.update_layout(template="plotly_white", height=420)
        return fig

    services = summary_df["service_label"].unique()
    for svc in services:
        svc_df = summary_df[summary_df["service_label"] == svc]
        fig.add_trace(
            go.Scatter(
                x=svc_df["period_end"],
                y=svc_df["net_flow"],
                mode="lines+markers",
                line=dict(shape="spline", smoothing=1.0),
                name=svc,
                hovertemplate="%{x|%b %Y}<br>Net Flow: %{y:,.0f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#444")
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Net Caseload Flow",
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=30, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemclick=False,
            itemdoubleclick=False,
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickformat=",", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


# Note: Compute a derived KPI from the current filtered dataset.
def _compute_discharge_quality_latest(
    df: pd.DataFrame,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return latest-month weighted averages for discharge quality metrics.

    Returns:
        (avg_treatment_days, avg_contacts, avg_ftf_contacts) — any may be None.
    """
    if df is None or df.empty:
        return None, None, None

    base = df.copy()
    if "period_end" not in base.columns:
        return None, None, None

    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base = base.dropna(subset=["period_end"])
    if base.empty:
        return None, None, None

    base["period_end"] = base["period_end"].dt.to_period("M").dt.to_timestamp("M")
    latest_period = base["period_end"].max()
    if pd.isna(latest_period):
        return None, None, None

    latest = base[base["period_end"] == latest_period]
    if latest.empty:
        return None, None, None

    def _mean_or_none(col: str) -> Optional[float]:
        if col not in latest.columns:
            return None
        s = pd.to_numeric(latest[col], errors="coerce").dropna()
        if s.empty:
            return None
        v = float(s.mean())
        return v if np.isfinite(v) else None

    return (
        _mean_or_none("average_length_of_treatment"),
        _mean_or_none("average_contacts_at_discharge"),
        _mean_or_none("average_ftf_contacts_at_discharge"),
    )


# Build and return a dashboard layout section or component.
def create_demand_tab(df, *, demand_percentile: Optional[int | float] = 60):
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
        className="nhs-card mb-4",
    )

    summary_table = create_service_line_summary_table(df)
    net_flow_trend_fig = _demand_net_flow_trend_figure(df_sorted)
    avg_treatment_days, avg_contacts, avg_ftf_contacts = (
        _compute_discharge_quality_latest(df_sorted)
    )

    net_flow_trend_card = dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    "Net Caseload Flow Over Time",
                    className="mb-0",
                )
            ),
            dbc.CardBody(
                dcc.Graph(
                    id="demand-net-flow-trend",
                    figure=net_flow_trend_fig,
                ),
                className="p-2",
            ),
        ],
        className="nhs-card mb-4",
    )

    discharge_quality_cards = html.Div(
        [
            html.H5(
                "Discharge Quality (Latest Month)",
                className="mb-3 text-center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_metric_card(
                            "Avg Treatment Length",
                            _format_ratio_card_value(avg_treatment_days, decimals=0),
                            "⏱️",
                            "primary",
                            description="Average days from referral start to first contact for discharged patients",
                        ),
                    ),
                    dbc.Col(
                        create_metric_card(
                            "Avg Contacts at Discharge",
                            _format_ratio_card_value(avg_contacts, decimals=1),
                            "📋",
                            "info",
                            description="Average total contacts per patient at point of discharge",
                        ),
                    ),
                    dbc.Col(
                        create_metric_card(
                            "Avg FTF Contacts at Discharge",
                            _format_ratio_card_value(avg_ftf_contacts, decimals=1),
                            "🤝",
                            "success",
                            description="Average face-to-face contacts per patient at point of discharge",
                        ),
                    ),
                ],
                className="g-3",
            ),
        ],
    )

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
            className="nhs-card mb-4",
        )

    sustainable_caseload = _compute_sustainable_caseload_latest(
        df_sorted,
        demand_percentile=demand_percentile,
    )
    demand_ratio_latest = _compute_demand_ratio_latest(
        df_sorted,
        demand_percentile=demand_percentile,
    )
    net_flow_latest = _compute_net_caseload_flow_latest(df_sorted)
    throughput_rate_latest = _compute_caseload_throughput_rate_latest(df_sorted)
    clearance_weeks_latest = _compute_clearance_time_weeks_latest(df_sorted)
    sustainable_card_row = dbc.Row(
        [
            dbc.Col(
                create_metric_card(
                    "Demand Ratio",
                    _format_percent_card_value(demand_ratio_latest, decimals=1),
                    "⚖️",
                    "primary",
                    description="Total throughput vs target. 100% and above means demand was met last month",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Sustainable Caseload",
                    _format_int_card_value(sustainable_caseload),
                    "🌿",
                    "success",
                    description="Estimated caseload size sustainable at current throughput (based on Demand Ratio)",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Net Caseload Flow",
                    _format_signed_int_card_value(net_flow_latest),
                    "🔄",
                    "warning",
                    description="Balance of patients entering vs leaving caseload (First Contacts vs Discharges)",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Caseload Throughput",
                    _format_percent_card_value(throughput_rate_latest, decimals=1),
                    "♻️",
                    "info",
                    description="Percentage of patients discharged from caseload last month",
                ),
            ),
            dbc.Col(
                create_metric_card(
                    "Clearance Time (Weeks)",
                    _format_ratio_card_value(clearance_weeks_latest, decimals=1),
                    "🧹",
                    "secondary",
                    description="Assuming referrals are closed, weeks to clear waiting list at last month's first contact rate",
                ),
            ),
        ],
        className="mb-4 justify-content-center",
    )

    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("👥 Demand Analysis", className="alert-heading"),
                    html.P(
                        "This tab breaks down patient demand in detail. It tracks referrals, waiting lists, caseload, contacts, and discharges "
                        "over time, alongside a summary table showing service-level performance against targets. "
                        "The Clinician Patient Facing Time filter sets the target threshold used in the demand summary table."
                    ),
                    html.P(
                        "Each chart includes a forecast toggle using ETS (Holt-Winters Exponential Smoothing) with a 95% confidence interval. "
                        "For best results, include at least 2 years of data when using forecasts."
                    ),
                ],
                className="nhs-info-box mt-4",
            ),
            sustainable_card_row,
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
                        discharge_quality_cards,
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(waiters_breakdown_card, width=12),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(net_flow_trend_card, width=12),
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


# Build and return a dashboard layout section or component.
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
        fig_by_service_line = _capacity_staff_by_service_line_figure(staffing_filtered)
        fig_staff_vs_case = _capacity_staff_vs_caseload_latest_figure(
            df, staffing_filtered
        )

        def graph_card(title: str, fig: go.Figure, graph_id: str):
            return dbc.Card(
                [
                    dbc.CardHeader(html.H5(title, className="mb-0")),
                    dbc.CardBody(dcc.Graph(id=graph_id, figure=fig), className="p-2"),
                ],
                className="nhs-card mb-4",
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

        capacity_graphs.append(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H5("Staff by Service Line", className="mb-0")
                                ),
                                dbc.CardBody(
                                    html.Div(
                                        dcc.Graph(
                                            id="capacity-staff-by-service-line",
                                            figure=fig_by_service_line,
                                        ),
                                        style={
                                            "maxHeight": "560px",
                                            "overflowY": "auto",
                                        },
                                    ),
                                    className="p-2",
                                ),
                            ],
                            className="nhs-card mb-4",
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
                        "This tab shows workforce capacity for the selected services. "
                        "It displays staffing levels broken down by staff group (e.g. qualified, unqualified, support) "
                        "and service line, helping identify where workforce gaps may exist relative to demand."
                    ),
                    html.P(
                        "Use the filters above to focus on specific providers or service lines. "
                        "The staffing table at the bottom provides a detailed breakdown of staff numbers."
                    ),
                    staffing_error if staffing_error is not None else None,
                ],
                className="nhs-info-box mt-4",
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


# Build and return a dashboard layout section or component.
def create_about_tab():
    """Create the About This Dashboard tab with full methodology and guidance."""

    return html.Div(
        [
            dbc.Alert(
                [
                    html.H4("ℹ️ About This Dashboard", className="alert-heading"),
                    html.P(
                        "This page explains what the dashboard does, how its metrics are calculated, "
                        "and how to interpret the data across each tab."
                    ),
                ],
                className="nhs-info-box mt-4",
            ),
            # --- Purpose ---
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Purpose", className="mb-0")),
                    dbc.CardBody(
                        [
                            html.P(
                                "This dashboard provides a data-driven view of demand, activity, and capacity across "
                                "NHFT's community and mental health services. It is designed to support operational managers, "
                                "service leads, and the Business Intelligence Service in understanding service pressures, "
                                "identifying trends, and making informed planning decisions."
                            ),
                            html.P(
                                "The platform tracks the full patient pathway from referral through to discharge and "
                                "provides standardised metrics across all active service lines. It brings together referral demand, "
                                "waiting list position, contact activity, caseload size, and discharge throughput into a single "
                                "platform, enabling services to be compared on a consistent basis."
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
            # --- How Demand Is Measured ---
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("How Demand Is Measured", className="mb-0")),
                    dbc.CardBody(
                        [
                            html.P(
                                "At the heart of the model is a demand benchmarking approach. Rather than relying on fixed "
                                "capacity targets (which require workforce data that is not currently available at the required "
                                "granularity), the model uses a configurable Clinician Patient Facing Time parameter to derive a "
                                "Clock Stop Target for each service line."
                            ),
                            html.P(
                                "This target represents the referral volume at a chosen percentile of each service's own history. "
                                "For example, at the 60th percentile, the target is the monthly referral volume that 60% of "
                                "historical months fall at or below."
                            ),
                            html.P(
                                [
                                    html.Strong("Demand Ratio"),
                                    " is then calculated by comparing actual throughput (first contacts and non-clock-stop "
                                    "discharges) against this target. A ratio of 100% or above indicates the service is keeping "
                                    "pace with demand; below 100% suggests a growing gap.",
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Sustainable Caseload"),
                                    " extends this framework by applying the Demand Ratio to the current caseload size, "
                                    "estimating how many patients the service can realistically sustain at its current throughput.",
                                ]
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
            # --- Key Operational Metrics ---
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H5("Key Operational Metrics", className="mb-0")
                    ),
                    dbc.CardBody(
                        [
                            html.H6(
                                "Caseload Throughput Rate", className="fw-bold mt-2"
                            ),
                            html.P(
                                "Measures the proportion of the caseload that is discharged each month. A higher rate indicates "
                                "patients are flowing through the service more actively; a declining rate suggests patients are "
                                "accumulating on the caseload without being discharged, which compounds capacity pressure over time."
                            ),
                            html.H6("Net Caseload Flow", className="fw-bold mt-3"),
                            html.P(
                                "Shows the monthly balance of patients entering the caseload (via first contacts) against those "
                                "leaving (via all discharge types). A positive value means the caseload is growing; a negative "
                                "value means it is shrinking. Persistent positive flow, combined with a low throughput rate, "
                                "signals that the service is heading towards an unsustainably large caseload."
                            ),
                            html.H6("Clearance Time", className="fw-bold mt-3"),
                            html.P(
                                "Estimates the number of weeks it would take to clear the current waiting list if no new referrals "
                                "were received, based on the latest month's rate of first contacts. A rising clearance time indicates "
                                "the waiting list is growing faster than the service can process it. This metric is particularly "
                                "useful for identifying services where capacity is insufficient to prevent waiting list deterioration."
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
            # --- Tabs in the Dashboard ---
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Tabs in the Dashboard", className="mb-0")),
                    dbc.CardBody(
                        [
                            html.H6("\U0001f4c8 Overview", className="fw-bold mt-2"),
                            html.P(
                                "Provides a high-level summary of service demand and capacity. KPI cards show the latest month's "
                                "headline figures while time series charts track referrals, waiters, caseload, contacts, and "
                                "discharges over time. A staff-vs-caseload scatter plot and a waiting list breakdown (under vs "
                                "over 18 weeks) provide additional operational context."
                            ),
                            html.H6(
                                "\U0001f465 Demand Analysis", className="fw-bold mt-3"
                            ),
                            html.P(
                                "Breaks down patient demand in detail with individual time series for each metric, each with "
                                "its own forecast toggle. A waiting list breakdown by time band is shown alongside a service-level "
                                "summary table that includes the Clock Stop Target (driven by the Clinician Patient Facing Time "
                                "filter), demand ratios, and contacts-per-caseload metrics where source data exists."
                            ),
                            html.H6(
                                "\U0001f4bc Capacity Analysis", className="fw-bold mt-3"
                            ),
                            html.P(
                                "Shows workforce capacity for the selected services, broken down by staff group and service line. "
                                "Due to current data constraints (see Data Limitations below), this tab presents capacity as a "
                                "reference view rather than a fully integrated demand-capacity model. It can be used to triangulate "
                                "demand signals with available workforce context."
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
            # --- Forecasting ---
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Forecasting", className="mb-0")),
                    dbc.CardBody(
                        [
                            html.P(
                                "Each chart includes an optional forecast toggle that generates a 6-month projection using "
                                "Exponential Smoothing (ETS / Holt-Winters) with a 95% confidence interval. Forecasts are "
                                "intended to support proactive planning rather than replace clinical judgement."
                            ),
                            html.P(
                                "For best results, include at least 2\u20133 years of data (24\u201336 monthly data points) in the "
                                "date filter so the model has enough history to learn seasonal patterns."
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
            # --- Data & Refresh ---
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Data Sources & Refresh", className="mb-0")),
                    dbc.CardBody(
                        [
                            html.P(
                                "Data is sourced from the NHFT Data Warehouse. Following each refresh, the dashboard "
                                "presents activity up to the last fully completed reporting month, in line with the Trust's "
                                "standard monthly reporting cycle."
                            ),
                            html.H6("Service Line Mapping", className="fw-bold mt-3"),
                            html.P(
                                "Patient-level data is mapped to service lines via the configuration table "
                                "[MIS_Config].[dbo].[tbl_org_current_RL9_Service_Line]. Only services flagged as Active and "
                                "RTT Report Enabled are included. Provider codes 996 and 998 are excluded."
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
            # --- Data Limitations ---
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H5(
                            "Data Limitations, Triangulation & Capacity Approach",
                            className="mb-0",
                        )
                    ),
                    dbc.CardBody(
                        [
                            html.P(
                                "During development, two practical constraints were identified that materially shape "
                                "the modelling approach:"
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        [
                                            html.Strong(
                                                "No historical workforce time series: "
                                            ),
                                            "Due to extraction limitations, historical staffing data was not obtainable at the "
                                            "required granularity. The staffing dataset is therefore a current snapshot only "
                                            "(no month-by-month staffing history).",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong(
                                                "Staff vs patient service line misalignment: "
                                            ),
                                            "Patient activity and workforce data do not naturally align by service line because "
                                            "service definitions are represented differently across systems (e.g. service lines vs "
                                            "cost centres / organisational hierarchies).",
                                        ]
                                    ),
                                ]
                            ),
                            html.P(
                                "As a result, the Capacity tab is implemented as a reference view (staff mix, staff by "
                                "service line), providing contextual workforce information rather than a fully time-aligned "
                                "capacity model. The project focuses on demand-based benchmarking using demand percentiles, "
                                "with the staffing snapshot providing supplementary capacity signals."
                            ),
                        ]
                    ),
                ],
                className="nhs-card mb-4",
            ),
        ]
    )


# ---------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting NHFT Demand-Capacity Dashboard...")
    logger.info("Dashboard will be available at: http://127.0.0.1:8050/")

    # Keep the debug reloader enabled so layout/code changes appear immediately
    # after save during development.
    try:
        app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=True)
    except OSError as e:
        logger.error("❌ Failed to start Dash server: %s", e)
        logger.error(
            "If you already have the dashboard running, stop it (Ctrl+C) or change the port."
        )
        raise
