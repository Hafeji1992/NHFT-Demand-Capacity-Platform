from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

FillMissing = Literal["zero", "ffill", "bfill", "interpolate", "drop"]
AggMethod = Literal["sum", "mean", "median", "count"]


def build_monthly_series(
    df: pd.DataFrame,
    value_col: str,
    *,
    date_col: str = "period_end",
    agg: AggMethod = "sum",
    fill_missing: FillMissing = "zero",
    freq: str = "M",
) -> pd.Series:
    """Aggregate a raw transactional/row-level DataFrame into a monthly time series.

    The dashboard uses month-level reporting. This helper converts rows into a
    DatetimeIndex series with a regular monthly frequency.

    Args:
        df: Input DataFrame containing a date column and the metric column.
        value_col: Column to aggregate (e.g., "referrals", "waiters").
        date_col: Column containing dates (default "period_end").
        agg: Aggregation method to apply within each month.
        fill_missing:
            How to handle months that have no rows after aggregation:
            - "zero": fill missing months with 0
            - "ffill"/"bfill": forward/back fill
            - "interpolate": time interpolation
            - "drop": keep only months present in data
        freq: Pandas frequency string. Use "M" for month-end or "MS" for month-start.

    Returns:
        Pandas Series with DatetimeIndex at monthly frequency.

    Raises:
        ValueError: if required columns are missing or data cannot be coerced.
    """

    if date_col not in df.columns:
        raise ValueError(f"DataFrame missing required date column: {date_col}")
    if value_col not in df.columns:
        raise ValueError(f"DataFrame missing required value column: {value_col}")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        bad = int(dates.isna().sum())
        raise ValueError(f"{bad} rows have invalid dates in column '{date_col}'")

    values = pd.to_numeric(df[value_col], errors="coerce")
    if values.isna().all():
        raise ValueError(f"All values are NaN after coercion for column '{value_col}'")

    s = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates))

    if agg == "sum":
        monthly = s.resample(freq).sum(min_count=1)
    elif agg == "mean":
        monthly = s.resample(freq).mean()
    elif agg == "median":
        monthly = s.resample(freq).median()
    elif agg == "count":
        monthly = s.resample(freq).count()
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    # Enforce a regular frequency index. resample() generally provides this already,
    # but asfreq() keeps us honest.
    monthly = monthly.asfreq(freq)

    if fill_missing == "drop":
        monthly = monthly.dropna()
    elif fill_missing == "zero":
        monthly = monthly.fillna(0.0)
    elif fill_missing == "ffill":
        monthly = monthly.ffill().fillna(0.0)
    elif fill_missing == "bfill":
        monthly = monthly.bfill().fillna(0.0)
    elif fill_missing == "interpolate":
        monthly = monthly.interpolate(method="time").fillna(0.0)
    else:
        raise ValueError(f"Unsupported fill_missing: {fill_missing}")

    # Statsmodels prefers finite numbers.
    monthly = monthly.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return monthly


def coerce_monthly_series(
    y: pd.Series,
    *,
    freq: str = "M",
    fill_missing: FillMissing = "zero",
) -> pd.Series:
    """Coerce an existing series into a regular monthly series.

    Useful if you already have a monthly aggregation (e.g., from the dashboard
    groupby on `year_month`) and want to ensure SARIMA can fit.
    """

    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("y must have a DatetimeIndex")

    y2 = pd.to_numeric(y, errors="coerce").astype(float)
    y2 = y2.sort_index()

    # Ensure regular monthly spacing
    y2 = y2.asfreq(freq)

    if fill_missing == "drop":
        y2 = y2.dropna()
    elif fill_missing == "zero":
        y2 = y2.fillna(0.0)
    elif fill_missing == "ffill":
        y2 = y2.ffill().fillna(0.0)
    elif fill_missing == "bfill":
        y2 = y2.bfill().fillna(0.0)
    elif fill_missing == "interpolate":
        y2 = y2.interpolate(method="time").fillna(0.0)
    else:
        raise ValueError(f"Unsupported fill_missing: {fill_missing}")

    y2 = y2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return y2
