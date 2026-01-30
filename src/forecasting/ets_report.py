"""NHFT ETS Console Report
========================

Generates a console-based forecasting report for a chosen metric using
Holt-Winters Exponential Smoothing (ETS):

- Loads a CSV dataset (default: data/patient_data.csv)
- Optionally filters by provider code and/or service line
- Aggregates to a regular monthly time series
- Performs lightweight ETS model selection using AIC over a small grid
- Fits the selected ETS model and prints a model summary (when available)
- Produces a multi-step forecast with approximate confidence intervals

Notes
-----
- ETS does not require stationarity in the same way ARIMA models do, so we do
  not run an ADF test here.
- Prediction intervals are approximate (residual-based) and intended for
  planning bands on the dashboard.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure src is on the path so `forecasting.*` imports work when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.preprocessing import build_monthly_series
from forecasting.ets import EtsForecaster, EtsSpec, small_grid_search_aic_ets


def _print_title(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _default_csv_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "patient_data.csv"


def _find_provider_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "provider_code_current",
        "ProviderCodeCurrent",
        "providerCodeCurrent",
        "Provider_Code_Current",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_service_line_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "service_line",
        "Service_Line",
        "serviceLine",
        "ServiceLine",
        "service_line_current",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise_code(value: object) -> str:
    s = str(value).strip().upper()
    if s == "" or s == "NAN":
        return ""
    if s.isdigit():
        s2 = s.lstrip("0")
        return s2 if s2 != "" else "0"
    return s


def _load_and_filter(
    *,
    csv_path: Path,
    metric: str,
    date_col: str,
    provider_code: Optional[str],
    service_line: Optional[str],
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. Available columns: {', '.join(map(str, df.columns[:25]))}"
        )
    if metric not in df.columns:
        raise ValueError(
            f"Metric column '{metric}' not found. Available columns: {', '.join(map(str, df.columns[:25]))}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if provider_code:
        prov_col = _find_provider_column(df)
        if prov_col is None:
            raise ValueError(
                "Provider filter requested but no provider_code_current-like column was found."
            )
        wanted = _normalise_code(provider_code)
        df = df[df[prov_col].map(_normalise_code) == wanted]

    if service_line:
        sl_col = _find_service_line_column(df)
        if sl_col is None:
            raise ValueError(
                "Service-line filter requested but no service_line-like column was found."
            )
        df = df[df[sl_col].astype(str).str.strip() == str(service_line).strip()]

    # Coerce metric to numeric for aggregation
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    return df


def _print_series_section(y: pd.Series) -> None:
    _print_title("Monthly Series Summary")
    print(f"Points:     {len(y)}")
    if len(y):
        print(f"Start:      {y.index.min().date()}")
        print(f"End:        {y.index.max().date()}")
        print(f"Total:      {float(np.nansum(y.values)):.0f}")
        print(f"Mean:       {float(np.nanmean(y.values)):.2f}")
        print(
            f"Std dev:    {float(np.nanstd(y.values, ddof=1)) if len(y) > 1 else 0.0:.2f}"
        )


def _print_model_selection_section(
    y: pd.Series, *, seasonal_period: int, show_warnings: bool
) -> EtsSpec:
    _print_title("Model Selection (ETS AIC Search)")

    print("Evaluating a small set of ETS specifications and selecting the model")
    print(
        "with the lowest AIC (Akaike Information Criterion), balancing fit vs complexity."
    )
    print(
        "Lower AIC indicates a better trade-off (not a guarantee of best future accuracy).\n"
    )

    # With short series, seasonality can still be useful but can also overfit.
    seasonal_opts = ("add", None) if len(y) >= 24 else (None,)

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        best = small_grid_search_aic_ets(
            y,
            seasonal_period=seasonal_period,
            trend=("add", None),
            seasonal=seasonal_opts,
            damped_trend=(True, False),
        )

    print("Selected ETS spec (lowest AIC within the small grid):")
    print(f"  trend:          {best.trend}")
    print(f"  seasonal:       {best.seasonal}")
    print(f"  seasonal_periods:{best.seasonal_periods}")
    print(f"  damped_trend:   {best.damped_trend}")

    return best


def _print_fit_and_forecast_section(
    y: pd.Series,
    *,
    spec: EtsSpec,
    months_ahead: int,
    conf_level: float,
    show_warnings: bool,
) -> None:
    _print_title("Model Fit")

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        model = EtsForecaster(spec=spec).fit(y)

    res = model._results  # internal statsmodels results

    summary_fn = getattr(res, "summary", None)
    if callable(summary_fn):
        print("Fitted ETS model summary:")
        try:
            print(summary_fn())
        except Exception:
            # Some versions do not support summary; fall back to params.
            pass

    # Always print a compact param block
    params = getattr(res, "params", None)
    if params is not None:
        print("\nFitted parameters (abbrev):")
        try:
            if isinstance(params, (pd.Series, dict)):
                s = pd.Series(params)
                print(s.head(20).to_string())
            else:
                print(str(params)[:800])
        except Exception:
            print(str(params)[:800])

    aic = getattr(res, "aic", None)
    if aic is not None:
        try:
            print(f"\nAIC: {float(aic):.3f}")
        except Exception:
            pass

    _print_title(
        f"Forecast ({months_ahead} months ahead, {int(conf_level*100)}% interval)"
    )
    fc = model.forecast(months_ahead, conf_level=conf_level)

    print("Forecast preview:")
    print(fc.head(min(months_ahead, 12)).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print an ETS console report (AIC selection, fit, forecast)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(_default_csv_path()),
        help="Path to input CSV (default: data/patient_data.csv)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="referrals",
        help="Metric column to forecast (e.g., referrals, waiters, staff)",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="period_end",
        help="Date column name (default: period_end)",
    )
    parser.add_argument(
        "--months-ahead",
        type=int,
        default=12,
        help="Forecast horizon in months (default: 12)",
    )
    parser.add_argument(
        "--conf-level",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Optional provider code to filter (e.g., 43)",
    )
    parser.add_argument(
        "--service-line",
        type=str,
        default=None,
        help="Optional service line name to filter",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show statsmodels warnings (default: hidden)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    _print_title("ETS Forecasting Report")
    print(f"CSV:         {csv_path}")
    print(f"Metric:      {args.metric}")
    print(f"Date column: {args.date_col}")
    if args.provider:
        print(f"Provider:    {args.provider}")
    if args.service_line:
        print(f"Service line:{args.service_line}")

    df = _load_and_filter(
        csv_path=csv_path,
        metric=str(args.metric),
        date_col=str(args.date_col),
        provider_code=args.provider,
        service_line=args.service_line,
    )

    if df.empty:
        raise RuntimeError("No rows remain after filtering; cannot build a series.")

    y = build_monthly_series(df, str(args.metric), date_col=str(args.date_col))
    if y.empty:
        raise RuntimeError("Monthly series is empty; check metric/date columns.")

    _print_series_section(y)

    # Model selection (AIC)
    spec = _print_model_selection_section(
        y, seasonal_period=12, show_warnings=bool(args.show_warnings)
    )

    # Fit + forecast
    _print_fit_and_forecast_section(
        y,
        spec=spec,
        months_ahead=int(args.months_ahead),
        conf_level=float(args.conf_level),
        show_warnings=bool(args.show_warnings),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
