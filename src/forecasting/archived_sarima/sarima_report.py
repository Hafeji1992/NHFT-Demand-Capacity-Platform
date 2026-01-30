"""
NHFT SARIMA Console Report (Archived)
====================================
Generates a console-based forecasting report for a chosen metric:

- Loads a CSV dataset (default: data/patient_data.csv)
- Aggregates to a regular monthly time series
- Runs an Augmented Dickey-Fuller (ADF) stationarity test
- Performs lightweight SARIMA model selection using AIC over a small grid
- Fits the selected SARIMA/SARIMAX model and prints a model summary
- Produces a multi-step forecast with confidence intervals

This script is kept for reference; the dashboard defaults to ETS.

"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ensure src is on the path so `forecasting.*` imports work when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecasting.preprocessing import build_monthly_series
from forecasting.archived_sarima.sarima import SarimaForecaster, SarimaSpec


# ---------------------------------------------------------------------
# Provider Selector (optional)
# ---------------------------------------------------------------------
# Set this to a provider code (e.g. "123") to run the SARIMA report for that
# provider only. Leave as None to default to ALL providers in the dataset.
PROVIDER_CODE_CURRENT: str | "43" = "43"


@dataclass(frozen=True)
class ADFResult:
    test_statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_values: dict


def _print_title(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _run_adf(y: pd.Series) -> ADFResult:
    y2 = pd.to_numeric(y, errors="coerce").astype(float).dropna()
    if len(y2) < 10:
        raise ValueError("Series too short for ADF test (need ~10+ points)")

    stat, pvalue, used_lag, n_obs, crit_vals, _ = adfuller(y2.values, autolag="AIC")
    return ADFResult(
        test_statistic=float(stat),
        p_value=float(pvalue),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in crit_vals.items()},
    )


def _print_adf_section(y: pd.Series) -> None:
    _print_title("ADF Stationarity Test")
    print("Null hypothesis (H0): series has a unit root (non-stationary).")
    print("If p-value < 0.05, reject H0 (evidence of stationarity).\n")

    res_level = _run_adf(y)
    print("Level series:")
    print(f"  Test statistic: {res_level.test_statistic:.4f}")
    print(f"  p-value:        {res_level.p_value:.4f}")
    print(f"  used lag:       {res_level.used_lag}")
    print(f"  n obs:          {res_level.n_obs}")
    for k, v in res_level.critical_values.items():
        print(f"  critical {k}:   {v:.4f}")

    y_diff = y.diff().dropna()
    if len(y_diff) >= 10:
        res_diff = _run_adf(y_diff)
        print("\nFirst-differenced series (Δy):")
        print(f"  Test statistic: {res_diff.test_statistic:.4f}")
        print(f"  p-value:        {res_diff.p_value:.4f}")
        for k, v in res_diff.critical_values.items():
            print(f"  critical {k}:   {v:.4f}")


def _aic_grid_search(
    y: pd.Series,
    *,
    seasonal_period: int = 12,
    p: Iterable[int] = (0, 1, 2),
    d: Iterable[int] = (0, 1),
    q: Iterable[int] = (0, 1, 2),
    P: Iterable[int] = (0, 1),
    D: Iterable[int] = (0, 1),
    Q: Iterable[int] = (0, 1),
    trend: str = "n",
    show_warnings: bool = False,
) -> pd.DataFrame:
    """Run a small SARIMA grid search and return all attempted models with AIC.

    Returns a DataFrame sorted by AIC ascending.
    """

    rows: list[dict] = []

    y2 = pd.to_numeric(y, errors="coerce").astype(float).sort_index()
    if y2.index.freq is None:
        y2 = y2.asfreq("ME")

    combos = list(itertools.product(p, d, q, P, D, Q))
    for idx, (pi, di, qi, Pi, Di, Qi) in enumerate(combos, start=1):
        order = (int(pi), int(di), int(qi))
        seasonal_order = (int(Pi), int(Di), int(Qi), int(seasonal_period))

        try:
            with warnings.catch_warnings():
                if not show_warnings:
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)

                model = SarimaForecaster(
                    spec=SarimaSpec(
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=trend,
                    ),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model.fit(y2)

                # statsmodels AIC is available on the underlying results
                aic = float(model._results.aic)  # type: ignore[attr-defined]

            if np.isfinite(aic):
                rows.append(
                    {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "trend": trend,
                        "aic": aic,
                    }
                )
        except Exception:
            # Some combinations fail to converge or error; skip.
            continue

        if idx in (1, 25, 50, 100, len(combos)):
            print(f"  Tried {idx}/{len(combos)} candidate specs...")

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("aic", ascending=True).reset_index(drop=True)
    return out


def _print_model_selection_section(
    y: pd.Series,
    *,
    seasonal_period: int,
    allow_seasonal_diff: bool,
    show_warnings: bool,
) -> SarimaSpec:
    _print_title("Model Selection (AIC Grid Search)")

    print("Evaluating a small set of SARIMA specifications and select the model")
    print(
        "with the lowest AIC (Akaike Information Criterion), balancing fit vs complexity."
    )
    print(
        "Lower AIC indicates a better trade-off (not a guarantee of best future accuracy).\n"
    )

    D = (0, 1) if allow_seasonal_diff else (0,)
    if not allow_seasonal_diff:
        print("Note: seasonal differencing (D=1) disabled due to limited history.")
        print("This improves stability when there are < ~24 monthly observations.\n")

    results = _aic_grid_search(
        y,
        seasonal_period=seasonal_period,
        p=(0, 1, 2),
        d=(0, 1),
        q=(0, 1, 2),
        P=(0, 1),
        D=D,
        Q=(0, 1),
        trend="n",
        show_warnings=show_warnings,
    )

    if results.empty:
        raise RuntimeError(
            "Grid search produced no valid models; cannot select a spec."
        )

    top_n = min(10, len(results))
    print("\nTop candidate models by AIC:")
    display = results.head(top_n).copy()
    display["order"] = display["order"].astype(str)
    display["seasonal_order"] = display["seasonal_order"].astype(str)
    print(display.to_string(index=False))

    best = results.iloc[0]
    best_spec = SarimaSpec(
        order=tuple(best["order"]),
        seasonal_order=tuple(best["seasonal_order"]),
        trend=str(best["trend"]),
    )

    print("\nSelected model (lowest AIC):")
    print(f"  order:          {best_spec.order}")
    print(f"  seasonal_order: {best_spec.seasonal_order}")
    print(f"  trend:          {best_spec.trend}")

    print("\nJustification:")
    print("- AIC-based selection chooses a parsimonious model that fits well without")
    print("  adding unnecessary parameters.")
    print(
        "- Seasonal period is fixed at 12 to represent annual seasonality in monthly data."
    )
    if not allow_seasonal_diff:
        print(
            "- Seasonal differencing is disabled to avoid over-differencing and instability"
        )
        print("  on short monthly histories.")

    return best_spec


def _print_fit_and_forecast_section(
    y: pd.Series,
    *,
    spec: SarimaSpec,
    months_ahead: int,
    conf_level: float,
    show_warnings: bool,
) -> None:
    _print_title("Model Fit")

    model = SarimaForecaster(
        spec=spec, enforce_stationarity=False, enforce_invertibility=False
    )
    with warnings.catch_warnings():
        if not show_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(y)

    print("Fitted SARIMAX model summary:")
    print(model._results.summary())  # type: ignore[attr-defined]

    _print_title(f"Forecast ({months_ahead} months ahead, {int(conf_level*100)}% CI)")
    fc = model.forecast(months_ahead, conf_level=conf_level)

    print("Forecast preview:")
    print(fc.head(min(months_ahead, 12)).to_string())


def _default_csv_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
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


def _normalise_provider_code(value: object) -> str:
    s = str(value).strip().upper()
    if s == "" or s == "NAN":
        return ""
    # Many NHS-style provider codes are strings that may contain leading zeros.
    # Normalising improves matching when users type e.g. "042" vs "42".
    if s.isdigit():
        s2 = s.lstrip("0")
        return s2 if s2 != "" else "0"
    return s


def _suggest_metric_columns(df: pd.DataFrame, *, date_col: str) -> list[str]:
    cols: list[str] = []

    for c in df.columns:
        if c == date_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))

    # Fall back to “any” columns if nothing is numeric yet (e.g., strings that
    # need coercion). This is still helpful for the user.
    return (
        sorted(cols) if cols else sorted([str(c) for c in df.columns if c != date_col])
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a SARIMA console report (ADF test, AIC selection, fit, forecast)."
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
        "--agg",
        type=str,
        default="sum",
        choices=["sum", "mean", "median", "count"],
        help="Monthly aggregation method (default: sum)",
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
        help="Confidence level for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=12,
        help="Seasonal period for monthly data (default: 12)",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show statsmodels convergence/starting-parameter warnings",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List candidate metric columns from the CSV and exit",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 2

    if len(sys.argv) == 1:
        print(
            f"No arguments provided; using defaults: --csv {csv_path} --metric {args.metric}"
        )

    _print_title("NHFT SARIMA Forecasting Console Report (Archived)")
    print(f"Data source: {csv_path}")
    print(f"Metric:      {args.metric}")
    print(f"Date col:    {args.date_col}")
    print(f"Agg:         {args.agg}")

    df = pd.read_csv(csv_path)

    provider_col = _find_provider_column(df)
    service_col = _find_service_line_column(df)
    provider_selector = (PROVIDER_CODE_CURRENT or "").strip()
    if provider_selector:
        if provider_col is None:
            _print_title("Provider filter requested, but column not found")
            print(f"Requested PROVIDER_CODE_CURRENT={provider_selector!r}")
            print("No provider column was found in the CSV.")
            print("Expected a column like: provider_code_current")
            return 2

        before = len(df)

        target = _normalise_provider_code(provider_selector)
        series_norm = df[provider_col].map(_normalise_provider_code)
        df = df[series_norm == target]
        after = len(df)

        _print_title("Provider Filter")
        print(f"Provider column: {provider_col}")
        print(f"Selected provider: {provider_selector}")
        if service_col is not None and after > 0:
            service_lines = (
                df[service_col]
                .astype(str)
                .replace("nan", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            service_lines = sorted({s.strip() for s in service_lines if str(s).strip()})
            if len(service_lines) == 1:
                print(f"Service line: {service_lines[0]}")
            elif len(service_lines) > 1:
                print(f"Service lines ({len(service_lines)}):")
                for s in service_lines:
                    print(f"- {provider_selector} - {s}")
        print(f"Rows: {before} -> {after}")
        if after == 0:
            print("No rows matched the selected provider; nothing to model.")

            # Help the user pick a valid provider code.
            available = (
                pd.Series(series_norm.unique())
                .dropna()
                .astype(str)
                .loc[lambda s: s.str.strip().ne("")]
                .sort_values()
                .tolist()
            )
            if available:
                print("\nAvailable provider codes (normalised; first 20):")
                for code in available[:20]:
                    print(f"- {code}")
            if provider_col is not None and service_col is not None and not df.empty:
                # (df is empty here, but keep logic symmetrical)
                pass
            return 2
    else:
        _print_title("Provider Filter")
        print("Selected provider: ALL")
        if provider_col is not None and service_col is not None and not df.empty:
            pairs = (
                df[[provider_col, service_col]]
                .copy()
                .astype(str)
                .replace("nan", pd.NA)
                .dropna()
                .drop_duplicates()
            )
            n_providers = pairs[provider_col].nunique()
            n_pairs = len(pairs)
            print(f"Providers in scope: {n_providers}")
            print(f"Provider-service combinations: {n_pairs}")

            sample_n = min(10, n_pairs)
            if sample_n > 0:
                print(f"\nSample provider-service pairs (first {sample_n}):")
                pairs = pairs.sort_values([provider_col, service_col]).head(sample_n)
                for _, row in pairs.iterrows():
                    print(f"- {row[provider_col]} - {row[service_col]}")
        elif provider_col is not None and not df.empty:
            print(f"Providers in scope: {df[provider_col].nunique()}")

    if args.list_metrics:
        _print_title("Candidate metric columns")
        for name in _suggest_metric_columns(df, date_col=args.date_col):
            print(f"- {name}")
        return 0

    if args.metric not in df.columns:
        _print_title("Metric column not found")
        print(f"Requested metric: {args.metric}")
        print(f"Available columns: {', '.join([str(c) for c in df.columns])}")
        print("\nTry one of these likely metric columns:")
        for name in _suggest_metric_columns(df, date_col=args.date_col):
            print(f"- {name}")
        print("\nExample:")
        candidate = _suggest_metric_columns(df, date_col=args.date_col)[0]
        print(
            "python src/forecasting/archived_sarima/sarima_report.py "
            f"--metric {candidate}"
        )
        return 2

    y = build_monthly_series(
        df,
        args.metric,
        date_col=args.date_col,
        agg=args.agg,  # type: ignore[arg-type]
        fill_missing="zero",
        freq="ME",
    )

    _print_title("Monthly Series Overview")
    print(f"Observations: {len(y)}")
    print(f"Date range:   {y.index.min().date()} to {y.index.max().date()}")
    print(f"Frequency:    {y.index.freqstr or 'None'}")
    print("\nLast 6 months:")
    print(y.tail(6).to_string())

    _print_adf_section(y)

    allow_seasonal_diff = len(y) >= 24
    best_spec = _print_model_selection_section(
        y,
        seasonal_period=args.seasonal_period,
        allow_seasonal_diff=allow_seasonal_diff,
        show_warnings=args.show_warnings,
    )

    _print_fit_and_forecast_section(
        y,
        spec=best_spec,
        months_ahead=args.months_ahead,
        conf_level=args.conf_level,
        show_warnings=args.show_warnings,
    )

    _print_title("Notes")
    print("- This report uses a small AIC grid search intended to be explainable and")
    print(
        "  computationally manageable. For formal model validation, consider a rolling"
    )
    print("  backtest and compare against a seasonal-naive baseline.")
    print("- Forecast intervals are model-based under SARIMA assumptions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
