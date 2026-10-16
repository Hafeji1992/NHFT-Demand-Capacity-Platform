"""Rolling-origin backtest of the production ETS forecaster.

Purpose
-------
Measures out-of-sample forecast accuracy of the platform's Holt-Winters ETS
(with the per-series AIC grid search) against three simple baselines, using
only information that would have been available at each forecast origin.
Optionally also backtests the archived SARIMA forecaster it replaced.

Method
------
For each service (provider code) the monthly series is built exactly as the
platform builds it. An expanding window is then rolled forward one month at a
time. At each origin the ETS specification is RE-SELECTED on the training data
only (no leakage), the model is fitted, and the next `horizon` months are
forecast and compared with what actually happened.

Baselines: naive (last value), seasonal naive (same month last year) and the
mean of the last 12 months.

SARIMA (--sarima): the archived SARIMA forecaster and its AIC order search
(src/forecasting/archived_sarima). By default the order is selected once per
service, on the first training window only, and the model is refitted at every
origin; the selection never sees data after the first origin, so there is no
leakage. Use --sarima-reselect to re-run the order search at every origin
(much slower). SARIMA fit failures, convergence warnings and fit times are
recorded, and SARIMA is compared with ETS on the folds where both produced a
forecast.

Metrics: WAPE (pooled), MASE (scaled by the in-sample seasonal-naive error),
win rates, error by horizon, empirical coverage of the 95% interval, and
Wilcoxon signed-rank tests on per-service MAE differences.

Run from the repository root, for example:
    python src/tests/ets_backtest.py --end-date 2026-02-28
    python src/tests/ets_backtest.py --end-date 2026-02-28 --sarima --jobs 6
    python src/tests/ets_backtest.py --providers 43 --sarima
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Make `forecasting.*` importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.ets import EtsForecaster, small_grid_search_aic_ets  # noqa: E402
from forecasting.preprocessing import build_monthly_series  # noqa: E402

SEASONAL_PERIOD = 12
MODELS = ("ets", "naive", "snaive", "mean12")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _key(name: str) -> str:
    """Case- and underscore-insensitive column key (PeriodEnd == period_end)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column(df: pd.DataFrame, wanted: str, *, required: bool = True):
    lookup = {_key(c): c for c in df.columns}
    col = lookup.get(_key(wanted))
    if col is None and required:
        raise SystemExit(
            f"Column '{wanted}' not found. Available columns: {list(df.columns)}"
        )
    return col


def normalise_provider_code(value) -> str:
    s = str(value).strip().upper()
    if s.isdigit():
        s = s.lstrip("0") or "0"
    return s


def baseline_forecasts(train: np.ndarray, horizon: int) -> dict:
    n = len(train)
    naive = np.repeat(train[-1], horizon)
    snaive = np.array(
        [train[n - SEASONAL_PERIOD + (k % SEASONAL_PERIOD)] for k in range(horizon)]
    )
    mean12 = np.repeat(train[-SEASONAL_PERIOD:].mean(), horizon)
    return {"naive": naive, "snaive": snaive, "mean12": mean12}


def seasonal_naive_scale(train: np.ndarray) -> float:
    """In-sample mean absolute seasonal-naive error (MASE denominator)."""
    if len(train) <= SEASONAL_PERIOD:
        return np.nan
    scale = np.mean(np.abs(train[SEASONAL_PERIOD:] - train[:-SEASONAL_PERIOD]))
    return float(scale) if scale > 0 else np.nan


def _wilcoxon_p(a: pd.Series, b: pd.Series):
    """Two-sided Wilcoxon signed-rank p-value on paired per-service MAEs."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return np.nan, 0
    diff = (a - b).to_numpy()
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) < 6:
        return np.nan, len(diff)
    return float(wilcoxon(diff).pvalue), len(diff)


# --------------------------------------------------------------------------
# SARIMA (archived model)
# --------------------------------------------------------------------------
def _sarima_select(train_s: pd.Series):
    from forecasting.archived_sarima.sarima import small_grid_search_aic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return small_grid_search_aic(train_s, seasonal_period=SEASONAL_PERIOD)


def _sarima_forecast(train_s: pd.Series, spec, horizon: int):
    """Fit and forecast; return (frame, convergence_warning_raised)."""
    from forecasting.archived_sarima.sarima import SarimaForecaster

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fc = SarimaForecaster(spec=spec).fit(train_s).forecast(horizon, conf_level=0.95)
    conv = any("Convergence" in w.category.__name__ for w in caught)
    return fc, conv


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest_series(
    y: pd.Series,
    *,
    horizon: int,
    min_train: int,
    step: int = 1,
    sarima: bool = False,
    sarima_reselect: bool = False,
):
    """Return (rows, n_folds, n_failures, n_skipped_folds, sarima_meta) for one series."""
    rows, failures, skipped = [], 0, 0
    n = len(y)
    origins = list(range(min_train, n - horizon + 1, step))
    values = y.to_numpy(dtype=float)
    sm = {"attempted": 0, "failed": 0, "conv_warn": 0, "fit_secs": 0.0,
          "select_secs": 0.0, "selections": 0}
    sarima_spec = None

    for t in origins:
        train_s, train = y.iloc[:t], values[:t]
        actual = values[t : t + horizon]
        if train.sum() <= 0 or np.count_nonzero(train[-SEASONAL_PERIOD:]) == 0:
            skipped += 1
            continue
        try:
            t_ets = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = small_grid_search_aic_ets(train_s, seasonal_period=SEASONAL_PERIOD)
                fc = EtsForecaster(spec=spec).fit(train_s).forecast(horizon, conf_level=0.95)
            ets_secs = time.perf_counter() - t_ets
        except Exception:
            failures += 1
            continue

        # SARIMA: same fold, same training data.
        s_fc, s_conv, s_order, s_secs = None, False, "", np.nan
        if sarima:
            sm["attempted"] += 1
            try:
                if sarima_spec is None or sarima_reselect:
                    t_sel = time.perf_counter()
                    sarima_spec = _sarima_select(train_s)
                    sm["select_secs"] += time.perf_counter() - t_sel
                    sm["selections"] += 1
                t_fit = time.perf_counter()
                s_fc, s_conv = _sarima_forecast(train_s, sarima_spec, horizon)
                s_secs = time.perf_counter() - t_fit
                sm["fit_secs"] += s_secs
                if not np.all(np.isfinite(s_fc["yhat"].to_numpy(dtype=float))):
                    raise ValueError("non-finite SARIMA forecast")
                s_order = f"{sarima_spec.order}x{sarima_spec.seasonal_order}"
                sm["conv_warn"] += int(s_conv)
            except Exception:
                sm["failed"] += 1
                s_fc = None

        base = baseline_forecasts(train, horizon)
        scale = seasonal_naive_scale(train)
        spec_txt = f"trend={spec.trend},seasonal={spec.seasonal},damped={spec.damped_trend}"
        for k in range(horizon):
            row = {
                "origin": y.index[t - 1],
                "h": k + 1,
                "target_date": y.index[t + k],
                "actual": actual[k],
                "ets": float(fc["yhat"].iloc[k]),
                "ets_lower": float(fc["yhat_lower"].iloc[k]),
                "ets_upper": float(fc["yhat_upper"].iloc[k]),
                "naive": float(base["naive"][k]),
                "snaive": float(base["snaive"][k]),
                "mean12": float(base["mean12"][k]),
                "mase_scale": scale,
                "spec": spec_txt,
                "ets_secs": ets_secs,
            }
            if sarima:
                ok = s_fc is not None
                row.update(
                    {
                        "sarima": float(s_fc["yhat"].iloc[k]) if ok else np.nan,
                        "sarima_lower": float(s_fc["yhat_lower"].iloc[k]) if ok else np.nan,
                        "sarima_upper": float(s_fc["yhat_upper"].iloc[k]) if ok else np.nan,
                        "sarima_order": s_order,
                        "sarima_conv_warn": bool(s_conv) if ok else np.nan,
                        "sarima_secs": s_secs,
                    }
                )
            rows.append(row)
    return rows, len(origins), failures, skipped, sm


def _run_one(job):
    """Worker for one service; top-level so it can run in a separate process."""
    code, svc, y, opts = job
    rows, f, fail, skip, sm = backtest_series(y, **opts)
    for r in rows:
        r["provider"], r["service_line"] = code, svc
    return code, svc, len(y), rows, f, fail, skip, sm


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def summarise(res: pd.DataFrame) -> dict:
    """Compute all headline statistics from the long results table."""
    out = {}
    for m in MODELS:
        res[f"ae_{m}"] = (res[m] - res["actual"]).abs()

    total_actual = res["actual"].sum()
    out["wape"] = {m: res[f"ae_{m}"].sum() / total_actual if total_actual else np.nan for m in MODELS}

    # per service
    g = res.groupby(["provider", "service_line"], sort=False)
    per = g.apply(
        lambda d: pd.Series(
            {
                **{f"mae_{m}": d[f"ae_{m}"].mean() for m in MODELS},
                "mase_scale": d["mase_scale"].mean(),
                "n_points": len(d),
                "coverage95": ((d["actual"] >= d["ets_lower"]) & (d["actual"] <= d["ets_upper"])).mean(),
            }
        ),
        include_groups=False,
    ).reset_index()
    for m in MODELS:
        per[f"mase_{m}"] = per[f"mae_{m}"] / per["mase_scale"]
    out["per_service"] = per

    out["median_mase"] = {m: per[f"mase_{m}"].median() for m in MODELS}
    out["mean_mase"] = {m: per[f"mase_{m}"].mean() for m in MODELS}

    valid = per.dropna(subset=["mae_ets", "mae_snaive"])
    out["n_services"] = len(per)
    out["win_rate_vs_snaive"] = float((valid["mae_ets"] < valid["mae_snaive"]).mean()) if len(valid) else np.nan
    out["win_rate_vs_mean12"] = float((valid["mae_ets"] < valid["mae_mean12"]).mean()) if len(valid) else np.nan
    out["win_rate_vs_naive"] = float((valid["mae_ets"] < valid["mae_naive"]).mean()) if len(valid) else np.nan

    # by horizon
    by_h = {}
    for h, d in res.groupby("h"):
        a = d["actual"].sum()
        by_h[h] = {
            **{m: d[f"ae_{m}"].sum() / a if a else np.nan for m in MODELS},
            "coverage95": float(((d["actual"] >= d["ets_lower"]) & (d["actual"] <= d["ets_upper"])).mean()),
        }
    out["by_horizon"] = by_h
    out["coverage95"] = float(((res["actual"] >= res["ets_lower"]) & (res["actual"] <= res["ets_upper"])).mean())

    # Wilcoxon on per-service MAE differences
    out["wilcoxon_p"], out["wilcoxon_n"] = _wilcoxon_p(valid["mae_ets"], valid["mae_snaive"])
    out["wilcoxon_mean12_p"], out["wilcoxon_mean12_n"] = _wilcoxon_p(valid["mae_ets"], valid["mae_mean12"])
    return out


def summarise_sarima(res: pd.DataFrame, sm: dict) -> dict:
    """ETS vs SARIMA on the folds where both produced a forecast."""
    out = {"meta": sm}
    common = res.dropna(subset=["sarima"]).copy()
    out["n_points"] = len(common)
    if common.empty:
        return out

    models = ("ets", "sarima", "snaive", "mean12")
    for m in models:
        common[f"ae_{m}"] = (common[m] - common["actual"]).abs()
    a = common["actual"].sum()
    out["wape"] = {m: common[f"ae_{m}"].sum() / a if a else np.nan for m in models}
    out["coverage"] = {
        "ets": float(((common["actual"] >= common["ets_lower"]) & (common["actual"] <= common["ets_upper"])).mean()),
        "sarima": float(((common["actual"] >= common["sarima_lower"]) & (common["actual"] <= common["sarima_upper"])).mean()),
    }

    per = common.groupby(["provider", "service_line"], sort=False).agg(
        mae_ets=("ae_ets", "mean"),
        mae_sarima=("ae_sarima", "mean"),
        mase_scale=("mase_scale", "mean"),
        n_points=("actual", "size"),
    ).reset_index()
    per["mase_ets"] = per["mae_ets"] / per["mase_scale"]
    per["mase_sarima"] = per["mae_sarima"] / per["mase_scale"]
    out["per_service"] = per
    out["n_services"] = len(per)
    out["median_mase"] = {m: per[f"mase_{m}"].median() for m in ("ets", "sarima")}
    out["ets_win_rate"] = float((per["mae_ets"] < per["mae_sarima"]).mean())
    out["wilcoxon_p"], out["wilcoxon_n"] = _wilcoxon_p(per["mae_ets"], per["mae_sarima"])

    folds = res.drop_duplicates(["provider", "origin"])
    out["ets_secs_per_fold"] = float(folds["ets_secs"].mean())
    out["sarima_fit_secs_per_fold"] = sm["fit_secs"] / max(1, sm["attempted"] - sm["failed"])
    out["sarima_select_secs_per_search"] = sm["select_secs"] / max(1, sm["selections"])
    return out


def print_summary(s: dict, meta: dict) -> None:
    pct = lambda x: "n/a" if pd.isna(x) else f"{100 * x:5.1f}%"
    num = lambda x: "n/a" if pd.isna(x) else f"{x:5.2f}"
    print("\n" + "=" * 64)
    print("ETS ROLLING-ORIGIN BACKTEST")
    print("=" * 64)
    print(f"Horizon: {meta['horizon']} months | min training: {meta['min_train']} months | step: {meta['step']}")
    print(f"Services supplied: {meta['n_supplied']} | evaluated: {s['n_services']}")
    print(f"Excluded (too short): {meta['n_short']} | folds skipped (all-zero training): {meta['n_skipped']}")
    print(f"Folds attempted: {meta['n_folds']} | ETS fit failures: {meta['n_failures']}")
    print(f"Forecast points evaluated: {meta['n_points']}")
    print("\nPooled WAPE (lower is better)")
    for m in MODELS:
        print(f"  {m:8s} {pct(s['wape'][m])}")
    print("\nMASE per service (below 1 beats in-sample seasonal naive)")
    for m in MODELS:
        print(f"  {m:8s} median {num(s['median_mase'][m])} | mean {num(s['mean_mase'][m])}")
    print(f"\nETS win rate vs seasonal naive: {pct(s['win_rate_vs_snaive'])}")
    print(f"ETS win rate vs 12-month mean:  {pct(s['win_rate_vs_mean12'])}")
    print(f"ETS win rate vs naive:          {pct(s['win_rate_vs_naive'])}")
    for label, p_key, n_key in (
        ("seasonal naive", "wilcoxon_p", "wilcoxon_n"),
        ("12-month mean", "wilcoxon_mean12_p", "wilcoxon_mean12_n"),
    ):
        if pd.isna(s[p_key]):
            print(f"Wilcoxon (ETS vs {label}, per-service MAE): not computed (n={s[n_key]})")
        else:
            print(f"Wilcoxon (ETS vs {label}, per-service MAE): p = {s[p_key]:.4f} (n={s[n_key]})")
    print("\nBy horizon (pooled WAPE | 95% interval coverage)")
    print("  h   ets     snaive  mean12  naive   coverage")
    for h, r in s["by_horizon"].items():
        print(f"  {h:<3d} {pct(r['ets'])} {pct(r['snaive'])} {pct(r['mean12'])} {pct(r['naive'])} {pct(r['coverage95'])}")
    print(f"\nOverall empirical coverage of the 95% interval: {pct(s['coverage95'])}")
    print("=" * 64)


def print_sarima_summary(s: dict, reselect: bool) -> None:
    pct = lambda x: "n/a" if pd.isna(x) else f"{100 * x:5.1f}%"
    num = lambda x: "n/a" if pd.isna(x) else f"{x:5.2f}"
    sm = s["meta"]
    print("\n" + "=" * 64)
    print("ETS vs SARIMA (archived model)")
    print("=" * 64)
    print(f"Order selection: {'re-selected at every origin' if reselect else 'once per service, on the first training window'}")
    print(f"SARIMA folds attempted: {sm['attempted']} | failed: {sm['failed']} ({pct(sm['failed'] / sm['attempted'] if sm['attempted'] else np.nan)})")
    ok = sm["attempted"] - sm["failed"]
    print(f"Convergence warnings on successful fits: {sm['conv_warn']} of {ok} ({pct(sm['conv_warn'] / ok if ok else np.nan)})")
    if not s.get("n_points"):
        print("No SARIMA forecasts to compare.")
        print("=" * 64)
        return
    print(f"Compared on {s['n_points']} forecast points across {s['n_services']} services (folds where both models produced a forecast)")
    print("\nPooled WAPE on the common folds")
    for m in ("ets", "sarima", "snaive", "mean12"):
        print(f"  {m:8s} {pct(s['wape'][m])}")
    print(f"\nMedian MASE: ets {num(s['median_mase']['ets'])} | sarima {num(s['median_mase']['sarima'])}")
    print(f"ETS win rate vs SARIMA: {pct(s['ets_win_rate'])}")
    if pd.isna(s["wilcoxon_p"]):
        print(f"Wilcoxon (ETS vs SARIMA, per-service MAE): not computed (n={s['wilcoxon_n']})")
    else:
        print(f"Wilcoxon (ETS vs SARIMA, per-service MAE): p = {s['wilcoxon_p']:.4f} (n={s['wilcoxon_n']})")
    print(f"\n95% interval coverage: ets {pct(s['coverage']['ets'])} | sarima {pct(s['coverage']['sarima'])}")
    print("\nTiming")
    print(f"  ETS selection + fit per fold:  {s['ets_secs_per_fold']:.3f}s")
    print(f"  SARIMA fit per fold:           {s['sarima_fit_secs_per_fold']:.3f}s")
    print(f"  SARIMA order search per run:   {s['sarima_select_secs_per_search']:.2f}s")
    print("=" * 64)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Rolling-origin backtest of the ETS forecaster")
    p.add_argument("--csv", default=str(root / "data" / "patient_data.csv"))
    p.add_argument("--metric", default="referrals")
    p.add_argument("--date-col", default="period_end")
    p.add_argument("--provider-col", default="provider_code_current")
    p.add_argument("--service-col", default="service_line")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--min-train", type=int, default=24)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--providers", nargs="*", default=None, help="Provider codes to include (default: all)")
    p.add_argument("--end-date", default=None, help="Last month to include, e.g. 2026-02-28. Use it to drop an incomplete final month")
    p.add_argument("--max-services", type=int, default=None, help="Cap on number of services (for quick trials)")
    p.add_argument("--sarima", action="store_true", help="Also backtest the archived SARIMA forecaster")
    p.add_argument("--sarima-reselect", action="store_true", help="Re-run the SARIMA order search at every origin (slow)")
    p.add_argument("--jobs", type=int, default=1, help="Services to run in parallel (default 1)")
    p.add_argument("--out", default=str(root / "data" / "backtest_ets_results.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    df = pd.read_csv(a.csv)
    metric_col = resolve_column(df, a.metric)
    date_col = resolve_column(df, a.date_col)
    prov_col = resolve_column(df, a.provider_col)
    svc_col = resolve_column(df, a.service_col, required=False)

    df["_prov"] = df[prov_col].map(normalise_provider_code)
    codes = sorted(df["_prov"].unique(), key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
    if a.providers:
        wanted = {normalise_provider_code(p) for p in a.providers}
        codes = [c for c in codes if c in wanted]
        if not codes:
            raise SystemExit(f"None of {sorted(wanted)} found in {prov_col}")
    if a.max_services:
        codes = codes[: a.max_services]

    opts = dict(horizon=a.horizon, min_train=a.min_train, step=a.step,
                sarima=a.sarima, sarima_reselect=a.sarima_reselect)

    # Build every series up front, so workers only receive what they need.
    jobs, n_short = [], 0
    for code in codes:
        sub = df[df["_prov"] == code]
        svc = str(sub[svc_col].mode().iloc[0]) if svc_col and not sub[svc_col].dropna().empty else ""
        y = build_monthly_series(sub, metric_col, date_col=date_col, agg="sum", fill_missing="zero", freq="ME")
        if a.end_date:
            y = y[y.index <= pd.Timestamp(a.end_date)]
        if len(y) < a.min_train + a.horizon:
            n_short += 1
            continue
        jobs.append((code, svc, y, opts))

    all_rows, n_folds, n_fail, n_skip = [], 0, 0, 0
    sm_total = {"attempted": 0, "failed": 0, "conv_warn": 0, "fit_secs": 0.0,
                "select_secs": 0.0, "selections": 0}
    t0 = time.time()

    def _collect(i, result):
        nonlocal n_folds, n_fail, n_skip
        code, svc, months, rows, f, fail, skip, sm = result
        all_rows.extend(rows)
        n_folds += f
        n_fail += fail
        n_skip += skip
        for k in sm_total:
            sm_total[k] += sm[k]
        extra = f" sarima_fails={sm['failed']}" if a.sarima else ""
        print(f"[{i}/{len(jobs)}] {code} {svc[:40]:40s} months={months} folds={f} fails={fail}{extra} ({time.time() - t0:.0f}s)", flush=True)

    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futures = [pool.submit(_run_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                _collect(i, fut.result())
    else:
        for i, j in enumerate(jobs, 1):
            _collect(i, _run_one(j))

    if not all_rows:
        print("No forecasts were evaluated. Check --min-train / --horizon against the series length.")
        return 1

    res = pd.DataFrame(all_rows).sort_values(["provider", "origin", "h"], key=lambda c: c.astype(str)).reset_index(drop=True)
    s = summarise(res)
    meta = dict(
        horizon=a.horizon, min_train=a.min_train, step=a.step, n_supplied=len(codes),
        n_short=n_short, n_skipped=n_skip, n_folds=n_folds, n_failures=n_fail, n_points=len(res),
    )
    print_summary(s, meta)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.sarima:
        ss = summarise_sarima(res, sm_total)
        print_sarima_summary(ss, a.sarima_reselect)
        if "per_service" in ss:
            ss["per_service"].to_csv(out.with_name(out.stem + "_sarima_by_service.csv"), index=False)

    res.drop(columns=[c for c in res.columns if c.startswith("ae_")]).to_csv(out, index=False)
    s["per_service"].to_csv(out.with_name(out.stem + "_by_service.csv"), index=False)
    print(f"\nSaved: {out}\nSaved: {out.with_name(out.stem + '_by_service.csv')}")
    if a.sarima:
        print(f"Saved: {out.with_name(out.stem + '_sarima_by_service.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())