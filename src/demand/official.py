"""Audit the official day-ahead load forecast (SMARD 411) against realised load (SMARD 410), and test
a simple correction.

Audit: error by year, by hour of day x month, and against the solar share of load.
Correction: divide each hourly forecast by (1 + bias), where bias is the mean percentage error for the
same hour and calendar month one year earlier. Only data a forecaster has on the day before is used:
the table for a month in year Y comes entirely from year Y-1. Two variants, fixed before testing:
  full       correct every hour
  selective  correct only (month, hour) cells whose bias last year was at least MIN_BIAS_PCT in size

Error convention everywhere: (forecast - actual) / actual. Positive = over-forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LAST_FULL_DAY = "2026-10-18"
MIN_BIAS_PCT = 3.0


def load_hourly(path: Path = DATA / "smard_hourly.csv") -> pd.DataFrame:
    h = pd.read_csv(path)
    h["date"] = h.start_local.str[:10]
    h = h[h.date <= LAST_FULL_DAY].dropna(subset=["actual_mwh", "forecast_mwh"]).copy()
    h["year"] = h.start_local.str[:4].astype(int)
    h["month"] = h.start_local.str[5:7].astype(int)
    h["hour"] = h.start_local.str[11:13].astype(int)
    h["err_pct"] = 100 * (h.forecast_mwh - h.actual_mwh) / h.actual_mwh
    return h


def audit(h: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_year = h.groupby("year").agg(hours=("err_pct", "size"), bias_pct=("err_pct", "mean"),
                                    mape_pct=("err_pct", lambda s: s.abs().mean())).round(2).reset_index()
    recent = h[h.year >= h.year.max() - 1]
    heat = recent.pivot_table(index="hour", columns="month", values="err_pct", aggfunc="mean").round(1)
    mid = h[h.hour.between(10, 15)].copy()
    mid["solar_share_pct"] = 100 * mid.solar_mwh / mid.actual_mwh
    bins = [-1, 5, 15, 30, 45, 60, 1000]
    labels = ["0-5", "5-15", "15-30", "30-45", "45-60", "60+"]
    mid["solar_bin"] = pd.cut(mid.solar_share_pct, bins, labels=labels)
    solar = mid.groupby("solar_bin", observed=True).err_pct.agg(hours="size", mean_error_pct="mean").round(2).reset_index()
    corr = pd.DataFrame({"hours": [len(mid)], "corr_error_vs_solar_share": [round(mid.err_pct.corr(mid.solar_share_pct), 3)],
                         "years": [f"{h.year.min()}-{h.year.max()}"]})
    return {"official_by_year": by_year, "official_bias_hour_month": heat.reset_index(), "official_error_vs_solar": solar,
            "official_solar_correlation": corr}


def corrected(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly forecast corrected with last year's same-month, same-hour bias. Rows without a prior year are dropped."""
    table = h.groupby(["year", "month", "hour"]).err_pct.mean().rename("bias_pct").reset_index()
    table["year"] += 1  # last year's table is used this year
    out = h.merge(table, on=["year", "month", "hour"], how="inner")
    out["forecast_corrected_mwh"] = out.forecast_mwh / (1 + out.bias_pct / 100)
    selective = out.bias_pct.abs() >= MIN_BIAS_PCT
    out["forecast_selective_mwh"] = np.where(selective, out.forecast_corrected_mwh, out.forecast_mwh)
    return out


def correction_results(h: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c = corrected(h)
    c["err_corr_pct"] = 100 * (c.forecast_corrected_mwh - c.actual_mwh) / c.actual_mwh
    c["err_sel_pct"] = 100 * (c.forecast_selective_mwh - c.actual_mwh) / c.actual_mwh
    hourly = c.groupby("year").agg(hours=("err_pct", "size"),
                                   official_mape_pct=("err_pct", lambda s: s.abs().mean()),
                                   corrected_mape_pct=("err_corr_pct", lambda s: s.abs().mean()),
                                   selective_mape_pct=("err_sel_pct", lambda s: s.abs().mean()),
                                   official_bias_pct=("err_pct", "mean"),
                                   corrected_bias_pct=("err_corr_pct", "mean")).round(2).reset_index()
    daily = c.groupby("date").agg(actual=("actual_mwh", "sum"), official=("forecast_mwh", "sum"),
                                  corrected=("forecast_corrected_mwh", "sum"), selective=("forecast_selective_mwh", "sum"),
                                  hours=("hour", "size"))
    daily = daily[daily.hours >= 23]
    daily["year"] = daily.index.str[:4].astype(int)
    for col in ("official", "corrected", "selective"):
        daily[f"{col}_ape"] = 100 * (daily[col] - daily.actual).abs() / daily.actual
    daily_t = daily.groupby("year").agg(days=("actual", "size"), official_mape_pct=("official_ape", "mean"),
                                        corrected_mape_pct=("corrected_ape", "mean"),
                                        selective_mape_pct=("selective_ape", "mean")).round(2).reset_index()
    c["midday_sunny_season"] = c.hour.between(10, 15) & c.month.between(4, 8)
    seg = c.groupby("midday_sunny_season").agg(hours=("err_pct", "size"), official_mape_pct=("err_pct", lambda s: s.abs().mean()),
                                               corrected_mape_pct=("err_corr_pct", lambda s: s.abs().mean()),
                                               selective_mape_pct=("err_sel_pct", lambda s: s.abs().mean())).round(2).reset_index()
    return {"correction_hourly_by_year": hourly, "correction_daily_by_year": daily_t, "correction_midday_apr_aug": seg}


def hit_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Share of cases where b is closer to zero than a."""
    return float(np.mean(np.abs(b) < np.abs(a)))
