"""Week-ahead forecast of German daily electricity demand, with a rolling-origin backtest.

Set-up, chosen to match how a planner works:
  * the forecast is made on the morning of day t, when realised load is known up to day t-1;
  * it covers days t+1 ... t+7 (horizon h = 1..7); h = 1 is the same target as the official
    day-ahead forecast, which is therefore a fair benchmark at h = 1 only.

Models:
  seasonal_naive  the last known value for the same weekday
  ridge           ridge regression on log(target / recent level), where the recent level is the mean of
                  days t-7..t-1; features: the target's weekday, national and regional holidays, bridge
                  days, the Christmas period, seasonal Fourier terms, the horizon, and the last known
                  same-weekday value relative to the level. Trained on the previous 3 years of origins
                  only and refitted every 28 days, so no forecast ever sees its own target.

Usage: python -m demand.weekahead   (writes reports/tables/weekahead_*.csv)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from demand.calendar import features

# numpy 2.0 with Apple's Accelerate BLAS raises spurious "divide by zero / overflow encountered in matmul"
# warnings on finite inputs. Checked: predictions are finite and match a closed-form ridge solution to 1e-15.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TABLES = ROOT / "reports" / "tables"
HORIZONS = range(1, 8)
TRAIN_YEARS = 3
REFIT_DAYS = 28
BACKTEST_START = "2024-01-01"
LAST_FULL_DAY = "2026-10-18"


def load_daily(path: Path = DATA / "smard_daily.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d.start_local.str[:10])
    d = d.set_index("date")[["actual_mwh", "forecast_mwh"]]
    d = d[d.index <= LAST_FULL_DAY]
    return d.asfreq("D")


def design(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per (origin, horizon) with features known on the origin morning and the target."""
    y = daily.actual_mwh
    level = y.shift(1).rolling(7, min_periods=7).mean()  # mean of t-7..t-1, as seen on the morning of t
    rows = []
    for h in HORIZONS:
        target_dates = y.index + pd.Timedelta(days=h)
        k = 1 if h <= 6 else 2
        same_wd = y.shift(7 * k - h)  # value on target - 7k, indexed by origin
        frame = pd.DataFrame({
            "origin": y.index, "h": h, "target_date": target_dates,
            "level": level.values, "same_weekday": same_wd.values,
            "target": y.reindex(target_dates).values,
            "official": daily.forecast_mwh.reindex(target_dates).values,
        })
        cf = features(target_dates)
        for col in cf.columns:
            frame[col] = cf[col].values
        rows.append(frame)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["level", "same_weekday"])
    return df


def matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(df["weekday"].astype(int), prefix="wd").astype(float)
    for h in HORIZONS:
        x[f"h{h}"] = (df["h"] == h).astype(float)
    flags = [c for c in df.columns if c.startswith("holiday_")] + ["bridge_day", "christmas_period"]
    for c in flags:
        x[c] = df[c].astype(float)
    for c in [c for c in df.columns if c.startswith(("sin", "cos"))]:
        x[c] = df[c]
    x["log_same_weekday_rel"] = np.log(df["same_weekday"] / df["level"])
    for col in x.columns:
        x[col] = x[col].astype(float)
    return x.reindex(columns=sorted(x.columns))


def training_rows(df: pd.DataFrame, first_origin: pd.Timestamp) -> pd.DataFrame:
    """Rows a model refitted on the morning of first_origin may learn from: targets already observed."""
    return df[(df.origin >= first_origin - pd.DateOffset(years=TRAIN_YEARS)) & (df.target_date < first_origin) & df.target.notna()]


def backtest(daily: pd.DataFrame) -> pd.DataFrame:
    df = design(daily)
    df["seasonal_naive"] = df["same_weekday"]
    df["ridge"] = np.nan
    origins = df.loc[(df.origin >= BACKTEST_START) & df.target.notna(), "origin"].drop_duplicates().sort_values()
    for i in range(0, len(origins), REFIT_DAYS):
        block = origins.iloc[i:i + REFIT_DAYS]
        first = block.iloc[0]
        train = training_rows(df, first)
        model = Ridge(alpha=1.0).fit(matrix(train), np.log(train.target / train.level))
        rows = df.origin.isin(block)
        df.loc[rows, "ridge"] = df.loc[rows, "level"] * np.exp(model.predict(matrix(df[rows])))
    out = df[(df.origin >= BACKTEST_START) & df.target.notna() & df.ridge.notna()].copy()
    return out


def scores(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in bt.groupby("h"):
        for model in ("seasonal_naive", "ridge", "official"):
            if model == "official" and h != 1:
                continue
            e = 100 * (g[model] - g.target) / g.target
            rows.append({"horizon_days": h, "model": model, "n": int(e.notna().sum()), "mape_pct": round(e.abs().mean(), 2),
                         "bias_pct": round(e.mean(), 2), "p90_abs_error_pct": round(e.abs().quantile(0.9), 2)})
    return pd.DataFrame(rows)


def holiday_scores(bt: pd.DataFrame) -> pd.DataFrame:
    special = bt.holiday_national | bt.bridge_day | bt.christmas_period
    rows = []
    for name, g in [("normal days", bt[~special]), ("holidays, bridge days, Christmas", bt[special])]:
        for model in ("seasonal_naive", "ridge"):
            e = 100 * (g[model] - g.target).abs() / g.target
            rows.append({"days": name, "model": model, "n": len(g), "mape_pct": round(e.mean(), 2)})
    return pd.DataFrame(rows)


def interval_quantiles(bt: pd.DataFrame) -> pd.DataFrame:
    """Empirical 10th/90th percentile of the ridge model's log error per horizon, for prediction intervals."""
    e = np.log(bt.target / bt.ridge)
    return bt.assign(e=e).groupby("h").e.quantile([0.1, 0.9]).unstack().rename(columns={0.1: "q10", 0.9: "q90"}).reset_index()


def next_week(daily: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Forecast from the latest origin (the morning after the last full day) with an 80 % interval."""
    df = design(daily.reindex(pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(days=8))))
    origin = daily.index.max() + pd.Timedelta(days=1)
    train = training_rows(df, origin)
    model = Ridge(alpha=1.0).fit(matrix(train), np.log(train.target / train.level))
    now = df[df.origin == origin].copy()
    now["forecast_mwh"] = now.level * np.exp(model.predict(matrix(now)))
    q = interval_quantiles(bt).set_index("h")
    now["low_mwh"] = now.forecast_mwh * np.exp(now.h.map(q.q10))
    now["high_mwh"] = now.forecast_mwh * np.exp(now.h.map(q.q90))
    return now[["origin", "h", "target_date", "forecast_mwh", "low_mwh", "high_mwh", "holiday_national", "bridge_day"]]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    daily = load_daily()
    bt = backtest(daily)
    bt.to_csv(TABLES / "weekahead_backtest.csv", index=False)
    s = scores(bt)
    s.to_csv(TABLES / "weekahead_scores.csv", index=False)
    holiday_scores(bt).to_csv(TABLES / "weekahead_holiday_scores.csv", index=False)
    nw = next_week(daily, bt)
    nw.to_csv(TABLES / "weekahead_next_week.csv", index=False)
    print(s.to_string(index=False))
    print(holiday_scores(bt).to_string(index=False))
    print(nw.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
