"""Calendar rules, no-lookahead guarantees, and the official-forecast correction."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from demand import official
from demand.calendar import easter, features, holidays
from demand.weekahead import design, training_rows


@pytest.mark.parametrize("year, expected", [(2019, dt.date(2019, 4, 21)), (2024, dt.date(2024, 3, 31)), (2025, dt.date(2025, 4, 20)), (2026, dt.date(2026, 4, 5))])
def test_easter(year, expected):
    assert easter(year) == expected


def test_national_holidays_2025():
    h = holidays(2025)
    assert len(h) == 9
    assert dt.date(2025, 5, 29) in h  # Ascension
    assert dt.date(2025, 6, 9) in h   # Whit Monday
    assert dt.date(2017, 10, 31) in holidays(2017)  # one-off Reformation Day 500th anniversary


def test_bridge_day_and_regional_holiday():
    f = features(pd.date_range("2025-05-29", "2025-06-20"))
    assert bool(f.loc["2025-05-30", "bridge_day"])           # Friday after Ascension
    assert bool(f.loc["2025-06-19", "holiday_corpus_christi"])
    assert not bool(f.loc["2025-06-19", "holiday_national"])


def synthetic_daily(days=60):
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({"actual_mwh": np.arange(1, days + 1, dtype=float) * 1000, "forecast_mwh": np.nan}, index=idx)


def test_features_only_use_data_known_on_the_origin_morning():
    daily = synthetic_daily()
    df = design(daily)
    y = daily.actual_mwh
    for r in df.sample(40, random_state=0).itertuples():
        known = y[y.index <= r.origin - pd.Timedelta(days=1)]
        assert r.level == pytest.approx(known.tail(7).mean())          # level = mean of t-7..t-1
        ref = r.target_date - pd.Timedelta(days=7 if r.h <= 6 else 14)
        assert ref <= r.origin - pd.Timedelta(days=1)                   # reference day already observed
        assert r.same_weekday == pytest.approx(y[ref])


def test_training_rows_never_include_unobserved_targets():
    df = design(synthetic_daily(400))
    first = pd.Timestamp("2025-10-01")
    tr = training_rows(df, first)
    assert len(tr) > 0
    assert tr.target_date.max() < first


def test_correction_uses_only_last_year():
    rows = []
    for year, bias in [(2023, 10.0), (2024, -10.0)]:
        for day in range(1, 4):
            for hour in range(24):
                actual = 100.0
                rows.append({"start_local": f"{year}-04-{day:02d}T{hour:02d}:00:00+02:00", "actual_mwh": actual,
                             "forecast_mwh": actual * (1 + bias / 100), "solar_mwh": 0.0})
    h = pd.DataFrame(rows)
    h["date"] = h.start_local.str[:10]
    h["year"] = h.start_local.str[:4].astype(int)
    h["month"] = h.start_local.str[5:7].astype(int)
    h["hour"] = h.start_local.str[11:13].astype(int)
    h["err_pct"] = 100 * (h.forecast_mwh - h.actual_mwh) / h.actual_mwh
    c = official.corrected(h)
    assert set(c.year) == {2024}                       # 2023 has no prior year, so it is not corrected
    assert c.bias_pct.to_numpy() == pytest.approx(10.0)  # 2024 is corrected with 2023's bias, not its own
    assert c.forecast_corrected_mwh.iloc[0] == pytest.approx(90.0 / 1.10)
