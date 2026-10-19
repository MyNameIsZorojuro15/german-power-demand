"""German public holidays and calendar features. No external holiday package: the rules are short.

National holidays apply in all 16 states. Regional ones are kept as separate flags because they move
national demand by less: Corpus Christi and All Saints cover the large southern and western states
(BW, BY, NW, RP, SL, and HE for Corpus Christi); Reformation Day the northern and eastern states;
Epiphany BW, BY and ST. Source: the public-holiday laws of the federal states.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def holidays(year: int) -> dict[dt.date, str]:
    e = easter(year)
    national = {
        dt.date(year, 1, 1): "new_year", e - dt.timedelta(days=2): "good_friday", e + dt.timedelta(days=1): "easter_monday",
        dt.date(year, 5, 1): "labour_day", e + dt.timedelta(days=39): "ascension", e + dt.timedelta(days=50): "whit_monday",
        dt.date(year, 10, 3): "unity_day", dt.date(year, 12, 25): "christmas_1", dt.date(year, 12, 26): "christmas_2",
    }
    if year == 2017:
        national[dt.date(2017, 10, 31)] = "reformation_500"  # one-off national holiday
    return national


REGIONAL = {
    "corpus_christi": lambda y: easter(y) + dt.timedelta(days=60),
    "all_saints": lambda y: dt.date(y, 11, 1),
    "reformation_day": lambda y: dt.date(y, 10, 31),
    "epiphany": lambda y: dt.date(y, 1, 6),
}


def features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features for each date: weekday, holidays, bridge days, Christmas period, season."""
    d = pd.DatetimeIndex(dates).normalize()
    years = sorted(set(d.year) | {y + 1 for y in d.year} | {y - 1 for y in d.year})
    nat = {k for y in years for k in holidays(y)}
    df = pd.DataFrame(index=d)
    df["weekday"] = d.weekday
    days = [x.date() for x in d]
    df["holiday_national"] = [x in nat for x in days]
    for name, rule in REGIONAL.items():
        dates_r = {rule(y) for y in years}
        df[f"holiday_{name}"] = [(x in dates_r) and (x not in nat) for x in days]
    # A bridge day: a Monday before a holiday Tuesday, or a Friday after a holiday Thursday.
    df["bridge_day"] = [
        (x.weekday() == 0 and (x + dt.timedelta(days=1)) in nat) or (x.weekday() == 4 and (x - dt.timedelta(days=1)) in nat)
        for x in days
    ]
    df["christmas_period"] = [(x.month == 12 and x.day >= 24) or (x.month == 1 and x.day <= 1) for x in days]
    doy = d.dayofyear.to_numpy()
    for k in (1, 2, 3):
        df[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return df
