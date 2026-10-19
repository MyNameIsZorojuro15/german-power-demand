"""Download German electricity consumption from SMARD (Bundesnetzagentur) into data/*.csv.

Series (confirmed on smard.de, Marktdaten, 2026-10-19; source ENTSO-E):
  410  Realised consumption, total (grid load), MWh per interval
  411  Forecast consumption for the following day, published by 10:00 the day before
  4068 Realised generation, photovoltaics (hourly file only), MWh per interval
Licence: CC BY 4.0, credit "Bundesnetzagentur | SMARD.de".

Two resolutions:
  hourly  2023-01-01 onwards (weekly files): for the hour-of-day view of the official forecast
  daily   2015-01-01 onwards (yearly files): for modelling daily demand

Usage: python -m demand.fetch
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BASE = "https://www.smard.de/app/chart_data"
SERIES = {410: "actual_mwh", 411: "forecast_mwh"}
TZ = "Europe/Berlin"
UA = {"User-Agent": "Mozilla/5.0 (portfolio analysis; non-commercial)"}


def _get(url: str) -> dict:
    for attempt in range(4):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url} -> {r.status_code}")


def series(filter_id: int, resolution: str, since: dt.date) -> pd.Series:
    stamps = _get(f"{BASE}/{filter_id}/DE/index_{resolution}.json")["timestamps"]
    cutoff = int(pd.Timestamp(since, tz=TZ).timestamp() * 1000)
    # A file starting just before the cutoff still holds data after it.
    keep = [s for i, s in enumerate(stamps) if i + 1 == len(stamps) or stamps[i + 1] > cutoff]
    rows = []
    for s in keep:
        rows += _get(f"{BASE}/{filter_id}/DE/{filter_id}_DE_{resolution}_{s}.json")["series"]
        time.sleep(0.15)
    out = pd.Series({ts: v for ts, v in rows}, dtype="float64").sort_index()
    out.index = pd.to_datetime(out.index, unit="ms", utc=True).tz_convert(TZ)
    return out[out.index >= pd.Timestamp(since, tz=TZ)]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for resolution, since, name in [("day", dt.date(2015, 1, 1), "daily"), ("hour", dt.date(2023, 1, 1), "hourly")]:
        wanted = {**SERIES, 4068: "solar_mwh"} if resolution == "hour" else SERIES
        frame = pd.DataFrame({col: series(fid, resolution, since) for fid, col in wanted.items()})
        frame.index.name = "start_local"
        frame.to_csv(DATA / f"smard_{name}.csv")
        print({name: len(frame), "first": str(frame.index.min()), "last": str(frame.index.max()),
               "missing_actual": int(frame.actual_mwh.isna().sum()), "missing_forecast": int(frame.forecast_mwh.isna().sum())})
    print({"seconds": round(time.perf_counter() - t0, 1)})


if __name__ == "__main__":
    main()
