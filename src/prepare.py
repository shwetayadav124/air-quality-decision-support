"""Step 1: clean the CPCB daily AQI bulletins, select well-covered cities, merge weather if present.

Input : data/aqi_master_raw.csv   (CPCB daily AQI bulletins, from urbanemissions-info/AQI_bulletins)
        data/weather_daily.csv    (optional; made by src/fetch_weather.py)
Output: data/panel.parquet, data/cities.csv
"""
import numpy as np
import pandas as pd
from common import *

raw = pd.read_csv(DATA / "aqi_master_raw.csv", low_memory=False)
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw["aqi"] = pd.to_numeric(raw["Index Value"], errors="coerce")
raw = raw.dropna(subset=["date", "aqi", "City"])
raw["City"] = raw["City"].str.strip()
raw = raw[(raw.date >= START) & (raw.date <= END)]

# prominent pollutant -> single clean label
pp = raw["Prominent Pollutant"].fillna("").str.upper().str.replace("OZONE", "O3")
raw["pollutant"] = pp.str.split(",").str[0].str.strip().replace({"": "UNKNOWN"})
raw.loc[~raw.pollutant.isin(["PM10", "PM2.5", "NO2", "SO2", "CO", "O3"]), "pollutant"] = "OTHER"

# duplicates (same city & date): average AQI, keep first pollutant
agg = raw.groupby(["City", "date"], as_index=False).agg(aqi=("aqi", "mean"), pollutant=("pollutant", "first"))

# select cities with good coverage overall and in the lockdown window (2020) and its 2019 placebo window
def cov(df, a, b):
    n = (b - a).days + 1
    return df[(df.date >= a) & (df.date <= b)].groupby("City").date.nunique() / n

c_all = cov(agg, START, END)
c_20 = cov(agg, LOCKDOWN_START, LOCKDOWN_END)
c_19 = cov(agg, LOCKDOWN_START - pd.DateOffset(years=1), LOCKDOWN_END - pd.DateOffset(years=1))
keep = c_all[(c_all >= 0.85) & (c_20.reindex(c_all.index) >= 0.85) & (c_19.reindex(c_all.index) >= 0.85)].index
print(f"cities with enough coverage: {len(keep)} of {agg.City.nunique()}")

grid = pd.MultiIndex.from_product([sorted(keep), pd.date_range(START, END)], names=["City", "date"])
panel = agg[agg.City.isin(keep)].set_index(["City", "date"]).reindex(grid).reset_index()
panel["observed"] = panel.aqi.notna()
# fill short gaps (<=3 days) only; longer gaps stay missing
panel["aqi"] = panel.groupby("City").aqi.transform(lambda s: s.interpolate(limit=3, limit_area="inside"))
panel["pollutant"] = panel.groupby("City").pollutant.transform(lambda s: s.ffill(limit=3))
panel = calendar_features(panel)
panel = panel.rename(columns={"City": "city"})

wf = DATA / "weather_daily.csv"
if wf.exists():
    w = pd.read_csv(wf, parse_dates=["date"])
    panel = panel.merge(w, on=["city", "date"], how="left")
    print("weather merged; coverage:", panel[WEATHER_COLS].notna().mean().round(3).to_dict())
else:
    print("NO weather file found -> panel is calendar-only (results will NOT be weather-adjusted)")

panel.to_parquet(DATA / "panel.parquet", index=False)
pd.DataFrame({"city": sorted(keep)}).to_csv(DATA / "cities.csv", index=False)
print(panel.shape, panel.date.min().date(), panel.date.max().date())
