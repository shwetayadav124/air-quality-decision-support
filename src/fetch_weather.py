"""Download daily weather for every city in data/cities.csv (RUN THIS ON YOUR OWN COMPUTER).

Uses the free Open-Meteo geocoding + ERA5 archive APIs (no API key).
Output: data/weather_daily.csv  with columns city,date,temp_mean,precip_sum,wind_max,rh_mean,pressure_mean
"""
import time
import sys
import requests
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
cities = pd.read_csv(DATA / "cities.csv")["city"].tolist()

GEO = "https://geocoding-api.open-meteo.com/v1/search"
ARCH = "https://archive-api.open-meteo.com/v1/archive"
VARS = {
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precip_sum",
    "wind_speed_10m_max": "wind_max",
    "relative_humidity_2m_mean": "rh_mean",
    "surface_pressure_mean": "pressure_mean",
}
ALIAS = {"Thiruvananthapuram": "Trivandrum", "Rajamahendravaram": "Rajahmundry"}
# The geocoder picks the wrong place for some names (e.g. "Gurgaon" -> a village in Maharashtra), so pin them.
MANUAL = {"Gurugram": (28.46, 77.03, "Gurugram (manual)", "Haryana")}


def get(url, params, tries=5):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (i + 1))        # back off on rate limits
    raise RuntimeError(f"{url} failed: {r.status_code} {r.text[:200]}")


frames, report = [], []
for c in cities:
    if c in MANUAL:
        la, lo, nm, st = MANUAL[c]
        g = {"latitude": la, "longitude": lo, "name": nm, "admin1": st}
    else:
        g = get(GEO, {"name": ALIAS.get(c, c), "count": 10, "country_code": "IN", "language": "en"}).get("results") or []
        if not g:
            print("!! could not geocode", c); continue
        g = g[0]
    j = get(ARCH, {"latitude": g["latitude"], "longitude": g["longitude"],
                   "start_date": "2017-01-01", "end_date": "2020-12-31",
                   "daily": ",".join(VARS), "timezone": "Asia/Kolkata"})["daily"]
    df = pd.DataFrame(j).rename(columns={"time": "date", **VARS})
    df.insert(0, "city", c)
    frames.append(df)
    report.append((c, g.get("name"), g.get("admin1"), round(g["latitude"], 2), round(g["longitude"], 2)))
    print(f"ok {c:25s} -> {g.get('name')}, {g.get('admin1')}")
    time.sleep(1.5)

pd.concat(frames).to_csv(DATA / "weather_daily.csv", index=False)
pd.DataFrame(report, columns=["city", "matched_name", "state", "lat", "lon"]).to_csv(DATA / "weather_geocode_check.csv", index=False)
print("saved data/weather_daily.csv  -> now rerun: python src/prepare.py")
