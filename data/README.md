# Data (not committed - too large)

| File | Where it comes from |
|---|---|
| `aqi_master_raw.csv` | CPCB daily city AQI bulletins, scraped by urbanemissions.info. Download `data/Processed/AllIndiaBulletins_Master.csv` from https://github.com/urbanemissions-info/AQI_bulletins and save it here as `aqi_master_raw.csv`. |
| `weather_daily.csv` | Run `python src/fetch_weather.py` (needs internet; free Open-Meteo API, no key). |

Then run `python src/prepare.py`, which writes `panel.parquet` and `cities.csv` in this folder.
