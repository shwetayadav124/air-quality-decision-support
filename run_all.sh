#!/usr/bin/env bash
# Full pipeline. Run from the project folder.
set -e
pip install -r requirements.txt
python src/prepare.py                       # clean AQI, pick cities (writes data/cities.csv)
[ -f data/weather_daily.csv ] || python src/fetch_weather.py   # needs internet; free Open-Meteo, no key
python src/prepare.py                       # re-run so weather is merged
python src/forecast.py                      # Module 2: forecast + alerts
python src/impact.py                        # Module 1: lockdown impact + placebo + CI
streamlit run app.py                        # dashboard
