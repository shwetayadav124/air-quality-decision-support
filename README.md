# Air Quality Decision Support
Next-day AQI forecasting, high-risk-day alerts and weather-adjusted policy impact for Indian cities.

**Question:** how much of the air-quality improvement during India's 2020 lockdown was caused by the lockdown itself,
rather than by weather and season?

## Quick start
```bash
pip install -r requirements.txt
# 1. put the AQI file in data/ (see data/README.md), then:
python src/prepare.py            # writes data/cities.csv
python src/fetch_weather.py      # once, with internet -> data/weather_daily.csv
python src/prepare.py            # again, so weather is merged
python src/forecast.py && python src/impact.py
streamlit run app.py
```
(or just `./run_all.sh`)

The dashboard reads `outputs/*.parquet`, which the two scripts create; those files and the raw data are not committed because of size.

## Data
| File | Source |
|---|---|
| `data/aqi_master_raw.csv` | CPCB daily city AQI bulletins, scraped by urbanemissions.info (github.com/urbanemissions-info/AQI_bulletins). Not committed: see `data/README.md` |
| `data/weather_daily.csv` | ERA5 reanalysis via Open-Meteo (created by `src/fetch_weather.py`). Not committed |

23 cities with >=85% daily coverage over 2017-2020 (and in the lockdown window) are used.

## Method
1. **prepare.py** - clean, deduplicate, select cities, build a daily grid, merge weather, add calendar/Diwali features.
2. **forecast.py** - LightGBM next-day AQI (lags, rolling stats, calendar, weather). Time-based tests (train<=2018 -> 2019; train<=2019 -> Jan-Mar 2020) against persistence and 7-day-mean baselines; alert metrics for AQI>=200.
3. **impact.py** - counterfactual forecasting: train on days before 25 Mar 2020, predict 25 Mar-31 May 2020 "as if nothing happened"; effect = actual - predicted. 7-day block-bootstrap 95% CI per city; four 2019 placebo windows measure the method's noise level; naive year-over-year change reported alongside.
4. **app.py** - Streamlit dashboard: impact, forecast & alerts, what-if simulator, limitations.

## Limitations
- CPCB AQI is an index (max sub-index), not concentrations; sources (traffic/industry/dust) cannot be separated.
- The lockdown cut all activity at once; the effect is the combined effect.
- The what-if simulator scales the lockdown effect linearly - a rough guide only.
- Forecasts use actual weather of the forecast day as a stand-in for a weather forecast.
- Post-lockdown gaps are extrapolations of a pre-lockdown model.

## Results (weather-adjusted, 23 cities, 25 Mar - 31 May 2020)
| Metric | Value |
|---|---|
| Median AQI change vs weather-adjusted "normal" | **-27%** (naive 2020-vs-2019: -36%) |
| Cities whose 95% CI is below zero | 22 / 23 |
| Cities beyond the 2019 placebo noise band (-26% to +24%) | 14 / 23 |
| Largest drops | Jodhpur -43%, Noida -38%, Howrah -37%, Ghaziabad -35%, Rohtak -34%, Delhi -33% |
| No detectable effect | Tirupati (+2%) |
| Prominent pollutant, lockdown vs same window 2019 | PM10 43% -> 21% of days; O3 18% -> 46% |

Next-day forecast (LightGBM vs "tomorrow = today"): MAE 26.7 vs 29.0 (2019 test), 30.4 vs 35.4 (Jan-Mar 2020 test); alert (AQI>=200) recall 0.84 vs 0.80 and 0.79 vs 0.76.

Reading the numbers: the naive comparison overstates the lockdown effect by roughly a quarter because 2020 weather was also
favourable; city effects between about -12% and -26% are statistically below zero but not larger than placebo noise, so treat them as
"likely real, size uncertain". AQI is an index; source attribution is not possible.
