"""Air Quality Decision Support dashboard.   Run:  streamlit run app.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUT = Path(__file__).parent / "outputs"
LS, LE = pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31")
CATS = [(0, 50, "Good", "#2e7d32"), (51, 100, "Satisfactory", "#7cb342"), (101, 200, "Moderate", "#f9a825"),
        (201, 300, "Poor", "#ef6c00"), (301, 400, "Very Poor", "#c62828"), (401, 500, "Severe", "#6a1b1a")]
ADVICE = {"Good": "Air is fine. Normal activity.",
          "Satisfactory": "Fine for most; very sensitive people may notice mild discomfort.",
          "Moderate": "People with asthma/heart disease: reduce prolonged outdoor exertion.",
          "Poor": "Avoid prolonged outdoor exercise. Children, elderly and people with lung/heart disease should stay indoors where possible.",
          "Very Poor": "Avoid outdoor activity. Schools should move activities indoors. Use masks (N95) if you must go out.",
          "Severe": "Health emergency conditions. Stay indoors; vulnerable groups should seek care if symptomatic."}


def cat(x):
    for lo, hi, n, c in CATS:
        if x <= hi:
            return n, c
    return CATS[-1][2], CATS[-1][3]


st.set_page_config(page_title="Air Quality Decision Support", layout="wide")


@st.cache_data
def load():
    d = {"eff": pd.read_csv(OUT / "impact_by_city.csv"),
         "ser": pd.read_parquet(OUT / "impact_series.parquet"),
         "mon": pd.read_csv(OUT / "impact_monthly_after.csv"),
         "pl": pd.read_csv(OUT / "impact_placebo.csv"),
         "mix": pd.read_csv(OUT / "pollutant_mix.csv"),
         "fm": pd.read_csv(OUT / "forecast_metrics.csv"),
         "fp": pd.read_parquet(OUT / "forecast_predictions.parquet"),
         "fi": pd.read_csv(OUT / "forecast_feature_importance.csv", index_col=0),
         "sum": json.load(open(OUT / "impact_summary.json"))}
    return d


D = load()
S = D["sum"]
st.title("Air Quality Decision Support")
st.caption("Next-day AQI forecasting, high-risk-day alerts and weather-adjusted policy impact for Indian cities "
           "(CPCB daily AQI, 2017-2020).")
if S["weather_adjusted"]:
    st.success("Weather-adjusted: the counterfactual model uses daily weather (ERA5 via Open-Meteo).")
else:
    st.warning("**Preview mode - NOT weather-adjusted.** `data/weather_daily.csv` was not available, so the 'normal' baseline only "
               "knows season, weekday, festivals and city level. Treat impact numbers as indicative; the placebo noise band below shows how uncertain they are.")

tab1, tab2, tab3, tab4 = st.tabs(["Lockdown impact", "Forecast & alerts", "What-if simulator", "Method & limits"])
cities = sorted(D["eff"].city)

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Median AQI change vs 'normal' (lockdown window)", f"{S['median_effect_pct']}%")
    c2.metric("Naive year-over-year change (unadjusted)", f"{S['median_naive_yoy_pct']}%")
    c3.metric("Cities with a drop beyond placebo noise", f"{S['cities_beyond_placebo']} / {S['n_cities']}")
    st.caption(f"Window {S['lockdown_window'][0]} to {S['lockdown_window'][1]}. Placebo (2019, no lockdown) 95% band: "
               f"{S['placebo_95_band_pct'][0]}% to {S['placebo_95_band_pct'][1]}% - effects inside this band cannot be distinguished from noise.")
    e = D["eff"].sort_values("effect_pct")
    fig = go.Figure()
    fig.add_bar(y=e.city, x=e.naive_yoy_pct, orientation="h", name="Naive YoY (2020 vs 2019)", marker_color="#b0bec5")
    fig.add_bar(y=e.city, x=e.effect_pct, orientation="h", name="Adjusted vs counterfactual", marker_color="#1565c0",
                error_x=dict(type="data", symmetric=False, array=e.ci_high_pct - e.effect_pct, arrayminus=e.effect_pct - e.ci_low_pct))
    fig.add_vrect(x0=S["placebo_95_band_pct"][0], x1=S["placebo_95_band_pct"][1], fillcolor="orange", opacity=0.12, line_width=0,
                  annotation_text="placebo noise band", annotation_position="top left")
    fig.update_layout(barmode="group", height=120 + 26 * len(e), xaxis_title="% change in AQI", margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)

    city = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0, key="c1")
    s = D["ser"][D["ser"].city == city].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[(s.date >= "2020-01-01") & (s.date <= "2020-12-31")].set_index("date")
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.aqi.rolling(7, min_periods=3).mean(), name="Actual AQI (7-day avg)", line=dict(color="#c62828"))
    fig.add_scatter(x=s.index, y=s.pred.rolling(7, min_periods=3).mean(), name="Expected without lockdown (7-day avg)", line=dict(color="#1565c0", dash="dash"))
    fig.add_vrect(x0=LS, x1=LE, fillcolor="grey", opacity=0.15, line_width=0, annotation_text="lockdown")
    fig.update_layout(height=380, yaxis_title="AQI", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    r = e[e.city == city].iloc[0]
    st.write(f"**{city}:** actual {r.actual_mean:.0f} vs expected {r.counterfactual_mean:.0f} -> **{r.effect_pct:.1f}%** "
             f"(95% CI {r.ci_low_pct:.1f}% to {r.ci_high_pct:.1f}%). Naive year-over-year: {r.naive_yoy_pct:.1f}%.")
    st.caption("After the window, the gap is an extrapolation of a pre-lockdown model - read June-December as indicative only.")
    st.subheader("Which pollutant set the AQI? (hint about sources)")
    st.dataframe(D["mix"].set_index("pollutant"), use_container_width=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    fm = D["fm"].set_index("set")
    st.subheader("Next-day forecast vs baselines (time-based test sets)")
    rows = []
    for k in fm.index:
        for m in ["persistence", "mean7", "lightgbm"]:
            rows.append({"test set": k, "model": m, "MAE": fm.loc[k, f"{m}_MAE"], "RMSE": fm.loc[k, f"{m}_RMSE"], "R2": fm.loc[k, f"{m}_R2"],
                         "alert recall": fm.loc[k, f"{m}_alert_recall"], "alert precision": fm.loc[k, f"{m}_alert_precision"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    city2 = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0, key="c2")
    p = D["fp"][D["fp"].city == city2].copy()
    p["date"] = pd.to_datetime(p["date"])
    split = st.radio("Test period", ["test_2019", "test_2020_preLockdown"], horizontal=True)
    p = p[p.split == split].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=p.date, y=p.actual, name="Actual", line=dict(color="#424242"))
    fig.add_scatter(x=p.date, y=p.forecast, name="Forecast (made the day before)", line=dict(color="#1565c0"))
    fig.add_hline(y=200, line_dash="dot", line_color="#ef6c00", annotation_text="Poor (alert threshold)")
    fig.update_layout(height=360, yaxis_title="AQI", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    last = p.iloc[-1]
    n, col = cat(last.forecast)
    st.markdown(f"**Latest forecast in this test period ({last.date.date()}):** AQI ~{last.forecast:.0f} - "
                f"<span style='color:{col};font-weight:700'>{n}</span>. {ADVICE[n]}", unsafe_allow_html=True)
    hit = p[(p.forecast >= 200) & (p.actual >= 200)].shape[0]; alarms = p[p.forecast >= 200].shape[0]; real = p[p.actual >= 200].shape[0]
    st.caption(f"Alert days in this period: {real} real Poor+ days; {alarms} alerts raised; {hit} correctly caught.")
    st.subheader("What drives the forecast")
    fi = D["fi"].head(10).iloc[::-1]
    st.plotly_chart(go.Figure(go.Bar(x=fi.mean_abs_contribution, y=fi.index, orientation="h")).update_layout(
        height=330, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="mean |contribution to forecast| (AQI points)"), use_container_width=True)
    st.caption("Forecasts use the actual weather of the forecast day as a stand-in for a weather forecast (when weather data is present).")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Policy what-if: how much could a partial activity cut lower AQI?")
    st.info("Rough estimate. It scales the city's measured lockdown effect linearly. The lockdown was an extreme, multi-source shutdown, "
            "so real partial measures (odd-even, construction ban) may behave differently. Use it to compare cities, not to forecast policy outcomes.")
    c = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0, key="c3")
    level = st.slider("Activity reduction, as % of the lockdown level", 0, 100, 40, 5)
    r = D["eff"][D["eff"].city == c].iloc[0]
    base = D["ser"][(D["ser"].city == c)].copy(); base["date"] = pd.to_datetime(base["date"])
    base = base[(base.date.dt.year == 2019)].groupby(base.date.dt.month).aqi.mean()
    k = level / 100
    sc = base * (1 + k * r.effect_pct / 100); lo = base * (1 + k * r.ci_low_pct / 100); hi = base * (1 + k * r.ci_high_pct / 100)
    m = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    fig.add_scatter(x=m, y=base.values, name="Typical 2019 AQI", line=dict(color="#424242"))
    fig.add_scatter(x=m, y=hi.values, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=m, y=lo.values, fill="tonexty", fillcolor="rgba(21,101,192,.15)", line=dict(width=0), name="95% range")
    fig.add_scatter(x=m, y=sc.values, name=f"With {level}% of lockdown-level cut", line=dict(color="#1565c0"))
    fig.update_layout(height=380, yaxis_title="Monthly mean AQI", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"{c}: measured lockdown effect {r.effect_pct:.1f}% (95% CI {r.ci_low_pct:.1f}% to {r.ci_high_pct:.1f}%) -> "
             f"scaled to {level}%: about **{k * r.effect_pct:.1f}%**.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.markdown(f"""
**Data.** CPCB daily city AQI bulletins (urbanemissions.info scrape), {S['n_cities']} cities with >=85% daily coverage 2017-2020.
Weather (if present): ERA5 daily via Open-Meteo.

**Impact method.** Train a model on pre-lockdown days (up to 24 Mar 2020) to learn what AQI normally looks like; predict
25 Mar-31 May 2020 as if nothing happened; effect = actual - predicted. Uncertainty: 7-day block bootstrap. Noise check: the same procedure applied to four 2019 windows (no lockdown).

**Forecast method.** LightGBM, global model over cities, lags/rolling stats + calendar (+weather). Compared with persistence and 7-day-mean baselines on time-based test sets.

**Limits (read before quoting numbers).**
- AQI is the CPCB index (max sub-index), not raw concentrations. Source attribution (vehicles vs industry vs dust) is **not** possible; the prominent-pollutant mix is only a hint.
- The lockdown shut down everything at once - the effect is the combined effect of all activity cuts.
- City AQI is an average of stations whose number changes over time.
- Without weather, the baseline cannot separate weather from policy. The placebo band quantifies this.
- Forecast skill is measured with actual weather standing in for a weather forecast.
""")
