"""Step 3 (Module 1): weather-adjusted lockdown impact via counterfactual forecasting.

Idea: learn what AQI "normally" looks like for each city (season, weekday, Diwali, city level, long-run level and -
if weather_daily.csv is present - weather). Train ONLY on data before the lockdown, predict the lockdown window as if
nothing had happened, and compare with what was actually measured.   effect = actual - counterfactual.

Checks built in:
  * placebo windows in 2019 (no lockdown) -> how big is "noise" for this method
  * block-bootstrap 95% CI per city
  * naive year-over-year comparison reported next to the adjusted number
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import *

WIN_DAYS = (LOCKDOWN_END - LOCKDOWN_START).days + 1
RNG = np.random.default_rng(0)


def features(df):
    f = ["doy_sin", "doy_cos", "dow", "diwali", "t_years", "city_cat"]
    if has_weather(df):
        df = df.sort_values(["city", "date"]).copy()
        for c, w in [("wind_max", 3), ("precip_sum", 3)]:
            df[f"{c}_3d"] = df.groupby("city")[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
        f += WEATHER_COLS + ["wind_max_3d", "precip_sum_3d"]
    return df, f


def counterfactual(df, feats, train_end, a, b):
    tr = df[(df.date <= train_end) & df.aqi.notna()]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    m.fit(tr[feats], tr["aqi"])
    w = df[(df.date >= a) & (df.date <= b)].copy()
    w["pred"] = np.clip(m.predict(w[feats]), 0, 500)
    return w, m


def city_effect(w, n_boot=1000, block=7):
    rows = []
    for city, x in w.dropna(subset=["aqi"]).groupby("city"):
        x = x.sort_values("date")
        res, pred = (x.aqi - x.pred).values, x.pred.values
        eff, base = res.mean(), pred.mean()
        # moving-block bootstrap over days
        n = len(res); nb = int(np.ceil(n / block)); boots = []
        for _ in range(n_boot):
            st = RNG.integers(0, n - block + 1, nb)
            idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
            boots.append(res[idx].mean() / pred[idx].mean() * 100)
        rows.append({"city": city, "n_days": n, "actual_mean": x.aqi.mean(), "counterfactual_mean": base,
                     "effect_abs": eff, "effect_pct": eff / base * 100,
                     "ci_low_pct": np.percentile(boots, 2.5), "ci_high_pct": np.percentile(boots, 97.5)})
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    df, feats = features(panel)
    df["city_cat"] = df["city"].astype("category")
    wadj = has_weather(df)
    print("weather-adjusted:", wadj, "| features:", feats)

    # ---- real lockdown window
    w, model = counterfactual(df, feats, PRE_END, LOCKDOWN_START, LOCKDOWN_END)
    eff = city_effect(w)

    # naive year-over-year change for the same window
    a19, b19 = LOCKDOWN_START - pd.DateOffset(years=1), LOCKDOWN_END - pd.DateOffset(years=1)
    y19 = df[(df.date >= a19) & (df.date <= b19)].groupby("city").aqi.mean()
    eff["naive_yoy_pct"] = (eff.set_index("city").actual_mean / y19.reindex(eff.city).values - 1).values * 100
    eff["significant"] = (eff.ci_low_pct > 0) | (eff.ci_high_pct < 0)

    # ---- placebo windows (no lockdown happened) -> noise level of the method
    pl = []
    for s in ["2019-03-25", "2019-06-03", "2019-08-12", "2019-10-21"]:
        a = pd.Timestamp(s); b = a + pd.Timedelta(days=WIN_DAYS - 1)
        wp, _ = counterfactual(df, feats, a - pd.Timedelta(days=1), a, b)
        e = city_effect(wp, n_boot=1)[["city", "effect_pct"]]
        e["window"] = s
        pl.append(e)
    pl = pd.concat(pl)
    lo, hi = np.percentile(pl.effect_pct, [2.5, 97.5])
    eff["beyond_placebo_noise"] = (eff.effect_pct < lo) | (eff.effect_pct > hi)
    eff = eff.sort_values("effect_pct")

    # ---- daily series for dashboard (2020 + 2019 for comparison)
    series = df[(df.date >= "2019-01-01")].copy()
    series["pred"] = np.clip(model.predict(series[feats]), 0, 500)
    series[["city", "date", "aqi", "pred"]].to_parquet(OUT / "impact_series.parquet", index=False)

    # ---- after the lockdown: monthly gap vs counterfactual (extrapolation -> indicative only)
    post = series[(series.date >= LOCKDOWN_START) & series.aqi.notna()].copy()
    post["month"] = post.date.dt.to_period("M").astype(str)
    mon = post.groupby(["city", "month"]).agg(actual=("aqi", "mean"), cf=("pred", "mean")).reset_index()
    mon["effect_pct"] = (mon.actual / mon.cf - 1) * 100
    mon.to_csv(OUT / "impact_monthly_after.csv", index=False)

    # ---- which pollutant drives the AQI? (hint about sources) lockdown window 2020 vs same window 2019
    def mix(a, b, lab):
        x = df[(df.date >= a) & (df.date <= b) & df.pollutant.notna()]
        return (x.pollutant.value_counts(normalize=True) * 100).rename(lab)
    pm = pd.concat([mix(a19, b19, "2019_same_window_%"), mix(LOCKDOWN_START, LOCKDOWN_END, "2020_lockdown_%")], axis=1).fillna(0).round(1)
    pm.to_csv(OUT / "pollutant_mix.csv")

    eff.round(2).to_csv(OUT / "impact_by_city.csv", index=False)
    pl.round(2).to_csv(OUT / "impact_placebo.csv", index=False)

    summary = {
        "weather_adjusted": bool(wadj),
        "n_cities": int(eff.city.nunique()),
        "lockdown_window": [str(LOCKDOWN_START.date()), str(LOCKDOWN_END.date())],
        "median_effect_pct": round(float(eff.effect_pct.median()), 1),
        "mean_effect_pct": round(float(eff.effect_pct.mean()), 1),
        "median_naive_yoy_pct": round(float(eff.naive_yoy_pct.median()), 1),
        "cities_significant_drop": int(((eff.ci_high_pct < 0)).sum()),
        "cities_beyond_placebo": int(eff.beyond_placebo_noise.sum()),
        "placebo_95_band_pct": [round(lo, 1), round(hi, 1)],
        "placebo_median_abs_pct": round(float(pl.effect_pct.abs().median()), 1),
    }
    json.dump(summary, open(OUT / "impact_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(eff[["city", "actual_mean", "counterfactual_mean", "effect_pct", "ci_low_pct", "ci_high_pct", "naive_yoy_pct"]].round(1).to_string(index=False))
    print(pm)


if __name__ == "__main__":
    main()
