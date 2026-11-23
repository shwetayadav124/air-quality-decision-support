"""Step 2 (Module 2): next-day AQI forecasting + high-risk-day alerts.

Forecast issued at the end of day d for day d+1.
Weather (if available) is the ACTUAL weather of day d+1 - a stand-in for a weather forecast.
Validation is strictly time-based:  train <= 2018  -> test 2019 ;   train <= 2019 -> test 2020-01-01..2020-03-24
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support
from common import *

ALERT = 200  # CPCB "Poor" or worse


def build(panel):
    df = panel.sort_values(["city", "date"]).copy()
    g = df.groupby("city").aqi
    for k in range(0, 7):
        df[f"lag{k}"] = g.shift(k)
    df["roll7"] = g.transform(lambda s: s.rolling(7, min_periods=5).mean())
    df["roll30"] = g.transform(lambda s: s.rolling(30, min_periods=20).mean())
    df["std7"] = g.transform(lambda s: s.rolling(7, min_periods=5).std())
    df["target"] = g.shift(-1)
    df["target_date"] = df.groupby("city").date.shift(-1)
    # calendar & weather belong to the TARGET day
    cal = calendar_features(df[["target_date"]].rename(columns={"target_date": "date"}).copy())
    for c in ["dow", "month", "doy_sin", "doy_cos", "diwali"]:
        df[c] = cal[c].values
    if has_weather(df):
        for c in WEATHER_COLS:
            df[c] = df.groupby("city")[c].shift(-1)
    df["city_cat"] = df["city"].astype("category")
    return df.dropna(subset=["target", "lag0", "roll7", "target_date"])


def feature_list(df):
    f = [f"lag{k}" for k in range(7)] + ["roll7", "roll30", "std7", "dow", "month", "doy_sin", "doy_cos", "diwali", "city_cat"]
    if has_weather(df):
        f += WEATHER_COLS
    return f


def fit(train, feats):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=0, verbose=-1)
    m.fit(train[feats], train["target"])
    return m


def evaluate(name, d, pred):
    y = d["target"].values
    out = {"set": name, "n": int(len(d))}
    for label, p in {"persistence": d["lag0"].values, "mean7": d["roll7"].values, "lightgbm": pred}.items():
        out[f"{label}_MAE"] = round(mean_absolute_error(y, p), 2)
        out[f"{label}_RMSE"] = round(float(np.sqrt(mean_squared_error(y, p))), 2)
        out[f"{label}_R2"] = round(r2_score(y, p), 3)
        yt, yp = (y >= ALERT), (p >= ALERT)
        pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        out[f"{label}_alert_precision"], out[f"{label}_alert_recall"], out[f"{label}_alert_f1"] = round(pr, 3), round(rc, 3), round(f1, 3)
    out["alert_base_rate"] = round(float((y >= ALERT).mean()), 3)
    return out


def main():
    panel = load_panel()
    df = build(panel)
    feats = feature_list(df)
    print("features:", feats)
    results, preds = [], []

    splits = [("test_2019", pd.Timestamp("2018-12-31"), pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")),
              ("test_2020_preLockdown", pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), PRE_END)]
    final_model = None
    for name, tr_end, te_a, te_b in splits:
        tr = df[df.target_date <= tr_end]
        te = df[(df.target_date >= te_a) & (df.target_date <= te_b)]
        m = fit(tr, feats)
        p = np.clip(m.predict(te[feats]), 0, 500)
        results.append(evaluate(name, te, p))
        preds.append(pd.DataFrame({"city": te.city.values, "date": te.target_date.values, "actual": te.target.values,
                                   "forecast": p, "persistence": te.lag0.values, "split": name}))
        final_model = (m, te)

    res = pd.DataFrame(results)
    print(res.T.to_string())
    res.to_csv(OUT / "forecast_metrics.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "forecast_predictions.parquet", index=False)

    # per-city error on the 2019 test
    p19 = preds[0].copy()
    pc = p19.groupby("city").apply(lambda x: pd.Series({
        "MAE_model": mean_absolute_error(x.actual, x.forecast),
        "MAE_persistence": mean_absolute_error(x.actual, x.persistence)})).round(2)
    pc.to_csv(OUT / "forecast_by_city_2019.csv")

    # feature importance (mean |SHAP-style contribution| from LightGBM) on the 2020 pre-lockdown test
    m, te = final_model
    contrib = m.predict(te[feats], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
    imp.round(3).to_csv(OUT / "forecast_feature_importance.csv", header=["mean_abs_contribution"])
    print("\ntop drivers:\n", imp.head(10).round(2).to_string())
    json.dump({"weather_adjusted": bool(has_weather(df)), "features": feats}, open(OUT / "forecast_meta.json", "w"))


if __name__ == "__main__":
    main()
