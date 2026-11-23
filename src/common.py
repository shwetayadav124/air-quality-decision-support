"""Shared paths, constants and helpers for the air-quality project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2020-12-31")
LOCKDOWN_START = pd.Timestamp("2020-03-25")   # national lockdown began
LOCKDOWN_END = pd.Timestamp("2020-05-31")     # phases 1-4; "Unlock 1" from 1 June
PRE_END = LOCKDOWN_START - pd.Timedelta(days=1)

# Diwali dates (fireworks spikes) - used as a calendar flag
DIWALI = pd.to_datetime(["2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14"])

WEATHER_COLS = ["temp_mean", "precip_sum", "wind_max", "rh_mean", "pressure_mean"]


def aqi_category(x):
    """CPCB AQI categories."""
    bins = [-np.inf, 50, 100, 200, 300, 400, np.inf]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    return pd.cut(x, bins=bins, labels=labels)


def load_panel():
    """Cleaned city-day panel produced by prepare.py (weather merged if available)."""
    p = DATA / "panel.parquet"
    if not p.exists():
        raise SystemExit("Run `python src/prepare.py` first.")
    return pd.read_parquet(p)


def has_weather(df):
    return all(c in df.columns for c in WEATHER_COLS) and df[WEATHER_COLS].notna().mean().min() > 0.8


def calendar_features(df):
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["t_years"] = (d - START).dt.days / 365.25
    diw = pd.Series(False, index=df.index)
    for dd in DIWALI:
        diw |= (d >= dd - pd.Timedelta(days=1)) & (d <= dd + pd.Timedelta(days=2))
    df["diwali"] = diw.astype(int)
    return df
