"""
generate_data.py
================
Builds the AQI training and holdout datasets from REAL CPCB data
(Kaggle "Air Quality Data in India" dataset — city_day.csv).

DATA SOURCE:
  Kaggle: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
  File:   city_day.csv  (place it in the data/ folder)
  Coverage: 26 Indian cities, 2015-01-01 to 2020-07-01, daily readings.

CITY SPLIT:
  Training cities (18): Delhi, Bengaluru, Kolkata, Chennai, Hyderabad,
                         Lucknow, Jaipur, Patna, Chandigarh, Amritsar,
                         Bhopal, Guwahati, Thiruvananthapuram, Amaravati,
                         Brajrajnagar, Gurugram, Jorapokhar, Talcher
  Holdout cities  (7):  Kochi, Coimbatore, Visakhapatnam, Mumbai,
                         Shillong, Ernakulam, Aizawl
  (Ahmedabad excluded — >30% of AQI readings exceed 500, indicating
   systematic sensor errors in this dataset.)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR / "city_day.csv"

CITY_META = {
    # --- Training cities ---
    "Delhi":              dict(state="Delhi",              region="North",     lat=28.61, lon=77.21, coastal=0, pop_density=11320, industrial_index=0.85),
    "Bengaluru":          dict(state="Karnataka",          region="South",     lat=12.97, lon=77.59, coastal=0, pop_density=4400,  industrial_index=0.22),
    "Kolkata":            dict(state="West Bengal",        region="East",      lat=22.57, lon=88.36, coastal=0, pop_density=24200, industrial_index=0.58),
    "Chennai":            dict(state="Tamil Nadu",         region="South",     lat=13.08, lon=80.27, coastal=1, pop_density=6700,  industrial_index=0.28),
    "Hyderabad":          dict(state="Telangana",          region="South",     lat=17.38, lon=78.49, coastal=0, pop_density=5500,  industrial_index=0.38),
    "Lucknow":            dict(state="Uttar Pradesh",      region="North",     lat=26.85, lon=80.95, coastal=0, pop_density=7300,  industrial_index=0.48),
    "Jaipur":             dict(state="Rajasthan",          region="North",     lat=26.91, lon=75.79, coastal=0, pop_density=7000,  industrial_index=0.32),
    "Patna":              dict(state="Bihar",              region="East",      lat=25.59, lon=85.14, coastal=0, pop_density=8100,  industrial_index=0.42),
    "Chandigarh":         dict(state="Punjab",             region="North",     lat=30.73, lon=76.78, coastal=0, pop_density=9200,  industrial_index=0.32),
    "Amritsar":           dict(state="Punjab",             region="North",     lat=31.63, lon=74.87, coastal=0, pop_density=7200,  industrial_index=0.38),
    "Bhopal":             dict(state="Madhya Pradesh",     region="Central",   lat=23.26, lon=77.41, coastal=0, pop_density=3800,  industrial_index=0.38),
    "Guwahati":           dict(state="Assam",              region="Northeast", lat=26.14, lon=91.73, coastal=0, pop_density=4100,  industrial_index=0.30),
    "Thiruvananthapuram": dict(state="Kerala",             region="South",     lat=8.52,  lon=76.94, coastal=1, pop_density=4400,  industrial_index=0.18),
    "Amaravati":          dict(state="Andhra Pradesh",     region="South",     lat=16.51, lon=80.52, coastal=0, pop_density=3200,  industrial_index=0.30),
    "Brajrajnagar":       dict(state="Odisha",             region="East",      lat=21.82, lon=83.92, coastal=0, pop_density=2800,  industrial_index=0.78),
    "Gurugram":           dict(state="Haryana",            region="North",     lat=28.46, lon=77.03, coastal=0, pop_density=10200, industrial_index=0.72),
    "Jorapokhar":         dict(state="Jharkhand",          region="East",      lat=23.67, lon=86.42, coastal=0, pop_density=3100,  industrial_index=0.80),
    "Talcher":            dict(state="Odisha",             region="East",      lat=20.95, lon=85.23, coastal=0, pop_density=2200,  industrial_index=0.85),
    # --- Holdout cities (NEVER used in training) ---
    "Kochi":              dict(state="Kerala",             region="South",     lat=9.93,  lon=76.27, coastal=1, pop_density=6300,  industrial_index=0.20),
    "Coimbatore":         dict(state="Tamil Nadu",         region="South",     lat=11.02, lon=76.96, coastal=0, pop_density=3900,  industrial_index=0.45),
    "Visakhapatnam":      dict(state="Andhra Pradesh",     region="South",     lat=17.69, lon=83.22, coastal=1, pop_density=5200,  industrial_index=0.48),
    "Mumbai":             dict(state="Maharashtra",        region="West",      lat=19.08, lon=72.88, coastal=1, pop_density=20600, industrial_index=0.52),
    "Shillong":           dict(state="Meghalaya",          region="Northeast", lat=25.57, lon=91.88, coastal=0, pop_density=3800,  industrial_index=0.15),
    "Ernakulam":          dict(state="Kerala",             region="South",     lat=9.98,  lon=76.29, coastal=1, pop_density=5900,  industrial_index=0.22),
    "Aizawl":             dict(state="Mizoram",            region="Northeast", lat=23.73, lon=92.72, coastal=0, pop_density=1800,  industrial_index=0.10),
}

TRAIN_CITIES = [
    "Delhi", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Lucknow",
    "Jaipur", "Patna", "Chandigarh", "Amritsar", "Bhopal", "Guwahati",
    "Thiruvananthapuram",
]
HOLDOUT_CITIES = [
    "Kochi", "Coimbatore", "Visakhapatnam", "Mumbai",
]

COLUMNS = [
    "city", "date", "state", "region", "latitude", "longitude",
    "population_density", "is_coastal", "industrial_index",
    "month", "month_sin", "month_cos", "day_of_year",
    "is_diwali_window", "is_monsoon", "is_winter",
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "precipitation_mm", "pressure_hpa",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "aqi"
]


def _add_engineered_features(df):
    df = df.copy()
    df["month"]       = df["date"].dt.month
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12).round(4)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12).round(4)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_winter"]   = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["is_monsoon"]  = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["is_diwali_window"] = (
        ((df["month"] == 10) & (df["date"].dt.day >= 15)) |
        ((df["month"] == 11) & (df["date"].dt.day <= 10))
    ).astype(int)
    return df


def _add_real_weather(df, weather_path):
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    df = df.merge(
        weather[["city", "date", "temperature_c", "humidity_pct",
                 "wind_speed_kmh", "precipitation_mm", "pressure_hpa"]],
        on=["city", "date"], how="left"
    )
    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm", "pressure_hpa"]:
        df[col] = df.groupby(["city", "month"])[col].transform(
            lambda x: x.fillna(x.median())
        )
    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm", "pressure_hpa"]:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_dataset(cities):
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"city_day.csv not found at {RAW_FILE}\n"
            f"Download from: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india\n"
            f"and place city_day.csv in the data/ folder."
        )

    raw = pd.read_csv(RAW_FILE, parse_dates=["Date"])
    raw = raw.rename(columns={
        "City": "city", "Date": "date",
        "PM2.5": "pm25", "PM10": "pm10",
        "NO2": "no2", "SO2": "so2",
        "CO": "co", "O3": "o3", "AQI": "aqi",
    })

    raw = raw[raw["city"].isin(cities)].copy()
    raw = raw.dropna(subset=["aqi"])
    raw = raw[raw["aqi"] <= 500].copy()
    raw = raw[raw["date"] >= "2017-01-01"].copy()

    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        raw[col] = raw.groupby(["city", raw["date"].dt.month])[col].transform(
            lambda x: x.fillna(x.median())
        )
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        raw[col] = raw[col].fillna(0)

    raw = _add_engineered_features(raw)

    meta_rows = []
    for city in cities:
        m = CITY_META[city]
        meta_rows.append({
            "city": city, "state": m["state"], "region": m["region"],
            "latitude": m["lat"], "longitude": m["lon"],
            "population_density": m["pop_density"],
            "is_coastal": m["coastal"],
            "industrial_index": m["industrial_index"],
        })
    raw = raw.merge(pd.DataFrame(meta_rows), on="city", how="left")

    weather_path = DATA_DIR / "weather_data.csv"
    raw = _add_real_weather(raw, weather_path)

    for col in ["pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
                "temperature_c", "humidity_pct", "wind_speed_kmh"]:
        raw[col] = raw[col].round(2)

    return raw[COLUMNS]


if __name__ == "__main__":
    print("Loading real CPCB data from city_day.csv...")

    train_df   = build_dataset(TRAIN_CITIES)
    holdout_df = build_dataset(HOLDOUT_CITIES)

    train_df.to_csv(DATA_DIR / "train_cities_aqi.csv",    index=False)
    holdout_df.to_csv(DATA_DIR / "holdout_cities_aqi.csv", index=False)

    all_cities = TRAIN_CITIES + HOLDOUT_CITIES
    meta_rows = []
    for city in all_cities:
        m = CITY_META[city]
        meta_rows.append({
            "city": city, "state": m["state"], "region": m["region"],
            "latitude": m["lat"], "longitude": m["lon"],
            "population_density": m["pop_density"],
            "is_coastal": m["coastal"],
            "industrial_index": m["industrial_index"],
            "is_holdout": city in HOLDOUT_CITIES,
        })
    pd.DataFrame(meta_rows).to_csv(DATA_DIR / "city_metadata.csv", index=False)

    print(f"Train cities  ({len(TRAIN_CITIES)}): {TRAIN_CITIES}")
    print(f"Holdout cities ({len(HOLDOUT_CITIES)}): {HOLDOUT_CITIES}")
    print(f"Train shape:   {train_df.shape}")
    print(f"Holdout shape: {holdout_df.shape}")
    print()
    print("AQI by city (annual average):")
    all_df = pd.concat([train_df, holdout_df])
    print(all_df.groupby("city")["aqi"].mean().round(1).sort_values().to_string())
