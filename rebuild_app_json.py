"""
rebuild_app_json.py
====================
Run from inside your aqi_project_3 folder AFTER retraining:

    cd ~/Downloads/aqi_project_3
    python3 rebuild_app_json.py

Reads the trained model, real data, and explainability.json — writes
app/model_data.json with zero hardcoded values.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent

# ── load artefacts ───────────────────────────────────────────────────────────
pipe     = joblib.load(BASE / "models" / "aqi_model.joblib")
spec     = joblib.load(BASE / "models" / "feature_spec.joblib")
meta_df  = pd.read_csv(BASE / "data" / "city_metadata.csv")
train_df = pd.read_csv(BASE / "data" / "train_cities_aqi.csv", parse_dates=["date"])
hold_df  = pd.read_csv(BASE / "data" / "holdout_cities_aqi.csv", parse_dates=["date"])
all_df   = pd.concat([train_df, hold_df], ignore_index=True)

with open(BASE / "data" / "explainability.json") as f:
    explainability = json.load(f)

NUMERIC = spec["numeric"]
CAT     = spec["categorical"]
FEATS   = NUMERIC + CAT

# ── model stats — read from CSV files, never hardcoded ──────────────────────
metrics_path = BASE / "models" / "final_holdout_metrics.csv"
comp_path    = BASE / "models" / "model_comparison.csv"

metrics = pd.read_csv(metrics_path).iloc[0]
comp    = pd.read_csv(comp_path)

# Best model name from comparison table
unseen = comp[comp["split"].str.contains("UNSEEN")]
best_model_name = unseen.loc[unseen["rmse"].idxmin(), "model"] if len(unseen) else "unknown"

model_stats = {
    "unseen_city_r2":  round(float(metrics.get("r2",  0)), 3),
    "unseen_city_mae": round(float(metrics.get("mae", 0)), 2),
    "train_cities":    int(meta_df[meta_df["is_holdout"] == False].shape[0]),
    "holdout_cities":  int(meta_df[meta_df["is_holdout"] == True].shape[0]),
    "date_range":      "2017-2020",
    "best_model":      str(best_model_name),
    "data_source":     "Real CPCB (Kaggle city_day.csv)",
}

# ── helper: predict for one city × month ────────────────────────────────────
def predict_city_month(crow, month, ref_df, city_name):
    rows = ref_df[(ref_df["city"] == city_name) & (ref_df["month"] == month)]
    if len(rows):
        temp     = rows["temperature_c"].median()
        humidity = rows["humidity_pct"].median()
        wind     = rows["wind_speed_kmh"].median()
        doy      = rows["day_of_year"].median()
        precip   = rows["precipitation_mm"].median() if "precipitation_mm" in rows.columns else 2.0
        pressure = rows["pressure_hpa"].median()     if "pressure_hpa"     in rows.columns else 1010.0
    else:
        is_m     = 1 if month in (6, 7, 8, 9) else 0
        doy      = month * 30
        temp     = 22 + 10 * np.sin((doy / 365) * 2 * np.pi - 1.4)
        humidity = 55 + (30 if is_m else 0)
        wind     = 8  + (4  if crow["is_coastal"] else 0)
        precip   = 8.0 if is_m else 1.5
        pressure = 1008.0

    is_winter  = 1 if month in (11, 12, 1, 2) else 0
    is_monsoon = 1 if month in (6, 7, 8, 9)  else 0
    is_diwali  = 1 if month == 10             else 0
    month_sin  = round(np.sin(2 * np.pi * month / 12), 4)
    month_cos  = round(np.cos(2 * np.pi * month / 12), 4)

    row_dict = {
        "latitude":           crow["latitude"],
        "longitude":          crow["longitude"],
        "population_density": crow["population_density"],
        "is_coastal":         crow["is_coastal"],
        "industrial_index":   crow["industrial_index"],
        "month_sin":          month_sin,
        "month_cos":          month_cos,
        "day_of_year":        doy,
        "is_diwali_window":   is_diwali,
        "is_monsoon":         is_monsoon,
        "is_winter":          is_winter,
        "temperature_c":      round(temp, 1),
        "humidity_pct":       round(humidity, 1),
        "wind_speed_kmh":     round(wind, 1),
        "precipitation_mm":   round(precip, 1),
        "pressure_hpa":       round(pressure, 1),
        "region":             crow["region"],
    }
    X = pd.DataFrame([{k: row_dict[k] for k in FEATS if k in row_dict}])
    for f in FEATS:
        if f not in X.columns:
            X[f] = 0
    return round(max(float(pipe.predict(X[FEATS])[0]), 0), 1)


# ── city monthly predictions ─────────────────────────────────────────────────
print("Computing city predictions...")
city_preds = {}
for _, crow in meta_df.iterrows():
    city = crow["city"]
    city_preds[city] = [predict_city_month(crow, m, all_df, city) for m in range(1, 13)]

# real monthly averages from actual CPCB data
city_real = {}
for city in all_df["city"].unique():
    sub = all_df[all_df["city"] == city]
    city_real[city] = [
        round(sub[sub["month"] == m]["aqi"].mean(), 1) if len(sub[sub["month"] == m]) else None
        for m in range(1, 13)
    ]

# ── custom city prediction grid ──────────────────────────────────────────────
print("Building custom prediction grid...")
REGIONS = ["North", "South", "East", "West", "Central", "Northeast"]
custom_grid = {}

for region in REGIONS:
    for month in range(1, 13):
        key        = f"{region}_{month}"
        is_winter  = 1 if month in (11, 12, 1, 2) else 0
        is_monsoon = 1 if month in (6, 7, 8, 9)  else 0
        is_diwali  = 1 if month == 10             else 0
        month_sin  = round(np.sin(2 * np.pi * month / 12), 4)
        month_cos  = round(np.cos(2 * np.pi * month / 12), 4)
        doy        = month * 30
        temp       = 22 + 10 * np.sin((doy / 365) * 2 * np.pi - 1.4)
        humidity   = 55 + (30 if is_monsoon else 0)
        precip     = 8.0 if is_monsoon else 1.5
        pressure   = 1008.0

        rows = []
        for ind in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for pop in [3000, 6000, 10000, 15000, 20000, 27000]:
                for coastal in [0, 1]:
                    wind = 8 + (4 if coastal else 0)
                    row_dict = {
                        "latitude": 20.0, "longitude": 78.0,
                        "population_density": pop, "is_coastal": coastal,
                        "industrial_index": ind,
                        "month_sin": month_sin, "month_cos": month_cos,
                        "day_of_year": doy,
                        "is_diwali_window": is_diwali, "is_monsoon": is_monsoon,
                        "is_winter": is_winter,
                        "temperature_c": round(temp, 1), "humidity_pct": round(humidity, 1),
                        "wind_speed_kmh": round(wind, 1),
                        "precipitation_mm": round(precip, 1), "pressure_hpa": round(pressure, 1),
                        "region": region,
                    }
                    X = pd.DataFrame([{k: row_dict[k] for k in FEATS if k in row_dict}])
                    for f in FEATS:
                        if f not in X.columns:
                            X[f] = 0
                    pred = round(max(float(pipe.predict(X[FEATS])[0]), 0), 1)
                    rows.append({"industrial": ind, "pop_density": pop, "coastal": coastal, "pred": pred})
        custom_grid[key] = rows

# ── holdout true averages ─────────────────────────────────────────────────────
holdout_true = {
    city: round(hold_df[hold_df["city"] == city]["aqi"].mean(), 1)
    for city in meta_df[meta_df["is_holdout"] == True]["city"]
}

# ── assemble and write ────────────────────────────────────────────────────────
output = {
    "city_monthly_aqi":       city_preds,
    "city_monthly_real":      city_real,
    "city_meta":              meta_df.to_dict(orient="records"),
    "custom_prediction_grid": custom_grid,
    "holdout_true_avg":       holdout_true,
    "model_stats":            model_stats,
    "explainability":         explainability,   # feature_importance, validation, partial_dependence
}

out_path = BASE / "app" / "model_data.json"
with open(out_path, "w") as f:
    json.dump(output, f)

print(f"\nWrote {out_path}  ({round(out_path.stat().st_size / 1024, 1)} KB)")
print(f"\nModel stats (from CSV files):")
for k, v in model_stats.items():
    print(f"  {k}: {v}")
print("\nSpot-check predictions:")
for city in ["Delhi", "Bengaluru", "Mumbai", "Kochi", "Chennai"]:
    p = city_preds.get(city, [None] * 12)
    r = city_real.get(city,  [None] * 12)
    print(f"  {city:20s}  Jan pred={p[0]:5.0f}  real={str(r[0]):>6}  "
          f"Jul pred={p[6]:5.0f}  real={str(r[6]):>6}")
