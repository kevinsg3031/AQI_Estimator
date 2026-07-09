"""
fetch_weather.py
================
Downloads real daily weather for all cities from Open-Meteo (free, no API key).
Run once — saves weather_data.csv to the data/ folder.
Takes ~3-4 minutes for all cities.

If you hit 429 (rate limit) errors, re-run — the script appends to any
existing weather_data.csv so already-fetched cities are skipped.
"""

import requests
import pandas as pd
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "weather_data.csv"

CITIES = {
    # Training cities
    "Delhi":              (28.61, 77.21),
    "Bengaluru":          (12.97, 77.59),
    "Kolkata":            (22.57, 88.36),
    "Chennai":            (13.08, 80.27),
    "Hyderabad":          (17.38, 78.49),
    "Lucknow":            (26.85, 80.95),
    "Jaipur":             (26.91, 75.79),
    "Patna":              (25.59, 85.14),
    "Chandigarh":         (30.73, 76.78),
    "Amritsar":           (31.63, 74.87),
    "Bhopal":             (23.26, 77.41),
    "Guwahati":           (26.14, 91.73),
    "Thiruvananthapuram": (8.52,  76.94),
    "Amaravati":          (16.51, 80.52),
    "Brajrajnagar":       (21.82, 83.92),
    "Gurugram":           (28.46, 77.03),
    "Jorapokhar":         (23.67, 86.42),
    "Talcher":            (20.95, 85.23),
    # Holdout cities
    "Kochi":              (9.93,  76.27),
    "Coimbatore":         (11.02, 76.96),
    "Visakhapatnam":      (17.69, 83.22),
    "Mumbai":             (19.08, 72.88),
    "Shillong":           (25.57, 91.88),
    "Ernakulam":          (9.98,  76.29),
    "Aizawl":             (23.73, 92.72),
}


def fetch_weather(city, lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": "2017-01-01",
        "end_date":   "2020-07-01",
        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "precipitation_sum",
            "surface_pressure_mean",
        ]),
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["daily"]
    return pd.DataFrame({
        "city":             city,
        "date":             pd.to_datetime(data["time"]),
        "temperature_c":    data["temperature_2m_mean"],
        "humidity_pct":     data["relative_humidity_2m_mean"],
        "wind_speed_kmh":   data["wind_speed_10m_mean"],
        "precipitation_mm": data["precipitation_sum"],
        "pressure_hpa":     data["surface_pressure_mean"],
    })


# Load already-fetched cities so we can skip them on re-runs
already_fetched = set()
if OUT_FILE.exists():
    existing = pd.read_csv(OUT_FILE)
    already_fetched = set(existing["city"].unique())
    print(f"Already fetched: {sorted(already_fetched)}")

all_weather = []
for city, (lat, lon) in CITIES.items():
    if city in already_fetched:
        print(f"Skipping {city} (already fetched)")
        continue
    print(f"Fetching {city}...")
    try:
        df = fetch_weather(city, lat, lon)
        all_weather.append(df)
        time.sleep(2)   # stay well within rate limits
    except Exception as e:
        print(f"  ERROR for {city}: {e}")

if all_weather:
    new_df = pd.concat(all_weather, ignore_index=True)
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE, parse_dates=["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(combined)} rows to {OUT_FILE}")
else:
    print("\nAll cities already fetched — nothing to update.")

# Summary
final = pd.read_csv(OUT_FILE)
print(f"\nWeather data coverage ({final['city'].nunique()} cities):")
print(final.groupby("city")[["temperature_c", "precipitation_mm"]].mean().round(1).to_string())
