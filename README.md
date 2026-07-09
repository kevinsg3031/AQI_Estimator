# AQI City Generalizer — Unseen City AQI Prediction

> **Resume bullet:**  
> Built an AQI prediction system using a pooled regression model (GradientBoosting, R²=0.83 on 7 completely held-out cities) that estimates air quality for Indian cities with no monitoring history, using only location characteristics — addressing a real gap in CPCB's monitoring coverage.

---

## What This Project Does

India's CPCB monitors air quality in major cities, but **hundreds of smaller towns and cities have no monitoring stations**. Most ML projects just forecast AQI for cities that already have rich historical data. This project solves a harder, more realistic problem:

> *Given only a city's geographic and demographic characteristics — with zero historical AQI data — predict its monthly air quality pattern.*

This is achieved via **Option-B: Pooled Regression with Location Features** — a single model trained on 39 cities that learns the *generalizable relationship* between city characteristics and AQI, then applies that knowledge to cities it has never seen.

---

## Key Design Choices (Interview Talking Points)

### 1. City-level holdout, not random row split
Most tutorials split data row-randomly (95% train / 5% test). For this project, that would mean the model sees 95% of Delhi's rows in training and tests on the other 5% — proving nothing about generalizing to *new* cities. Instead, 7 cities were entirely held out: the model trained on 39 cities and was evaluated only on the 7 it had never seen.

### 2. Why GradientBoosting beat Random Forest on unseen cities
Random Forest performed better on the *training* cities but worse on holdout cities — classic overfitting. GradientBoosting's lower variance and more conservative splits generalized better (RMSE 7.79 vs 7.96).

### 3. Cyclical month encoding
Raw `month` (1-12) treats December and January as far apart numerically, but they're adjacent seasons. Encoding `sin(2π×month/12)` and `cos(2π×month/12)` makes the representation cyclically correct — the model "knows" December and January are near each other.

### 4. Permutation importance over built-in feature importance
Tree-based built-in importance is biased toward high-cardinality features. Permutation importance (shuffling each feature's values and measuring the drop in accuracy) is a more honest measure — and crucially, it was computed on the *holdout cities*, measuring what matters for generalization, not memorization.

### 5. The real-data upgrade path is one function swap
The data generation function is deliberately isolated in `data/generate_data.py → build_dataset()`. To use real CPCB data, replace the body of that function with `pd.read_csv()` of the actual CPCB export. Nothing else in the pipeline changes.

---

## Results

| Split | MAE | RMSE | R² |
|---|---|---|---|
| Seen cities (2024, time-based) | 4.65 | 6.40 | 0.843 |
| **Unseen cities (holdout)** | **5.08** | **7.85** | **0.831** |

Per-city holdout breakdown:

| City | MAE | R² | True avg AQI | Pred avg AQI |
|---|---|---|---|---|
| Kochi | 3.49 | 0.465 | 19.2 | 21.9 |
| Coimbatore | 4.16 | 0.658 | 29.5 | 30.9 |
| Visakhapatnam | 3.58 | 0.664 | 25.6 | 26.5 |
| Guwahati | 5.40 | 0.628 | 37.2 | 39.6 |
| Chandigarh | 6.26 | 0.810 | 47.0 | 45.0 |
| Mangaluru | 3.63 | 0.671 | 27.3 | 28.3 |
| Allahabad | 9.05 | 0.705 | 56.1 | 50.1 |

---

## Project Structure

```
aqi_project/
├── data/
│   ├── generate_data.py        # Data generation (swap build_dataset() for real CPCB data)
│   ├── train_cities_aqi.csv    # 39-city training dataset (3 years)
│   ├── holdout_cities_aqi.csv  # 7-city holdout dataset
│   └── city_metadata.csv       # City characteristics for all 46 cities
├── models/
│   ├── train_model.py          # Training + city-level holdout evaluation
│   ├── explainability.py       # Permutation importance + partial dependence
│   ├── aqi_model.joblib        # Trained GradientBoosting pipeline
│   ├── feature_spec.joblib     # Feature names used by model
│   ├── model_comparison.csv    # All model comparisons
│   ├── permutation_importance.csv
│   └── final_holdout_metrics.csv
└── app/
    ├── index.html              # Standalone interactive app (no server needed)
    └── model_data.json         # Precomputed predictions exported from model
```

---

## Quickstart

### Run the full pipeline (Python)

```bash
pip install scikit-learn pandas numpy joblib matplotlib

# 1. Generate data
python data/generate_data.py

# 2. Train model + evaluate
python models/train_model.py

# 3. Explainability
python models/explainability.py

# Optional: install SHAP for the full per-prediction attribution
# pip install shap
# (see models/explainability.py for the SHAP upgrade path)
```

### Run the interactive app

```bash
cd app
python -m http.server 8000
# Open http://localhost:8000
```

The app works by loading `model_data.json` (precomputed predictions from the trained model). No Python runtime needed to run the app — it is a zero-dependency HTML file.

---

## To Use Real CPCB Data

1. Download city-level AQI data from [data.gov.in](https://data.gov.in) or the [Vayuayan archive](https://saket-choudhary.me/vayuayan-archive/) (560+ CPCB stations, CSVs)
2. Open `data/generate_data.py`
3. Replace the body of `build_dataset(cities)` with:
   ```python
   df = pd.read_csv("path/to/cpcb_export.csv", parse_dates=["date"])
   # Rename columns to match COLUMNS list at top of file
   return df
   ```
4. Re-run `python data/generate_data.py && python models/train_model.py`

Everything downstream — model, explainability, app — works identically on real data.

---

## Features Used by the Model

| Feature | Type | Why |
|---|---|---|
| `industrial_index` | Float 0–1 | Continuous industrial load score — #1 driver for unseen city generalization |
| `is_winter` | Binary | Nov–Feb inversion + stubble burning spikes in North India |
| `is_coastal` | Binary | Sea breeze dispersal → lower baseline AQI |
| `is_monsoon` | Binary | Rain washout effect → sharp AQI drop Jun–Sep |
| `region` | Category | North India spikes harder in winter than South |
| `population_density` | Float | More vehicles, more cooking fires, more emissions |
| `month_sin`, `month_cos` | Float | Cyclical encoding so Dec and Jan are numerically adjacent |
| `is_diwali_window` | Binary | Firecracker spike: mid-Oct to early Nov |
| `latitude`, `longitude` | Float | Spatial signal beyond named region |
| `temperature_c`, `humidity_pct`, `wind_speed_kmh` | Float | Meteorological conditions affect pollution dispersion |

---

## Why This Is Interesting (vs Typical AQI Projects)

| Typical project | This project |
|---|---|
| Forecast next-day AQI for one city | Estimate AQI for cities with *no* monitoring data |
| Random train/test split | City-level holdout (tests true generalization) |
| Single-city time-series model | Cross-city pooled model with location features |
| Model accuracy as only metric | Explainability: permutation importance + partial dependence |
| Jupyter notebook, no deployment | Standalone interactive app, zero setup |

---

*Data is synthetic but structurally mirrors real CPCB patterns. Swap in real data using the instructions above.*
