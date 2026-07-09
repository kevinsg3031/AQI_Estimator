"""Train a model for AQI climatology in cities absent from training.

The prediction target is a city's typical AQI for a month of the year.  Daily
AQI is not identifiable from location and weather alone (pollutant readings or
AQI lags would be needed), so fitting daily rows mostly learns noise.  We first
collapse repeated years to city-month climatologies and validate by holding out
entire cities.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "latitude", "longitude", "population_density", "is_coastal",
    "industrial_index", "month_sin", "month_cos", "temperature_c",
    "humidity_pct", "wind_speed_kmh", "precipitation_mm", "pressure_hpa",
]
CATEGORICAL_FEATURES = ["region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "aqi"


def load_data():
    train = pd.read_csv(DATA_DIR / "train_cities_aqi.csv", parse_dates=["date"])
    holdout = pd.read_csv(DATA_DIR / "holdout_cities_aqi.csv", parse_dates=["date"])
    return train, holdout


def city_month_climatology(df):
    """One row per city/month, averaging across available years."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    aggregations = {feature: "mean" for feature in NUMERIC_FEATURES}
    aggregations.update({"region": "first", TARGET: "mean", "date": "count"})
    result = df.groupby(["city", "month"], as_index=False).agg(aggregations)
    return result.rename(columns={"date": "observation_count"})


def build_pipeline(model, scale=False):
    numeric_transformer = StandardScaler() if scale else "passthrough"
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def candidates():
    return {
        # Similar-city interpolation is a natural fit for unseen locations.
        "KNeighbors": build_pipeline(
            KNeighborsRegressor(n_neighbors=8, weights="distance"), scale=True
        ),
        "Ridge": build_pipeline(Ridge(alpha=100), scale=True),
        "ExtraTrees": build_pipeline(ExtraTreesRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=3,
            max_features=0.8, random_state=42, n_jobs=-1,
        )),
        "GradientBoosting": build_pipeline(GradientBoostingRegressor(
            n_estimators=200, max_depth=2, min_samples_leaf=8,
            learning_rate=0.03, loss="huber", random_state=42,
        )),
    }


def metrics(y_true, predictions):
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "r2": r2_score(y_true, predictions),
    }


def main():
    daily_train, daily_holdout = load_data()
    train = city_month_climatology(daily_train)
    holdout = city_month_climatology(daily_holdout)
    X, y, groups = train[FEATURES], train[TARGET], train["city"]

    # Every fold contains cities the estimator has never seen.  The external
    # holdout remains untouched until model choice and fitting are complete.
    cv = GroupKFold(n_splits=5)
    rows = []
    print(f"Training on {len(train)} city-month climatologies from {groups.nunique()} cities...\n")
    for name, pipeline in candidates().items():
        scores = cross_validate(
            pipeline, X, y, groups=groups, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1,
        )
        row = {
            "model": name, "split": "grouped-city-cv",
            "mae": -scores["test_mae"].mean(),
            "rmse": -scores["test_rmse"].mean(),
            "r2": scores["test_r2"].mean(),
            "r2_std": scores["test_r2"].std(),
        }
        rows.append(row)
        print(f"  {name:18s} MAE={row['mae']:6.2f}  RMSE={row['rmse']:6.2f}  R2={row['r2']:6.3f} ± {row['r2_std']:.3f}")

    comparison = pd.DataFrame(rows).sort_values("rmse")
    best_name = comparison.iloc[0]["model"]
    best_pipeline = candidates()[best_name]
    best_pipeline.fit(X, y)

    holdout_predictions = best_pipeline.predict(holdout[FEATURES])
    final = metrics(holdout[TARGET], holdout_predictions)
    print(f"\nSelected by grouped CV: {best_name}")
    print(f"  [UNSEEN-cities-final] MAE={final['mae']:.2f}  RMSE={final['rmse']:.2f}  R2={final['r2']:.3f}")

    comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    pd.DataFrame([{"model": best_name, "split": "UNSEEN-cities-final", **final}]).to_csv(
        MODEL_DIR / "final_holdout_metrics.csv", index=False
    )
    joblib.dump(best_pipeline, MODEL_DIR / "aqi_model.joblib")
    joblib.dump(
        {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
         "target_granularity": "city-month-climatology"},
        MODEL_DIR / "feature_spec.joblib",
    )
    print(f"\nSaved model -> {MODEL_DIR / 'aqi_model.joblib'}")
    print(f"Saved comparison table -> {MODEL_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
