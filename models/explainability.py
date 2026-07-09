"""
explainability.py
==================
Generates model explainability artifacts: permutation importance and
partial dependence curves.

NOTE ON SHAP: This sandbox cannot install new pip packages (no internet
access). SHAP specifically is unavailable here. However:
  - Permutation importance answers the same core question SHAP does
    ("how much does each feature actually matter for predictions"), and
    is computed natively in scikit-learn (no extra dependency).
  - Partial dependence shows the SHAPE of each feature's effect (e.g. "AQI
    rises sharply with industrial_index above 0.5"), which is what SHAP
    dependence plots also show.

>>> SWAP-IN POINT <<<
If you run this project on a machine WITH internet access, install shap
(`pip install shap`) and replace `run_permutation_importance()` below with:

    import shap
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    X_transformed = pipe.named_steps["preprocess"].transform(X_sample)
    shap_values = explainer.shap_values(X_transformed)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)

This gives per-PREDICTION attribution (not just global importance), which
is the fuller form of explainability SHAP is known for. The permutation
importance computed below is a legitimate, citable substitute for the
global-importance story, but per-row SHAP waterfall plots are the
upgrade path once you have internet access.
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance, partial_dependence

MODEL_DIR = "/Users/kevin/Downloads/aqi_project_3/models"
DATA_DIR = "/Users/kevin/Downloads/aqi_project_3/data"

NUMERIC_FEATURES = [
    "latitude", "longitude", "population_density", "is_coastal",
    "industrial_index", "month_sin", "month_cos",
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "precipitation_mm", "pressure_hpa",
]
CATEGORICAL_FEATURES = ["region"]


def city_month_climatology(df):
    """Match the city-month target granularity used during training."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    aggregations = {feature: "mean" for feature in NUMERIC_FEATURES}
    aggregations.update({"region": "first", "aqi": "mean"})
    return df.groupby(["city", "month"], as_index=False).agg(aggregations)


def run_permutation_importance():
    pipe = joblib.load(f"{MODEL_DIR}/aqi_model.joblib")
    holdout = city_month_climatology(pd.read_csv(f"{DATA_DIR}/holdout_cities_aqi.csv"))

    X = holdout[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = holdout["aqi"]

    predictions = pipe.predict(X)

    validation_df = pd.DataFrame({
        "city": holdout["city"],
        "actual": y,
        "predicted": predictions
    })

    validation_df = (
        validation_df
        .groupby("city", as_index=False)
        .mean(numeric_only=True)
    )

    result = permutation_importance(
        pipe, X, y, n_repeats=15, random_state=42, n_jobs=1, scoring="neg_mean_absolute_error"
    )

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp_df.to_csv(f"{MODEL_DIR}/permutation_importance.csv", index=False)
    feature_importance_json = imp_df.to_dict(orient="records")
    print("Permutation importance (computed on UNSEEN cities -- this measures")
    print("how much each feature matters for generalizing to new cities):\n")
    print(imp_df.to_string(index=False))
    return {
        "feature_importance": feature_importance_json,
        "validation": validation_df.to_dict(orient="records")
    }


def run_partial_dependence():
    """Shows the SHAPE of the relationship for the two features that matter
    most for cross-city generalization: industrial_index and population_density."""
    pipe = joblib.load(f"{MODEL_DIR}/aqi_model.joblib")
    train = city_month_climatology(pd.read_csv(f"{DATA_DIR}/train_cities_aqi.csv"))
    X = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    pd_results = {}
    for feature in [
    "latitude",
    "longitude",
    "population_density",
    "is_coastal",
    "industrial_index",
    "month_sin",
    "month_cos",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "pressure_hpa"
    ]:
        pdp = partial_dependence(pipe, X, [feature], kind="average", grid_resolution=20)
        pd_results[feature] = {
            "grid_values": pdp["grid_values"][0].tolist(),
            "average": pdp["average"][0].tolist(),
        }

    joblib.dump(pd_results, f"{MODEL_DIR}/partial_dependence.joblib")
    print("\nPartial dependence summary:")
    for feature, data in pd_results.items():
        lo, hi = data["average"][0], data["average"][-1]
        print(f"  {feature:20s}: AQI effect moves from {lo:.1f} to {hi:.1f} across its range")
    return pd_results


if __name__ == "__main__":

    importance_data = run_permutation_importance()
    pd_data = run_partial_dependence()

    explainability = {
        "feature_importance":
            importance_data["feature_importance"],

        "validation":
            importance_data["validation"],

        "partial_dependence":
            pd_data
    }

    with open(
        f"{DATA_DIR}/explainability.json",
        "w"
    ) as f:
        json.dump(explainability, f, indent=2)

    print("Saved explainability.json")
