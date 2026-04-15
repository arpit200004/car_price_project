from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.data_loader import DATA_FILE, load_market_data


MODEL_FILE  = Path("models/car_price_model_v3.pkl")
REPORT_FILE = Path("reports/model_report.json")

CATEGORICAL_COLUMNS = [
    "brand",
    "model",
    "car_name",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "owner_type",
    "body_type",
    "drive_type",
]
NUMERIC_COLUMNS = [
    "car_age",
    "km_driven",
    "km_per_year",
    "mileage",
    "engine_cc",
    "max_power",
    "max_torque",
    "seats",
    "length",
    "width",
    "height",
]
FEATURE_COLUMNS  = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET_COLUMN    = "price"


def build_preprocessor_ordinal() -> ColumnTransformer:
    """Ordinal encoder — works well with Hist/ExtraTrees."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_COLUMNS,
            ),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    prep = build_preprocessor_ordinal()

    return {
        # ─── Histogram Gradient Boosting ─────────────────────────────
        # This is sklearn's fastest, most accurate tree booster.
        # Naturally handles missing values and works very well with ordinal cats.
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", prep),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=600,
                        max_depth=8,
                        learning_rate=0.06,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        max_leaf_nodes=63,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=30,
                        random_state=42,
                    ),
                ),
            ]
        ),
        # ─── Gradient Boosting ────────────────────────────────────────
        "gradient_boosting_regressor": Pipeline(
            steps=[
                ("preprocess", prep),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=800,
                        max_depth=7,
                        learning_rate=0.08,
                        subsample=0.85,
                        min_samples_leaf=10,
                        max_features=0.7,
                        random_state=42,
                    ),
                ),
            ]
        ),
        # ─── ExtraTrees (very fast, good baseline) ────────────────────
        "extra_trees_regressor": Pipeline(
            steps=[
                ("preprocess", prep),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_depth=35,
                        max_features=0.6,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    preds = np.clip(preds, a_min=0, a_max=None)

    nonzero = y_true != 0
    mape_val = (
        float(np.mean(np.abs((y_true[nonzero] - preds[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any() else 0.0
    )
    # Median Absolute Percentage Error — more robust to outliers
    median_ape = (
        float(np.median(np.abs((y_true[nonzero] - preds[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any() else 0.0
    )

    return {
        "mae":       float(mean_absolute_error(y_true, preds)),
        "mape":      mape_val,
        "median_ape": median_ape,
        "median_ae": float(median_absolute_error(y_true, preds)),
        "rmse":      float(np.sqrt(mean_squared_error(y_true, preds))),
        "r2":        float(r2_score(y_true, preds)),
        "bias":      float(np.mean(preds - y_true)),
    }


def main() -> None:
    df = load_market_data()
    
    print("\n" + "="*50)
    print("DATASET PREVIEW (first 10 rows):")
    print(df.head(10).to_string())
    print("="*50 + "\n")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        corr_cols = ["price", "car_age", "km_driven", "engine_cc", "max_power", "max_torque", "mileage"]
        corr_matrix = df[corr_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="YlOrBr", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        heatmap_path = "reports/correlation_heatmap.png"
        plt.savefig(heatmap_path)
        print(f"✅ Correlation Heatmap saved to '{heatmap_path}'\n")
    except ImportError:
        print("matplotlib or seaborn not installed, skipping heatmap generation.\n")
        
    X  = df[FEATURE_COLUMNS].copy()
    y  = np.log1p(df[TARGET_COLUMN].copy())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidate_metrics: dict[str, dict[str, float]] = {}
    for name, pipeline in build_candidates().items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        preds = np.expm1(pipeline.predict(X_test))
        candidate_metrics[name] = evaluate_predictions(np.expm1(y_test), preds)
        m = candidate_metrics[name]
        print(f"  MAE={m['mae']:,.0f}  MAPE={m['mape']:.2f}%  MedAPE={m['median_ape']:.2f}%  RMSE={m['rmse']:,.0f}  R²={m['r2']:.4f}  Bias={m['bias']:+,.0f}")

    # Pick winner: lowest MAPE
    winner_name = min(candidate_metrics, key=lambda n: candidate_metrics[n]["mape"])
    print(f"\nWinner: {winner_name}")

    # Retrain winner on 100% of data
    final_model = build_candidates()[winner_name]
    final_model.fit(X, y)

    bundle = {
        "model":              final_model,
        "data_file":          str(DATA_FILE),
        "feature_columns":    FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns":    NUMERIC_COLUMNS,
        "target_column":      TARGET_COLUMN,
        "training_rows":      int(len(df)),
        "selected_model_name": winner_name,
        "candidate_metrics":  candidate_metrics,
        "dataset_name":       "Kaggle Cardekho detailed used cars dataset (calibrated)",
        "price_calibrated":   True,
    }

    with MODEL_FILE.open("wb") as fh:
        pickle.dump(bundle, fh)

    report = {
        "data_file":          str(DATA_FILE),
        "dataset_name":       bundle["dataset_name"],
        "training_rows":      int(len(df)),
        "feature_columns":    FEATURE_COLUMNS,
        "selected_model_name": winner_name,
        "candidate_metrics":  candidate_metrics,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
