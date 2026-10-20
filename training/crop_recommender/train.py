"""
train.py — Crop Recommender (XGBoost)

Tabular ML model: N, P, K, temperature, humidity, pH, rainfall → crop label.

WHY XGBOOST
------------
- XGBoost consistently beats simpler models (RF, LR) on structured/tabular data
- Handles feature interactions naturally (nutrient ratios, pH×rainfall etc.)
- Fast training and inference on CPU — no GPU needed
- Built-in feature importance for explainability (which features matter most)
- SHAP integration for per-prediction explanations

DATASET
-------
Crop Recommendation Dataset (Kaggle: atharvaingle/crop-recommendation-dataset)
- 2200 rows, 22 crop classes
- Features: N, P, K (mg/kg), temperature (°C), humidity (%), ph, rainfall (mm)

USAGE
-----
    python training/crop_recommender/train.py
    python training/crop_recommender/train.py --data-dir data/processed/crop_recommendation
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=BASE_DIR / "data" / "processed" / "crop_recommendation")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "models")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = args.data_dir / "train.csv"
    val_path = args.data_dir / "val.csv"
    test_path = args.data_dir / "test.csv"

    if not train_path.exists():
        # Try loading the raw CSV and splitting inline
        raw_candidates = list((BASE_DIR / "data" / "raw" / "crop_recommendation").rglob("*.csv"))
        if not raw_candidates:
            print("No crop recommendation data found.")
            print("Run: python scripts/prepare_crop_tabular.py")
            return
        print(f"No pre-split data found. Using raw CSV: {raw_candidates[0]}")
        df = pd.read_csv(raw_candidates[0])
        from sklearn.model_selection import train_test_split
        label_col = [c for c in df.columns if c.lower() in ("label", "crop")]
        label_col = label_col[0] if label_col else df.columns[-1]
        train_df, test_df = train_test_split(df, test_size=0.15, stratify=df[label_col], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.176, stratify=train_df[label_col], random_state=42)
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    # Identify features and target
    label_col = None
    for candidate in ["label", "Label", "crop", "Crop"]:
        if candidate in train_df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = train_df.columns[-1]

    feature_cols = [c for c in train_df.columns if c != label_col]
    print(f"Features: {feature_cols}")
    print(f"Target: {label_col} ({train_df[label_col].nunique()} classes)")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val = le.transform(val_df[label_col])
    y_test = le.transform(test_df[label_col])

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Train XGBoost
    try:
        from xgboost import XGBClassifier
        print("\nTraining XGBoost...")
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            objective="multi:softmax",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )
    except ImportError:
        print("XGBoost not installed, falling back to RandomForest...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    # Evaluate
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:      {test_acc:.4f}")
    print(f"\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Save model
    model_path = args.output_dir / "crop_recommender.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "feature_cols": feature_cols}, f)
    print(f"Model saved: {model_path}")

    # Feature importance plot
    try:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_cols)),
                importances[sorted_idx],
                align="center")
        ax.set_yticks(range(len(feature_cols)))
        ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
        ax.set_xlabel("Feature Importance")
        ax.set_title("Crop Recommender — Feature Importance")
        ax.invert_yaxis()
        fig.tight_layout()

        fig_path = args.output_dir / "crop_recommender_feature_importance.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Feature importance plot: {fig_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

    # Save metadata
    metadata = {
        "model_type": "XGBClassifier" if "XGB" in type(model).__name__ else "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "feature_cols": feature_cols,
        "classes": list(le.classes_),
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    meta_path = args.output_dir / "crop_recommender_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
