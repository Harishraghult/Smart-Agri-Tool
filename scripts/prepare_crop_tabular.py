"""
prepare_crop_tabular.py

Prepares the Crop Recommendation tabular dataset for the crop recommender model.

Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
  - 2200 rows, 7 features: N, P, K, temperature, humidity, ph, rainfall
  - 22 crop labels (rice, maize, chickpea, kidneybeans, etc.)
  - Chosen because it's the standard benchmark for tabular crop recommendation
    and covers the exact feature set available from soil tests + weather data

No pHash deduplication needed — this is tabular data, not images.
Standard stratified train/val/test split is appropriate.

USAGE
-----
    python scripts/prepare_crop_tabular.py
    python scripts/prepare_crop_tabular.py --input data/raw/crop_recommendation/Crop_recommendation.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent


def find_csv(raw_dir):
    """Find the crop recommendation CSV in raw data."""
    candidates = list(raw_dir.rglob("*.csv"))
    for c in candidates:
        if "crop" in c.name.lower() or "recommendation" in c.name.lower():
            return c
    if candidates:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to the Crop Recommendation CSV. Auto-detected if not specified."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BASE_DIR / "data" / "processed" / "crop_recommendation",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Find input CSV
    if args.input and args.input.exists():
        csv_path = args.input
    else:
        raw_dir = BASE_DIR / "data" / "raw" / "crop_recommendation"
        csv_path = find_csv(raw_dir)
        if csv_path is None:
            print(f"Could not find crop recommendation CSV in {raw_dir}")
            print("Run: python scripts/download_all_datasets.py")
            return

    print(f"Input CSV: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Identify the target column (usually 'label' or 'crop')
    label_col = None
    for candidate in ["label", "Label", "crop", "Crop"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        # Use the last column as label
        label_col = df.columns[-1]
    print(f"Target column: '{label_col}' ({df[label_col].nunique()} classes)")
    print(f"\nClass distribution:\n{df[label_col].value_counts().to_string()}\n")

    # Stratified train/val/test split
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df[label_col]
    )
    relative_val = args.val_size / (1 - args.test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=relative_val, random_state=args.seed, stratify=train_df[label_col]
    )

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(args.output_dir / "val.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    print(f"Split sizes:")
    print(f"  train: {len(train_df):5d} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  val:   {len(val_df):5d} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  test:  {len(test_df):5d} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nOutput: {args.output_dir}/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
