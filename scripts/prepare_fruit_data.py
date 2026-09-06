"""
prepare_fruit_data.py

Prepares Fruits-360 and Banana Ripeness datasets for the ripeness classifier.

IMPORTANT: This is a SEPARATE model/dataset track from the leaf disease
classifier. Per project requirements, ripeness models use fresh weights,
NOT the disease CNN weights. Different data, different head, different
training script.

Fruits-360 (https://www.kaggle.com/datasets/moltean/fruits):
  - 131 fruit/vegetable classes, ~90k images, uniform white background
  - Chosen as the base for learning fruit visual features (shape, color,
    texture) before specializing to ripeness staging

Banana Ripeness (https://www.kaggle.com/datasets/mrmars1010/banana-ripeness-classification-dataset):
  - Banana images at different ripeness stages (unripe/ripe/overripe)
  - Chosen for fine-tuning ripeness stage classification on real data

PIPELINE
--------
1. Scan Fruits-360 for class subfolders (already has train/test split)
2. Create val split from train (80/20) preserving class balance
3. Scan Banana Ripeness for ripeness stage classes
4. Create train/val/test split for banana ripeness data
5. Write manifests

USAGE
-----
    python scripts/prepare_fruit_data.py --dataset fruits360
    python scripts/prepare_fruit_data.py --dataset banana_ripeness
"""

import argparse
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)  # reproducible splits


def collect_images_from_dir(root_dir):
    """Collect images grouped by class (subdirectory name)."""
    class_to_paths = defaultdict(list)
    if not root_dir.exists():
        return class_to_paths
    for class_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if images:
            class_to_paths[class_dir.name].extend(images)
    return class_to_paths


def split_list(items, train_frac=0.70, val_frac=0.15):
    """Split a list into train/val/test."""
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def copy_files(file_list, dest_dir, class_name, split_name, manifest_rows):
    """Copy files to dest_dir/split/class/ and record in manifest."""
    split_dir = dest_dir / split_name / class_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        dst = split_dir / src.name
        if dst.exists():
            dst = split_dir / f"{src.stem}_{random.randint(0,9999)}{src.suffix}"
        shutil.copy2(src, dst)
        manifest_rows.append({
            "class": class_name, "split": split_name, "filename": dst.name
        })


def prepare_fruits360(output_dir):
    """
    Fruits-360 already has Training/Test split.
    We keep those and carve a val set from Training (80/20).
    """
    raw = RAW_DIR / "fruits360"
    # Fruits-360 has structure: fruits-360/Training/ClassName/ and fruits-360/Test/ClassName/
    # Find the actual directories
    train_candidates = list(raw.rglob("Training"))
    test_candidates = list(raw.rglob("Test"))

    if not train_candidates:
        print(f"  Could not find Training/ folder under {raw}")
        print(f"  Expected: data/raw/fruits360/.../Training/<class>/")
        return

    train_root = train_candidates[0]
    test_root = test_candidates[0] if test_candidates else None
    manifest_rows = []

    print(f"  Train source: {train_root}")
    if test_root:
        print(f"  Test source:  {test_root}")

    train_classes = collect_images_from_dir(train_root)
    test_classes = collect_images_from_dir(test_root) if test_root else {}

    print(f"\n  {'Class':40s}  {'train':>5}  {'val':>4}  {'test':>4}")
    print("  " + "-" * 60)

    for class_name in sorted(train_classes.keys()):
        images = train_classes[class_name]
        random.shuffle(images)
        # 80% train, 20% val from the original training set
        n_val = max(1, int(len(images) * 0.2))
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        copy_files(train_imgs, output_dir, class_name, "train", manifest_rows)
        copy_files(val_imgs, output_dir, class_name, "val", manifest_rows)

        # Use original test set if available
        test_imgs = test_classes.get(class_name, [])
        if test_imgs:
            copy_files(test_imgs, output_dir, class_name, "test", manifest_rows)

        print(f"  {class_name:40s}  {len(train_imgs):5d}  {len(val_imgs):4d}  {len(test_imgs):4d}")

    return manifest_rows


def prepare_banana_ripeness(output_dir):
    """Split banana ripeness data into train/val/test (70/15/15)."""
    raw = RAW_DIR / "banana_ripeness"
    manifest_rows = []

    # Find class directories (may be nested)
    class_to_paths = defaultdict(list)
    if raw.exists():
        for d in raw.rglob("*"):
            if d.is_dir():
                images = [f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                if images and not any(
                    sub.is_dir() for sub in d.iterdir()
                ):
                    class_to_paths[d.name].extend(images)

    if not class_to_paths:
        # Fallback: direct subdirectories
        class_to_paths = collect_images_from_dir(raw)

    if not class_to_paths:
        print(f"  No class folders found under {raw}")
        return []

    print(f"\n  {'Class':40s}  {'train':>5}  {'val':>4}  {'test':>4}")
    print("  " + "-" * 60)

    for class_name, images in sorted(class_to_paths.items()):
        train_imgs, val_imgs, test_imgs = split_list(images)
        copy_files(train_imgs, output_dir, class_name, "train", manifest_rows)
        copy_files(val_imgs, output_dir, class_name, "val", manifest_rows)
        copy_files(test_imgs, output_dir, class_name, "test", manifest_rows)
        print(f"  {class_name:40s}  {len(train_imgs):5d}  {len(val_imgs):4d}  {len(test_imgs):4d}")

    return manifest_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["fruits360", "banana_ripeness"])
    args = parser.parse_args()

    output_dir = PROCESSED_DIR / args.dataset
    print(f"\nProcessing: {args.dataset}")
    print(f"Output: {output_dir}\n")

    if args.dataset == "fruits360":
        manifest_rows = prepare_fruits360(output_dir)
    else:
        manifest_rows = prepare_banana_ripeness(output_dir)

    if manifest_rows:
        manifest_path = output_dir / "split_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "split", "filename"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        total = len(manifest_rows)
        train_n = sum(1 for r in manifest_rows if r["split"] == "train")
        val_n = sum(1 for r in manifest_rows if r["split"] == "val")
        test_n = sum(1 for r in manifest_rows if r["split"] == "test")
        print(f"\n  Done. Total: {total} | train: {train_n}  val: {val_n}  test: {test_n}")
        print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
