"""
prepare_weed_data.py

Prepares DeepWeeds and CropAndWeed datasets for weed detection/segmentation.

DeepWeeds (https://github.com/AlexOlsen/DeepWeeds):
  - 8 weed species from rangeland/pastoral environments + negative class
  - ~17.5k images captured in situ with GoPro cameras
  - Chosen for training weed classification model on real field conditions

CropAndWeed (https://github.com/cropandweed/cropandweed-dataset):
  - Crop-vs-weed segmentation dataset with pixel-level annotations
  - Multi-crop, multi-weed species annotations
  - Chosen for weed segmentation training (pixel masks available)

USAGE
-----
    python scripts/prepare_weed_data.py --dataset deepweeds
    python scripts/prepare_weed_data.py --dataset cropandweed
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

random.seed(42)


def collect_images_recursive(root_dir):
    """Find all class directories containing images."""
    class_to_paths = defaultdict(list)
    if not root_dir.exists():
        return class_to_paths
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            class_name = path.parent.name
            class_to_paths[class_name].append(path)
    return class_to_paths


def split_list(items, train_frac=0.70, val_frac=0.15):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def copy_to_split(images, dest_dir, class_name, split_name, manifest_rows):
    split_dir = dest_dir / split_name / class_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        dst = split_dir / src.name
        if dst.exists():
            dst = split_dir / f"{src.stem}_{random.randint(0, 9999)}{src.suffix}"
        shutil.copy2(src, dst)
        manifest_rows.append({
            "class": class_name, "split": split_name, "filename": dst.name
        })


def prepare_deepweeds(output_dir):
    """
    DeepWeeds: images/ folder with all images + labels.csv mapping filename→label.
    If labels.csv exists, use it. Otherwise, fall back to subdirectory structure.
    """
    raw = RAW_DIR / "deepweeds"
    manifest_rows = []

    # Check for labels CSV (DeepWeeds ships images flat with a CSV)
    labels_csv = None
    for candidate in raw.rglob("labels.csv"):
        labels_csv = candidate
        break

    if labels_csv and labels_csv.exists():
        print(f"  Using labels from: {labels_csv}")
        images_dir = labels_csv.parent / "images"
        if not images_dir.exists():
            images_dir = labels_csv.parent

        # Parse labels
        class_to_paths = defaultdict(list)
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("Filename", row.get("filename", ""))
                label = row.get("Label", row.get("label", row.get("Species", "")))
                img_path = images_dir / filename
                if img_path.exists():
                    class_to_paths[str(label)].append(img_path)
    else:
        # Fall back to subdirectory structure
        class_to_paths = collect_images_recursive(raw)

    if not class_to_paths:
        print(f"  No images found under {raw}")
        return []

    print(f"\n  {'Class':40s}  {'train':>5}  {'val':>4}  {'test':>4}")
    print("  " + "-" * 60)

    for class_name, images in sorted(class_to_paths.items()):
        train_imgs, val_imgs, test_imgs = split_list(images)
        copy_to_split(train_imgs, output_dir, class_name, "train", manifest_rows)
        copy_to_split(val_imgs, output_dir, class_name, "val", manifest_rows)
        copy_to_split(test_imgs, output_dir, class_name, "test", manifest_rows)
        print(f"  {class_name:40s}  {len(train_imgs):5d}  {len(val_imgs):4d}  {len(test_imgs):4d}")

    return manifest_rows


def prepare_cropandweed(output_dir):
    """
    CropAndWeed: segmentation dataset. Copy images and masks preserving structure.
    The dataset has images and annotations directories.
    """
    raw = RAW_DIR / "cropandweed"
    manifest_rows = []

    # Look for image directories
    class_to_paths = collect_images_recursive(raw)

    if not class_to_paths:
        print(f"  No class folders found under {raw}")
        print(f"  This dataset may need manual setup. Check the README in the repo.")
        return []

    print(f"\n  {'Class':40s}  {'train':>5}  {'val':>4}  {'test':>4}")
    print("  " + "-" * 60)

    for class_name, images in sorted(class_to_paths.items()):
        if len(images) < 3:
            print(f"  {class_name:40s}  skipped (only {len(images)} images)")
            continue
        train_imgs, val_imgs, test_imgs = split_list(images)
        copy_to_split(train_imgs, output_dir, class_name, "train", manifest_rows)
        copy_to_split(val_imgs, output_dir, class_name, "val", manifest_rows)
        copy_to_split(test_imgs, output_dir, class_name, "test", manifest_rows)
        print(f"  {class_name:40s}  {len(train_imgs):5d}  {len(val_imgs):4d}  {len(test_imgs):4d}")

    return manifest_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["deepweeds", "cropandweed"])
    args = parser.parse_args()

    output_dir = PROCESSED_DIR / args.dataset
    print(f"\nProcessing: {args.dataset}")
    print(f"Output: {output_dir}\n")

    if args.dataset == "deepweeds":
        manifest_rows = prepare_deepweeds(output_dir)
    else:
        manifest_rows = prepare_cropandweed(output_dir)

    if manifest_rows:
        manifest_path = output_dir / "split_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "split", "filename"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        total = len(manifest_rows)
        print(f"\n  Done. {total} images → {output_dir}")
        print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
