"""
inspect_raw.py

Quick sanity check: counts images per class folder so you can see class
imbalance before you commit to a split. Run this before dedupe_split.py
so any surprises (missing folders, near-empty classes, non-image files)
show up early.

Usage:
    python inspect_raw.py
"""

from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent

# --- Actual paths on disk (adjust only if you move the folders) -----------
PLANTVILLAGE_DIR = BASE_DIR / "plantvillage dataset" / "data"
PLANTDOC_TRAIN_DIR = BASE_DIR / "PlantDoc dataset" / "train"
PLANTDOC_TEST_DIR  = BASE_DIR / "PlantDoc dataset" / "test"
# --------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images_per_class(root_dir: Path, label: str):
    """Count images in each direct class subfolder of root_dir."""
    counts = Counter()
    if not root_dir.exists():
        print(f"  (MISSING: {root_dir})")
        return counts

    for class_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if n > 0:
            counts[class_dir.name] = n

    if counts:
        total = sum(counts.values())
        print(f"\n=== {label} ===")
        print(f"Total images : {total:,}")
        print(f"Total classes: {len(counts)}")
        smallest = sorted(counts.items(), key=lambda kv: kv[1])[:5]
        print("Smallest classes (these might be too thin to split 3-ways):")
        for name, n in smallest:
            print(f"  {n:5d}  {name}")
    else:
        print(f"\n=== {label} === (no images found under {root_dir})")
    return counts


def main():
    count_images_per_class(PLANTVILLAGE_DIR,     "PlantVillage (38 classes)")
    count_images_per_class(PLANTDOC_TRAIN_DIR,   "PlantDoc / train  (pre-split)")
    count_images_per_class(PLANTDOC_TEST_DIR,    "PlantDoc / test   (pre-split)")


if __name__ == "__main__":
    main()
