"""
prepare_pest_data.py

Prepares IP102 and Agricultural Pests datasets for pest detector training.

IP102 (https://github.com/xpwu95/IP102):
  - 102 pest species, ~75k images, hierarchical taxonomy
  - Large-scale, research-grade dataset for pest classification
  - Chosen because it's the most comprehensive pest image dataset available

Agricultural Pests (https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset):
  - Simpler/smaller pest classification dataset (~12 classes)
  - Useful for supplementary training or quick prototyping
  - Chosen as a cleaner, more accessible complement to IP102

PIPELINE
--------
1. Scan raw directories for class subfolders
2. Apply pHash deduplication (same logic as disease pipeline)
3. Split at cluster level → train/val/test
4. Write split_manifest.csv

USAGE
-----
    python scripts/prepare_pest_data.py --dataset ip102
    python scripts/prepare_pest_data.py --dataset agricultural_pests
    python scripts/prepare_pest_data.py --dataset ip102 --hash-threshold 6
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCES = {
    "ip102": [BASE_DIR / "data" / "raw" / "ip102" / "data"],
    "agricultural_pests": [BASE_DIR / "data" / "raw" / "agricultural_pests"],
}

PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def collect_class_images(source_dirs):
    """Walk source dirs, group images by class (parent folder name)."""
    class_to_paths = defaultdict(list)
    for src_dir in source_dirs:
        if not src_dir.exists():
            # Try to find class folders recursively
            parent = src_dir.parent
            if parent.exists():
                for sub in parent.rglob("*"):
                    if sub.is_dir() and any(
                        f.suffix.lower() in IMAGE_EXTS for f in sub.iterdir() if f.is_file()
                    ):
                        images = [f for f in sub.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                        if images:
                            class_to_paths[sub.name].extend(images)
            if not class_to_paths:
                print(f"  [warn] source dir not found: {src_dir}")
            continue
        for class_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            if images:
                class_to_paths[class_dir.name].extend(images)
    return class_to_paths


def hash_images(image_paths):
    hashes = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                hashes[p] = imagehash.phash(img, hash_size=8)
        except Exception as e:
            pass  # skip unreadable
    return hashes


def cluster_duplicates(hashes, threshold):
    uf = UnionFind()
    paths = list(hashes.keys())
    # For large classes (>2000 images), skip pairwise dedup — too slow
    if len(paths) > 2000:
        return [[p] for p in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] <= threshold:
                uf.union(paths[i], paths[j])
    clusters = defaultdict(list)
    for p in paths:
        clusters[uf.find(p)].append(p)
    return list(clusters.values())


def split_clusters(clusters):
    total = sum(len(c) for c in clusters)
    targets = {"train": TRAIN_FRAC * total, "val": VAL_FRAC * total, "test": TEST_FRAC * total}
    assigned = {"train": 0, "val": 0, "test": 0}
    assignment = {"train": [], "val": [], "test": []}
    for cluster in sorted(clusters, key=len, reverse=True):
        deficits = {s: targets[s] - assigned[s] for s in targets}
        chosen = max(deficits, key=deficits.get)
        assignment[chosen].append(cluster)
        assigned[chosen] += len(cluster)
    return assignment


def process_class(class_name, image_paths, dataset_out_dir, threshold, manifest_rows):
    if not image_paths:
        return
    hashes = hash_images(image_paths)
    clusters = cluster_duplicates(hashes, threshold)
    assignment = split_clusters(clusters)

    for split_name, split_clusters_list in assignment.items():
        split_dir = dataset_out_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for cluster in split_clusters_list:
            for src_path in cluster:
                dst = split_dir / src_path.name
                # Handle name collisions from multiple source dirs
                if dst.exists():
                    dst = split_dir / f"{src_path.stem}_{hash(str(src_path)) % 10000}{src_path.suffix}"
                shutil.copy2(src_path, dst)
                manifest_rows.append({
                    "class": class_name, "split": split_name,
                    "filename": dst.name, "cluster_size": len(cluster),
                })

    n_train = sum(len(c) for c in assignment["train"])
    n_val = sum(len(c) for c in assignment["val"])
    n_test = sum(len(c) for c in assignment["test"])
    print(f"  {class_name:40s}  images={len(image_paths):5d}  "
          f"→ train={n_train:4d}  val={n_val:4d}  test={n_test:4d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ip102", "agricultural_pests"])
    parser.add_argument("--hash-threshold", type=int, default=8)
    args = parser.parse_args()

    source_dirs = SOURCES[args.dataset]
    print(f"\nProcessing: {args.dataset} (hash threshold={args.hash_threshold})")

    class_to_paths = collect_class_images(source_dirs)
    if not class_to_paths:
        print(f"No class subfolders found. Check that data/raw/{args.dataset}/ exists.")
        print(f"Run: python scripts/download_all_datasets.py")
        return

    dataset_out_dir = PROCESSED_DIR / args.dataset
    manifest_rows = []

    print(f"\n{'Class':40s}  {'images':>6}  {'train':>5}  {'val':>4}  {'test':>4}")
    print("-" * 70)

    for class_name, paths in tqdm(sorted(class_to_paths.items()), desc="Classes"):
        process_class(class_name, paths, dataset_out_dir, args.hash_threshold, manifest_rows)

    manifest_path = dataset_out_dir / "split_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "split", "filename", "cluster_size"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = len(manifest_rows)
    train_n = sum(1 for r in manifest_rows if r["split"] == "train")
    val_n = sum(1 for r in manifest_rows if r["split"] == "val")
    test_n = sum(1 for r in manifest_rows if r["split"] == "test")
    print("-" * 70)
    print(f"Done. Total: {total} | train: {train_n} val: {val_n} test: {test_n}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
