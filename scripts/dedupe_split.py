"""
dedupe_split.py

Leakage-safe train/val/test split for PlantVillage and PlantDoc.

WHY THIS EXISTS
---------------
Neither PlantVillage nor PlantDoc ships metadata saying "these 4 photos
came from the same physical leaf / same photo session". PlantVillage in
particular is known to contain near-duplicate and augmented copies of the
same source image. A plain random train/test split can put near-identical
images on both sides of the split, which quietly inflates your reported
accuracy -- the model isn't generalising, it's half-memorising test images
it already saw a duplicate of in training.

HOW THIS SCRIPT FIXES IT
------------------------
1. Compute a perceptual hash (pHash) for every image, within each class.
2. Union-find images whose pHash Hamming distance is below a threshold
   into "duplicate clusters" (near-identical images, incl. simple
   rotations/crops/mirrors that pHash is robust to).
3. Split at the CLUSTER level, not the image level, so every image from
   the same source ends up entirely in train, entirely in val, or
   entirely in test -- never split across them.
4. Do this per-class so class balance is preserved as closely as
   possible despite the grouping constraint.
5. Copy (not move) files into
       data/processed/plantvillage/{train,val,test}/<class>/
       data/processed/plantdoc/{train,val,test}/<class>/
   and write a manifest CSV recording every decision.

USAGE
-----
    python dedupe_split.py --dataset plantvillage
    python dedupe_split.py --dataset plantdoc
    python dedupe_split.py --dataset plantvillage --hash-threshold 6

PlantDoc note
-------------
PlantDoc already ships a train/test split, BUT it has known near-duplicates
across that split. This script IGNORES the original split and re-splits from
scratch using the combined pool (train + test together), applying the same
cluster-level deduplication logic. This gives you a cleaner, reproducible,
leakage-free split you can cite in your report.
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

# --- Source directories (matching the actual folder names on disk) ---------
SOURCES = {
    "plantvillage": [BASE_DIR / "plantvillage dataset" / "data"],
    "plantdoc":     [BASE_DIR / "PlantDoc dataset" / "train",
                     BASE_DIR / "PlantDoc dataset" / "test"],
}

# --- Output directory for processed (split) data --------------------------
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# --------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15   # must sum to 1.0 with the two above


# ─── Union-Find ───────────────────────────────────────────────────────────

class UnionFind:
    """Standard union-find so duplicate images always end up in one cluster."""

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


# ─── Helpers ──────────────────────────────────────────────────────────────

def collect_class_images(source_dirs):
    """
    Walk all source_dirs and group image paths by class name.
    Returns: dict[class_name -> list[Path]]
    Handles both flat layouts (PlantVillage) and pre-split layouts (PlantDoc
    with train/ and test/ subdirectories): in both cases, the directory
    immediately containing the images is treated as the class label.
    """
    class_to_paths = defaultdict(list)
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"  [warn] source dir not found: {src_dir}")
            continue
        for class_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            images = [f for f in class_dir.iterdir()
                      if f.suffix.lower() in IMAGE_EXTS]
            if images:
                class_to_paths[class_dir.name].extend(images)
    return class_to_paths


def hash_images(image_paths):
    """Returns {path: ImageHash}. Skips unreadable files with a warning."""
    hashes = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                hashes[p] = imagehash.phash(img, hash_size=8)
        except Exception as e:
            print(f"  [warn] could not hash {p.name}: {e}")
    return hashes


def cluster_duplicates(hashes, threshold):
    """
    Groups images whose pHash Hamming distance <= threshold.
    O(n^2) per class -- fine for class sizes up to a few thousand.
    """
    uf = UnionFind()
    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] <= threshold:
                uf.union(paths[i], paths[j])

    clusters = defaultdict(list)
    for p in paths:
        clusters[uf.find(p)].append(p)
    return list(clusters.values())


def split_clusters(clusters):
    """
    Greedy size-balanced split at the cluster level.
    Deterministic (no RNG) -- re-running produces the identical split.
    """
    total_images = sum(len(c) for c in clusters)
    targets = {
        "train": TRAIN_FRAC * total_images,
        "val":   VAL_FRAC   * total_images,
        "test":  TEST_FRAC  * total_images,
    }
    assigned_counts = {"train": 0, "val": 0, "test": 0}
    assignment      = {"train": [], "val": [], "test": []}

    for cluster in sorted(clusters, key=len, reverse=True):
        deficits = {s: targets[s] - assigned_counts[s] for s in targets}
        chosen = max(deficits, key=deficits.get)
        assignment[chosen].append(cluster)
        assigned_counts[chosen] += len(cluster)

    return assignment


def process_class(class_name, image_paths, dataset_out_dir, threshold, manifest_rows):
    if not image_paths:
        return

    hashes     = hash_images(image_paths)
    clusters   = cluster_duplicates(hashes, threshold)
    n_dupes    = sum(len(c) - 1 for c in clusters if len(c) > 1)
    assignment = split_clusters(clusters)

    for split_name, split_clusters_list in assignment.items():
        split_dir = dataset_out_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for cluster in split_clusters_list:
            for src_path in cluster:
                dst_path = split_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                manifest_rows.append({
                    "class":        class_name,
                    "split":        split_name,
                    "filename":     src_path.name,
                    "cluster_size": len(cluster),
                })

    print(
        f"  {class_name:40s}  images={len(image_paths):5d}  "
        f"clusters={len(clusters):5d}  dupes={n_dupes:4d}  "
        f"-> train={sum(len(c) for c in assignment['train']):4d}  "
        f"val={sum(len(c) for c in assignment['val']):4d}  "
        f"test={sum(len(c) for c in assignment['test']):4d}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["plantvillage", "plantdoc"]
    )
    parser.add_argument(
        "--hash-threshold", type=int, default=8,
        help=(
            "Max Hamming distance for two images to count as near-duplicates "
            "(0 = exact match only, higher = more aggressive grouping). "
            "8 is a reasonable starting point for 8x8 pHash."
        ),
    )
    args = parser.parse_args()

    source_dirs = SOURCES[args.dataset]
    print(f"\nProcessing: {args.dataset}  (hash threshold={args.hash_threshold})")
    print(f"Source dirs: {[str(s) for s in source_dirs]}")

    class_to_paths = collect_class_images(source_dirs)
    if not class_to_paths:
        raise SystemExit(
            f"No class subfolders with images found. "
            f"Check that the dataset folders exist at the expected paths."
        )

    dataset_out_dir = PROCESSED_DIR / args.dataset
    manifest_rows   = []

    print(f"\n{'Class':40s}  {'images':>6}  {'clusters':>8}  {'dupes':>5}  "
          f"{'train':>5}  {'val':>4}  {'test':>4}")
    print("-" * 90)

    for class_name, image_paths in sorted(class_to_paths.items()):
        process_class(class_name, image_paths, dataset_out_dir,
                      args.hash_threshold, manifest_rows)

    manifest_path = dataset_out_dir / "split_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "split", "filename", "cluster_size"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = len(manifest_rows)
    train_n = sum(1 for r in manifest_rows if r["split"] == "train")
    val_n   = sum(1 for r in manifest_rows if r["split"] == "val")
    test_n  = sum(1 for r in manifest_rows if r["split"] == "test")

    print("-" * 90)
    print(f"Done.  Total images: {total}  |  train: {train_n}  val: {val_n}  test: {test_n}")
    print(f"Manifest : {manifest_path}")
    print(f"Output   : {dataset_out_dir}/train|val|test/<class>/")
    print(
        "\nKeep split_manifest.csv -- cite it in your report as evidence "
        "the split was done at the deduplicated-cluster level, not randomly."
    )


if __name__ == "__main__":
    main()
