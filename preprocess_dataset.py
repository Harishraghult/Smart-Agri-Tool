"""
preprocess_dataset.py

ONE-FILE, preprocessing-only pipeline for a leaf image dataset. No
training code here -- this just takes you from raw, messy photos to a
clean, deduplicated, split, MobileNetV2-ready dataset on disk.

WHAT IT DOES (in order, per class)
-----------------------------------
1. Quality check   -- rejects unreadable files and too-blurry photos
                       (Laplacian-variance blur metric).
2. Cleanup          -- denoises, optionally removes background/crops to
                       the leaf's bounding box (saturation-based
                       segmentation), then resizes+pads to a square
                       (224x224 by default, MobileNetV2's input size)
                       WITHOUT distorting the leaf's shape.
3. Deduplication     -- perceptual-hashes (pHash) every cleaned image and
                       unions near-identical images (incl. simple
                       rotations/crops/mirrors) into clusters, so
                       duplicate/near-duplicate photos don't leak across
                       train/val/test.
4. Split             -- splits at the CLUSTER level (not image level) so
                       every image from the same source photo lands
                       entirely in one split, using a deterministic,
                       size-balanced greedy assignment (no RNG needed).
5. Manifest          -- writes a CSV recording every image's outcome
                       (kept/rejected/duplicate-cluster/split) for your
                       report's methodology section.

EXPECTED INPUT LAYOUT
----------------------
    raw_dataset/
        classA/
            img1.jpg
            img2.jpg
        classB/
            img1.jpg
        ...
(one subfolder per class, directly containing images)

OUTPUT LAYOUT
-------------
    processed_dataset/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
        preprocess_manifest.csv

USAGE
-----
    python preprocess_dataset.py --input raw_dataset --output processed_dataset

    # tune knobs:
    python preprocess_dataset.py --input raw_dataset --output processed_dataset \\
        --img-size 224 --blur-threshold 60 --hash-threshold 8 \\
        --train-frac 0.70 --val-frac 0.15 --test-frac 0.15

    # skip background removal (e.g. your photos already have clean backgrounds):
    python preprocess_dataset.py --input raw_dataset --output processed_dataset --skip-bg-removal

Needs: opencv-python, numpy, imagehash, Pillow, tqdm
    pip install opencv-python numpy imagehash Pillow tqdm --break-system-packages
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── Step 1: quality check ─────────────────────────────────────────────

def is_too_blurry(img_bgr, threshold):
    """Laplacian-variance blur metric: sharp images have high edge-content
    variance; blurry/out-of-focus photos are smoothed out (low variance)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance


# ─── Step 2: cleanup (denoise, background removal, resize+pad) ────────

def remove_background(img_rgb):
    """
    Heuristic leaf/background segmentation via Otsu thresholding on the
    HSV saturation channel (leaves are usually more saturated than plain
    backgrounds). Crops to the leaf's bounding box. Falls back to the
    original image if no confident leaf region is found, so a cluttered
    background never gets wrongly cropped into nothing.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 0.05 * img_rgb.shape[0] * img_rgb.shape[1]:
        return img_rgb  # implausibly small -> segmentation likely failed, keep original

    pad = int(0.05 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img_rgb.shape[1]), min(y + h + pad, img_rgb.shape[0])
    return img_rgb[y0:y1, x0:x1]


def resize_with_padding(img_rgb, target_size):
    """Aspect-ratio-preserving resize + pad to a square so leaf shape
    isn't distorted (matters for margin/shape-sensitive classification)."""
    h, w = img_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    top, left = (target_size - nh) // 2, (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def clean_image(src_path, dst_path, img_size, remove_bg, blur_threshold):
    """Runs quality-check + cleanup on one image. Returns (status, blur_variance)."""
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return "unreadable", None

    too_blurry, variance = is_too_blurry(img_bgr, blur_threshold)
    if too_blurry:
        return "rejected_blurry", variance

    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 15)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if remove_bg:
        img_rgb = remove_background(img_rgb)

    img_rgb = resize_with_padding(img_rgb, img_size)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return "ok", variance


# ─── Step 3: deduplication (union-find over pHash) ─────────────────────

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


def hash_images(image_paths):
    hashes = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                hashes[p] = imagehash.phash(img, hash_size=8)
        except Exception as e:
            print(f"  [warn] could not hash {p.name}: {e}")
    return hashes


def cluster_duplicates(hashes, threshold):
    """Groups images whose pHash Hamming distance <= threshold. O(n^2) per
    class -- fine for class sizes up to a few thousand."""
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


# ─── Step 4: leakage-safe split at cluster level ───────────────────────

def split_clusters(clusters, train_frac, val_frac, test_frac):
    """Deterministic, size-balanced greedy split -- no RNG, reproducible."""
    total = sum(len(c) for c in clusters)
    targets = {"train": train_frac * total, "val": val_frac * total, "test": test_frac * total}
    assigned = {"train": 0, "val": 0, "test": 0}
    assignment = {"train": [], "val": [], "test": []}

    for cluster in sorted(clusters, key=len, reverse=True):
        deficits = {s: targets[s] - assigned[s] for s in targets}
        chosen = max(deficits, key=deficits.get)
        assignment[chosen].append(cluster)
        assigned[chosen] += len(cluster)

    return assignment


# ─── Per-class pipeline ─────────────────────────────────────────────────

def process_class(class_name, image_paths, tmp_clean_dir, output_dir,
                   img_size, remove_bg, blur_threshold, hash_threshold,
                   train_frac, val_frac, test_frac, manifest_rows):
    if not image_paths:
        return

    # 1+2: quality check + cleanup -> write to a temp "cleaned" staging area
    cleaned_paths = []
    for src_path in tqdm(image_paths, desc=f"{class_name[:35]:35s}", unit="img"):
        dst_path = tmp_clean_dir / class_name / src_path.name
        status, variance = clean_image(src_path, dst_path, img_size, remove_bg, blur_threshold)
        if status == "ok":
            cleaned_paths.append(dst_path)
        else:
            manifest_rows.append({
                "class": class_name, "filename": src_path.name, "status": status,
                "blur_variance": f"{variance:.1f}" if variance is not None else "",
                "split": "", "cluster_size": "",
            })

    if not cleaned_paths:
        print(f"  [warn] no usable images left in class '{class_name}' after quality check")
        return

    # 3: dedup on the cleaned images (so hashing matches what the model sees)
    hashes = hash_images(cleaned_paths)
    clusters = cluster_duplicates(hashes, hash_threshold)
    n_dupes = sum(len(c) - 1 for c in clusters if len(c) > 1)

    # 4: split at cluster level
    assignment = split_clusters(clusters, train_frac, val_frac, test_frac)

    for split_name, split_clusters_list in assignment.items():
        split_dir = output_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for cluster in split_clusters_list:
            for cleaned_path in cluster:
                dst_path = split_dir / cleaned_path.name
                shutil.copy2(cleaned_path, dst_path)
                manifest_rows.append({
                    "class": class_name, "filename": cleaned_path.name, "status": "ok",
                    "blur_variance": "", "split": split_name, "cluster_size": len(cluster),
                })

    print(
        f"  {class_name:35s}  kept={len(cleaned_paths):5d}  clusters={len(clusters):5d}  "
        f"dupes={n_dupes:4d}  -> train={sum(len(c) for c in assignment['train']):4d}  "
        f"val={sum(len(c) for c in assignment['val']):4d}  "
        f"test={sum(len(c) for c in assignment['test']):4d}"
    )


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Raw dataset root (class subfolders inside)")
    parser.add_argument("--output", required=True, type=Path, help="Where to write processed train/val/test folders")
    parser.add_argument("--img-size", type=int, default=224, help="Output square size (MobileNetV2 default: 224)")
    parser.add_argument("--blur-threshold", type=float, default=60.0, help="Laplacian variance cutoff; lower = stricter")
    parser.add_argument("--hash-threshold", type=int, default=8, help="pHash Hamming distance cutoff for duplicates")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--skip-bg-removal", action="store_true", help="Skip background segmentation, just quality-check + resize")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the intermediate cleaned-but-unsplit images")
    args = parser.parse_args()

    if abs((args.train_frac + args.val_frac + args.test_frac) - 1.0) > 1e-6:
        raise SystemExit("--train-frac + --val-frac + --test-frac must sum to 1.0")
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    class_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {args.input}")

    tmp_clean_dir = args.output / "_tmp_cleaned"
    manifest_rows = []

    print(f"\nInput  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Image size: {args.img_size}x{args.img_size}   Background removal: {'OFF' if args.skip_bg_removal else 'ON'}")
    print(f"Blur threshold: {args.blur_threshold}   Hash threshold: {args.hash_threshold}")
    print(f"Split: train={args.train_frac} val={args.val_frac} test={args.test_frac}\n")
    print(f"{'Class':35s}  {'kept':>6}  {'clust':>6}  {'dupes':>5}  {'train':>5}  {'val':>4}  {'test':>4}")
    print("-" * 90)

    for class_dir in class_dirs:
        image_paths = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        process_class(
            class_dir.name, image_paths, tmp_clean_dir, args.output,
            args.img_size, not args.skip_bg_removal, args.blur_threshold,
            args.hash_threshold, args.train_frac, args.val_frac, args.test_frac,
            manifest_rows,
        )

    if not args.keep_temp and tmp_clean_dir.exists():
        shutil.rmtree(tmp_clean_dir)

    manifest_path = args.output / "preprocess_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "filename", "status", "blur_variance", "split", "cluster_size"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    kept = sum(1 for r in manifest_rows if r["status"] == "ok")
    rejected_blurry = sum(1 for r in manifest_rows if r["status"] == "rejected_blurry")
    unreadable = sum(1 for r in manifest_rows if r["status"] == "unreadable")
    train_n = sum(1 for r in manifest_rows if r["split"] == "train")
    val_n = sum(1 for r in manifest_rows if r["split"] == "val")
    test_n = sum(1 for r in manifest_rows if r["split"] == "test")

    print("-" * 90)
    print(f"Kept: {kept}   Rejected (blurry): {rejected_blurry}   Unreadable: {unreadable}")
    print(f"Split -> train: {train_n}   val: {val_n}   test: {test_n}")
    print(f"Manifest: {manifest_path}")
    print(f"Processed dataset: {args.output}/train|val|test/<class>/")


if __name__ == "__main__":
    main()