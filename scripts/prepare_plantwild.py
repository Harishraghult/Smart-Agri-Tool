"""
Map PlantWild (in-the-wild disease photos) onto the field-photo disease model's classes.

PlantWild: https://huggingface.co/datasets/uqtwei2/PlantWild  (licence CC-BY-NC-ND 4.0: non-commercial
use only; do not redistribute modified copies of the dataset). Expected layout after unzipping:
    data/raw/plantwild/plantwild/{images/<class>/..., classes.txt, trainval.txt}   (v1, 89 classes)
    data/raw/plantwild/plantwild_v2/<class>/...                                  (v2, 115 classes, no healthy)

Only classes with an equivalent in the field model / PlantVillage knowledge base are used. Five of
them are new to the field model (apple black rot, cherry powdery mildew, citrus greening, healthy
corn, strawberry leaf scorch).

Splits and leakage guards (perceptual dHash, Hamming distance <= --hash-threshold):
  * v1 images keep PlantWild's official split (trainval.txt: 0=test, 1=train, 2=val)
  * a v2 image that duplicates a v1 image inherits that image's split; other v2 images -> train
  * any image duplicating a PlantDoc val/test photo is dropped (keeps the PlantDoc benchmark clean)
  * train images duplicating a PlantWild val/test image are dropped

Usage:
    python scripts/prepare_plantwild.py
Output: data/processed/plantwild_mapped/{train,val,test}/<field class>/*.jpg + manifest.csv
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = {"0": "test", "1": "train", "2": "val"}

# PlantWild class -> field-model class (PlantDoc naming; new classes use PlantDoc-style names too)
CLASS_MAP = {
    "apple leaf": "Apple_leaf", "apple rust": "Apple_rust_leaf", "apple scab": "Apple_Scab_Leaf",
    "apple black rot": "Apple_black_rot",  # new
    "bell pepper leaf": "Bell_pepper_leaf", "bell pepper leaf spot": "Bell_pepper_leaf_spot",
    "bell pepper bacterial spot": "Bell_pepper_leaf_spot",
    "blueberry leaf": "Blueberry_leaf", "cherry leaf": "Cherry_leaf",
    "cherry powdery mildew": "Cherry_powdery_mildew",  # new
    "citrus greening disease": "Citrus_greening",  # new
    "corn gray leaf spot": "Corn_Gray_leaf_spot", "corn northern leaf blight": "Corn_leaf_blight",
    "corn rust": "Corn_rust_leaf", "corn leaf": "Corn_leaf",  # new (healthy)
    "grape black rot": "grape_leaf_black_rot", "grape leaf": "grape_leaf",
    "peach leaf": "Peach_leaf",
    "potato early blight": "Potato_leaf_early_blight", "potato late blight": "Potato_leaf_late_blight",
    "potato leaf": "Potato_leaf",
    "raspberry leaf": "Raspberry_leaf", "soybean leaf": "Soyabean_leaf",
    "squash powdery mildew": "Squash_Powdery_mildew_leaf",
    "strawberry leaf": "Strawberry_leaf", "strawberry leaf scorch": "Strawberry_leaf_scorch",  # new
    "tomato bacterial leaf spot": "Tomato_leaf_bacterial_spot", "tomato early blight": "Tomato_Early_blight_leaf",
    "tomato late blight": "Tomato_leaf_late_blight", "tomato leaf": "Tomato_leaf",
    "tomato leaf mold": "Tomato_mold_leaf", "tomato mosaic virus": "Tomato_leaf_mosaic_virus",
    "tomato septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato_leaf_yellow_virus",
}


def dhash(path, size=8):
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except OSError:
        return None
    a = np.asarray(img, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def near(h, hashes, threshold):
    """Index of the nearest hash within threshold, else None."""
    if h is None or len(hashes) == 0:
        return None
    d = np.unpackbits(np.bitwise_xor(hashes, h), axis=1).sum(axis=1)
    i = int(d.argmin())
    return i if d[i] <= threshold else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantwild-dir", type=Path, default=BASE_DIR / "data/raw/plantwild")
    parser.add_argument("--plantdoc-dir", type=Path, default=BASE_DIR / "data/processed/plantdoc")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data/processed/plantwild_mapped")
    parser.add_argument("--hash-threshold", type=int, default=6)
    args = parser.parse_args()

    v1_root = args.plantwild_dir / "plantwild"
    classes = {l.split(" ", 1)[0]: l.strip().split(" ", 1)[1]
               for l in open(v1_root / "classes.txt", encoding="utf-8") if l.strip()}

    # v1 with official split
    items = []  # (path, field_class, split, source)
    for line in open(v1_root / "trainval.txt", encoding="utf-8"):
        if not line.strip():
            continue
        rel, cid, mode = line.strip().rsplit("=", 2)
        cls = CLASS_MAP.get(classes[cid])
        if cls:
            items.append((v1_root / "images" / rel, cls, SPLIT_NAMES[mode], "v1"))
    v1_items = list(items)
    print(f"v1 mapped images: {len(v1_items)}")

    # Hash everything once
    pd_held = [p for s in ("val", "test") for p in (args.plantdoc_dir / s).rglob("*") if p.suffix.lower() in EXTS]
    pd_hashes = np.stack([h for h in (dhash(p) for p in pd_held) if h is not None])
    v1_hash = [dhash(p) for p, *_ in v1_items]
    valid_v1 = [i for i, h in enumerate(v1_hash) if h is not None]
    v1_stack = np.stack([v1_hash[i] for i in valid_v1])

    # v2: inherit the split of a v1 duplicate, else train
    v2_root = args.plantwild_dir / "plantwild_v2"
    v2_dups = 0
    for d in sorted(p for p in v2_root.iterdir() if p.is_dir()):
        cls = CLASS_MAP.get(d.name)
        if not cls:
            continue
        for p in sorted(x for x in d.iterdir() if x.suffix.lower() in EXTS):
            j = near(dhash(p), v1_stack, args.hash_threshold)
            if j is not None:
                v2_dups += 1
                continue  # the v1 copy already carries this photo and its split
            items.append((p, cls, "train", "v2"))
    print(f"v2 mapped images added: {len(items) - len(v1_items)} (skipped {v2_dups} duplicates of v1 photos)")

    # Leakage filters
    hashes = [v1_hash[i] if i < len(v1_items) else dhash(it[0]) for i, it in enumerate(items)]
    held_idx = [i for i, it in enumerate(items) if it[2] in ("val", "test") and hashes[i] is not None]
    held_stack = np.stack([hashes[i] for i in held_idx]) if held_idx else np.zeros((0, 8), np.uint8)
    kept, dropped_pd, dropped_pw = [], 0, 0
    for i, (p, cls, split, src) in enumerate(items):
        h = hashes[i]
        if h is None:
            continue
        if near(h, pd_hashes, args.hash_threshold) is not None:
            dropped_pd += 1
            continue
        if split == "train" and near(h, held_stack, args.hash_threshold) is not None:
            dropped_pw += 1
            continue
        kept.append((p, cls, split, src))
    print(f"dropped as duplicates of PlantDoc val/test: {dropped_pd}; train dropped as PlantWild val/test dupes: {dropped_pw}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    rows = []
    for k, (p, cls, split, src) in enumerate(kept):
        dest = args.output_dir / split / cls / f"{src}_{k}{p.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        rows.append({"split": split, "class": cls, "source": src, "original": str(p.relative_to(args.plantwild_dir)),
                     "file": dest.name})
    with open(args.output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["split"]) for r in rows)
    print(f"written: {dict(counts)} across {len({r['class'] for r in rows})} classes -> {args.output_dir}")


if __name__ == "__main__":
    main()
