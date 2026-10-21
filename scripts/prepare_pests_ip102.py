"""
Build the expanded pest dataset: IP102 (102 crop pest species) + Agricultural Pests (12 classes)
+ "no pest" photos, as a manifest of image paths with leakage-checked splits.

Inputs
  data/raw/ip102_hf/*.parquet   IP102 from huggingface.co/datasets/EnmmmmOvO/insect-pest-dataset (MIT),
                                official split: train 45,095 / validation 7,508 / test 22,619;
                                label = index into IP102 classes.txt (0-based)
  data/processed/agricultural_pests/{train,val,test}/<class>/   existing pHash-split pest photos
  PlantVillage / PlantDoc / DeepWeeds "Negative" photos          -> label "no_pest"
Labels and pest groups: backend/app/knowledge_base/pest_types.json

Leakage guard: train/val images within --hash-threshold bits (dHash) of a test image, and train
images near a val image, are dropped (IP102 contains near-duplicate photos across its splits).

Usage:
    python scripts/prepare_pests_ip102.py
Output: data/processed/pests_ip102/images/<split>/<label>/*.jpg (IP102 only) + manifest.csv
"""

import argparse
import csv
import io
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from prepare_disease_types import dhash64, near_any  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS) if Path(folder).is_dir() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data/processed/pests_ip102")
    parser.add_argument("--hash-threshold", type=int, default=4)
    parser.add_argument("--negatives-per-split", type=str, default="3000,400,800")
    args = parser.parse_args()
    rows = []

    # 1. IP102 parquet shards -> JPEG files
    out_img = args.output_dir / "images"
    shards = sorted((args.data_dir / "raw/ip102_hf").glob("*.parquet"))
    assert len(shards) == 7, f"expected 7 IP102 parquet shards, found {len(shards)} (download incomplete?)"
    for shard in shards:
        split = SPLITS[shard.name.split("-")[0]]
        table = pq.read_table(shard)
        imgs, labels = table.column("image").to_pylist(), table.column("label").to_pylist()
        for i, (img, label) in enumerate(zip(imgs, labels)):
            dest = out_img / split / f"{label:03d}" / f"{shard.stem}_{i}.jpg"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = img["bytes"]
                try:
                    Image.open(io.BytesIO(data)).verify()
                except Exception:
                    continue  # skip unreadable images
                dest.write_bytes(data)
            rows.append({"path": str(dest), "label": f"ip102:{label}", "split": split, "source": "ip102"})
        print(f"{shard.name}: {len(labels)} images", flush=True)

    # 2. Agricultural Pests (existing leakage-safe split)
    ap = args.data_dir / "processed/agricultural_pests"
    for split in ("train", "val", "test"):
        for cls_dir in sorted((ap / split).iterdir()):
            for p in images(cls_dir):
                rows.append({"path": str(p), "label": f"agripests:{cls_dir.name}", "split": split, "source": "agripests"})

    # 3. "no pest" photos: lab leaves, field leaves, rangeland scenes (each from its own split)
    sys.path.insert(0, str(BASE_DIR / "training/field_classifiers"))
    import train as fc  # noqa: E402

    pools = fc.negative_pools(args.data_dir, {"plantvillage", "plantdoc", "deepweeds_negative"})
    for split, n in zip(("train", "val", "test"), map(int, args.negatives_per_split.split(","))):
        for p in fc.balanced_sample(pools[split], n):
            rows.append({"path": str(p), "label": "no_pest", "split": split, "source": "negatives"})

    # 4. Leakage guard across splits
    print(f"hashing {len(rows)} images...", flush=True)
    hashes = np.array([dhash64(r["path"]) or np.uint64(0) for r in rows], dtype=np.uint64)
    split_of = np.array([r["split"] for r in rows])
    drop = np.zeros(len(rows), dtype=bool)
    for split, ref_splits in (("train", ("val", "test")), ("val", ("test",))):
        idx = np.where(split_of == split)[0]
        refs = hashes[np.isin(split_of, ref_splits)]
        drop[idx[near_any(hashes[idx], refs, args.hash_threshold)]] = True
    print(f"dropped near-duplicates of held-out photos: {dict(Counter(split_of[drop]))}")
    rows = [r for r, d in zip(rows, drop) if not d]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"total: {dict(Counter(r['split'] for r in rows))}, labels: {len({r['label'] for r in rows})} "
          f"-> {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
