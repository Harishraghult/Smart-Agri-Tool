"""
Build the crop-independent disease-type dataset (manifest only, no image copies).

Labels from PlantVillage, PlantDoc (+ leaf crops) and PlantWild v1/v2 are mapped to disease types via
backend/app/knowledge_base/disease_types.json ("label_to_type"; null = excluded, e.g. fruit rots).

Splits (leakage-safe):
  * PlantVillage: existing pHash split (train/test); 5% of train per class held out as val
  * PlantDoc: existing train/val/test; leaf crops are train-only
  * PlantWild v1: official split (trainval.txt: 0=test, 1=train, 2=val)
  * PlantWild v2 photos not already in v1: grouped into near-duplicate clusters, each cluster
    assigned 80/10/10 to train/val/test by a stable hash, so v2-only types also get test photos
Cross-dataset guard: any train/val image within --hash-threshold bits (dHash) of any test image, and
any train image near a val image, is dropped. Over-represented PlantVillage classes are capped in train.

PlantWild licence: CC-BY-NC-ND 4.0 (non-commercial use only).

Usage:
    python scripts/prepare_disease_types.py
Output: data/processed/disease_types/manifest.csv  (path, type, split, source, label)
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = {"0": "test", "1": "train", "2": "val"}


def images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS) if Path(folder).is_dir() else []


def dhash64(path):
    try:
        a = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    except OSError:
        return None
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def near_any(query, reference, threshold, chunk=4096):
    """Boolean mask: query[i] is within `threshold` bits of some reference hash."""
    out = np.zeros(len(query), dtype=bool)
    if len(reference) == 0:
        return out
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk, None]
        out[i:i + chunk] = (np.bitwise_count(q ^ reference[None, :]) <= threshold).any(axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--mapping", type=Path, default=BASE_DIR / "backend/app/knowledge_base/disease_types.json")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data/processed/disease_types")
    parser.add_argument("--hash-threshold", type=int, default=6)
    parser.add_argument("--pv-train-cap", type=int, default=500, help="max train images per PlantVillage class")
    args = parser.parse_args()

    l2t = json.load(open(args.mapping, encoding="utf-8"))["label_to_type"]
    rng = random.Random(42)
    rows = []  # dicts: path, type, split, source, label

    def add(path, label, split, source):
        t = l2t[label]
        if t is not None:
            rows.append({"path": str(path), "type": t, "split": split, "source": source, "label": label})

    # PlantVillage
    pv = args.data_dir / "processed/plantvillage"
    for cls_dir in sorted((pv / "train").iterdir()):
        files = images(cls_dir)
        rng.shuffle(files)
        n_val = max(1, len(files) // 20)
        for p in files[:n_val]:
            add(p, cls_dir.name, "val", "plantvillage")
        for p in files[n_val:n_val + args.pv_train_cap]:
            add(p, cls_dir.name, "train", "plantvillage")
    train_classes = {d.name for d in (pv / "train").iterdir() if d.is_dir() and images(d)}
    resplit = []
    for cls_dir in sorted((pv / "test").iterdir()):
        if cls_dir.name in train_classes:
            for p in images(cls_dir):
                add(p, cls_dir.name, "test", "plantvillage")
            continue
        # The on-disk split put every image of this class in test (12 classes: all tomato, squash,
        # strawberry). Re-split 70/10/20 by near-duplicate cluster so the same leaf stays on one side.
        files = images(cls_dir)
        hashes = np.array([h if (h := dhash64(p)) is not None else np.uint64(0) for p in files], dtype=np.uint64)
        cluster = {}
        for i in range(len(files)):
            if i in cluster:
                continue
            for j in np.where(np.bitwise_count(hashes ^ hashes[i]) <= args.hash_threshold)[0]:
                cluster.setdefault(int(j), i)
        n_train = 0
        for i, p in enumerate(files):
            bucket = int(hashlib.md5(files[cluster[i]].name.encode()).hexdigest(), 16) % 10
            split = "test" if bucket < 2 else "val" if bucket == 2 else "train"
            if split == "train":
                if n_train >= args.pv_train_cap:
                    continue
                n_train += 1
            add(p, cls_dir.name, split, "plantvillage")
        resplit.append(cls_dir.name)
    print(f"PlantVillage classes re-split (were test-only on disk): {len(resplit)}")

    # PlantDoc whole photos + leaf crops
    pd = args.data_dir / "processed/plantdoc"
    for split in ("train", "val", "test"):
        for cls_dir in sorted((pd / split).iterdir()):
            for p in images(cls_dir):
                add(p, cls_dir.name, split, "plantdoc")
    for cls_dir in sorted((args.data_dir / "processed/plantdoc_crops/train").iterdir()):
        for p in images(cls_dir):
            add(p, cls_dir.name, "train", "plantdoc_crop")

    # PlantWild v1 (official split)
    pw = args.data_dir / "raw/plantwild"
    classes = {l.split(" ", 1)[0]: l.strip().split(" ", 1)[1]
               for l in open(pw / "plantwild/classes.txt", encoding="utf-8") if l.strip()}
    v1_paths = []
    for line in open(pw / "plantwild/trainval.txt", encoding="utf-8"):
        if line.strip():
            rel, cid, mode = line.strip().rsplit("=", 2)
            add(pw / "plantwild/images" / rel, classes[cid], SPLIT_NAMES[mode], "plantwild_v1")
            v1_paths.append(pw / "plantwild/images" / rel)
    print(f"collected {len(rows)} rows before PlantWild v2; hashing images (takes a few minutes)...")

    # Hash every non-crop image (crops are sub-images and can't be compared to whole photos)
    for r in rows:
        r["hash"] = dhash64(r["path"]) if r["source"] != "plantdoc_crop" else None
    v1_hashes = np.array([r["hash"] for r in rows if r["source"] == "plantwild_v1" and r["hash"] is not None],
                         dtype=np.uint64)

    # PlantWild v2: skip copies of v1 photos; cluster the rest and split by cluster
    v2_new = []
    for cls_dir in sorted(p for p in (pw / "plantwild_v2").iterdir() if p.is_dir()):
        if l2t.get(cls_dir.name) is None:
            continue
        for p in images(cls_dir):
            h = dhash64(p)
            if h is not None:
                v2_new.append((p, cls_dir.name, h))
    v2_hashes = np.array([h for *_, h in v2_new], dtype=np.uint64)
    is_v1_copy = near_any(v2_hashes, v1_hashes, args.hash_threshold)
    cluster_of = {}
    for i in range(len(v2_new)):  # greedy clustering of near-duplicates among new v2 photos
        if is_v1_copy[i] or i in cluster_of:
            continue
        members = np.where(np.bitwise_count(v2_hashes ^ v2_hashes[i]) <= args.hash_threshold)[0]
        for j in members:
            cluster_of.setdefault(int(j), i)
    added_v2 = 0
    for i, (p, label, h) in enumerate(v2_new):
        if is_v1_copy[i]:
            continue
        bucket = int(hashlib.md5(str(v2_new[cluster_of[i]][0].name).encode()).hexdigest(), 16) % 10
        split = "test" if bucket == 0 else "val" if bucket == 1 else "train"
        rows.append({"path": str(p), "type": l2t[label], "split": split, "source": "plantwild_v2",
                     "label": label, "hash": h})
        added_v2 += 1
    print(f"PlantWild v2: {added_v2} new photos added, {int(is_v1_copy.sum())} were copies of v1 photos")

    # Cross-dataset leakage guard
    def stack(split):
        return np.array([r["hash"] for r in rows if r["split"] == split and r["hash"] is not None], dtype=np.uint64)

    test_h, val_h = stack("test"), stack("val")
    drop = set()
    for split, refs in (("train", np.concatenate([test_h, val_h])), ("val", test_h)):
        idx = [i for i, r in enumerate(rows) if r["split"] == split and r["hash"] is not None]
        hit = near_any(np.array([rows[i]["hash"] for i in idx], dtype=np.uint64), refs, args.hash_threshold)
        drop |= {idx[k] for k in np.where(hit)[0]}
    dropped = Counter(rows[i]["split"] for i in drop)
    rows = [r for i, r in enumerate(rows) if i not in drop]
    print(f"dropped as near-duplicates of held-out photos: {dict(dropped)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "type", "split", "source", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    table = defaultdict(Counter)
    for r in rows:
        table[r["type"]][r["split"]] += 1
    print(f"\n{'type':18s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for t in sorted(table):
        print(f"{t:18s} {table[t]['train']:6d} {table[t]['val']:5d} {table[t]['test']:5d}")
    print(f"total: {Counter(r['split'] for r in rows)} -> {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
