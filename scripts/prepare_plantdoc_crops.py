"""
Cut individual-leaf crops out of the PlantDoc object-detection dataset (Pascal VOC boxes)
to add training data for the field-photo disease classifier.

Leakage guard: the detection dataset and the PlantDoc classification dataset contain the same
source photos under different filenames. Every detection photo whose perceptual hash (dHash)
is within --hash-threshold bits of any photo in data/processed/plantdoc/{val,test} is skipped,
so no evaluation photo leaks into training via its crops.

Usage:
    python scripts/prepare_plantdoc_crops.py
Output: data/processed/plantdoc_crops/train/<class>/*.jpg + crops_manifest.csv
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png"}


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def hamming(h, hashes):
    return np.unpackbits(np.bitwise_xor(hashes, h), axis=1).sum(axis=1)


def normalise(name):
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objdet-dir", type=Path,
                        default=BASE_DIR / "data/raw/plantdoc_objdet/PlantDoc-Object-Detection-Dataset-master")
    parser.add_argument("--plantdoc-dir", type=Path, default=BASE_DIR / "data/processed/plantdoc")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data/processed/plantdoc_crops")
    parser.add_argument("--hash-threshold", type=int, default=10)
    parser.add_argument("--min-side", type=int, default=40)
    parser.add_argument("--pad", type=float, default=0.08)
    args = parser.parse_args()

    classes = sorted(p.name for p in (args.plantdoc_dir / "train").iterdir() if p.is_dir())
    by_norm = {normalise(c): c for c in classes}
    # Labelled in the detection set but absent from the classification set: healthy potato leaves.
    # Without it, every healthy potato photo is forced into a disease class.
    by_norm.setdefault(normalise("Potato leaf"), "Potato_leaf")

    held_out = [p for s in ("val", "test") for p in (args.plantdoc_dir / s).rglob("*") if p.suffix.lower() in EXTS]
    held_hashes = np.stack([dhash(p) for p in held_out])
    print(f"{len(classes)} classes; hashed {len(held_out)} held-out val/test photos")

    out_root = args.output_dir / "train"
    rows, skipped_leak, unknown, n_photos = [], 0, set(), 0
    for split_dir in ("TRAIN", "TEST"):
        for xml_path in sorted((args.objdet_dir / split_dir).glob("*.xml")):
            img_path = xml_path.with_suffix(".jpg")
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except OSError:
                continue
            if hamming(dhash(img_path), held_hashes).min() <= args.hash_threshold:
                skipped_leak += 1
                continue
            n_photos += 1
            W, H = img.size
            for i, obj in enumerate(ET.parse(xml_path).getroot().iter("object")):
                cls = by_norm.get(normalise(obj.findtext("name", "")))
                if cls is None:
                    unknown.add(obj.findtext("name", "").strip())
                    continue
                bb = obj.find("bndbox")
                x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                px, py = (x1 - x0) * args.pad, (y1 - y0) * args.pad
                box = (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))
                if min(box[2] - box[0], box[3] - box[1]) < args.min_side:
                    continue
                dest = out_root / cls / f"{split_dir.lower()}_{xml_path.stem}_{i}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                img.crop(box).save(dest, quality=95)
                rows.append({"class": cls, "source": f"{split_dir}/{img_path.name}", "crop": dest.name})

    with open(args.output_dir / "crops_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "source", "crop"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"photos used: {n_photos}, skipped as val/test duplicates: {skipped_leak}, crops written: {len(rows)}")
    if unknown:
        print("unmapped labels (ignored):", sorted(unknown))


if __name__ == "__main__":
    main()
