"""
reorganize_by_disease.py

Reads the already-deduplicated-and-split data from data/processed/ and
regroups class folders by DISEASE keyword instead of by plant species.

For example:
  data/processed/plantvillage/train/Apple___Cedar_apple_rust/
  data/processed/plantvillage/train/Corn_(maize)___Common_rust_/
  data/processed/plantdoc/train/Apple_rust_leaf/
  ...all land in data/processed_by_disease/rust/train/<original_class_name>/

IMPORTANT -- split integrity guarantee
---------------------------------------
This script NEVER reassigns any image between train/val/test. Each image
is written to the SAME split bucket it already belongs to (the one
assigned by dedupe_split.py). The disease regrouping is purely a
reorganisation of the folder hierarchy, not a new split.

Disease keyword mapping
-----------------------
  rust          -- any class whose name contains "rust" (case-insensitive)
  blight        -- any class whose name contains "blight" or "scorch"
                   (Strawberry Leaf Scorch is botanically a blight-family
                   disease and is frequently grouped with blights in
                   disease-classification literature)
  powdery_mildew -- any class whose name contains "powdery_mildew" or
                   "powdery mildew"

Classes that match NONE of the above keywords are silently skipped
(healthy classes, mosaic virus, etc.). They are counted and reported at
the end so you can confirm nothing unexpected was dropped.

USAGE
-----
    python reorganize_by_disease.py
    python reorganize_by_disease.py --processed-dir data/processed
                                    --output-dir data/processed_by_disease
                                    --datasets plantvillage plantdoc

Needs: nothing beyond the stdlib (uses shutil.copy2, not hard links, for
       maximum Windows compatibility).
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Disease keyword rules  (order matters: first match wins)
# ---------------------------------------------------------------------------
DISEASE_RULES = [
    ("rust",           ["rust"]),
    ("blight",         ["blight", "scorch"]),
    ("powdery_mildew", ["powdery_mildew", "powdery mildew"]),
]


def classify_class_name(class_name: str):
    """Return the disease bucket name, or None if no rule matches."""
    lower = class_name.lower().replace("___", "_").replace("(", "").replace(")", "")
    for disease, keywords in DISEASE_RULES:
        if any(kw in lower for kw in keywords):
            return disease
    return None


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path,
                        default=BASE_DIR / "data" / "processed",
                        help="Root of dedupe_split.py output (default: data/processed)")
    parser.add_argument("--output-dir", type=Path,
                        default=BASE_DIR / "data" / "processed_by_disease",
                        help="Where to write disease-grouped folders (default: data/processed_by_disease)")
    parser.add_argument("--datasets", nargs="+",
                        default=["plantvillage", "plantdoc"],
                        help="Dataset subdirectory names to scan (default: plantvillage plantdoc)")
    args = parser.parse_args()

    if not args.processed_dir.exists():
        raise SystemExit(f"Processed dir not found: {args.processed_dir}\n"
                         f"Run dedupe_split.py first.")

    splits = ["train", "val", "test"]

    # disease -> split -> list of (src_class_folder, disease_class_subfolder_name)
    # We track this per disease/split so we can print counts before copying.
    plan = defaultdict(lambda: defaultdict(list))   # plan[disease][split] -> [(src_dir, dst_class)]
    skipped_classes = []

    print(f"\nScanning {args.processed_dir} ...")
    print(f"Datasets : {args.datasets}")
    print(f"Output   : {args.output_dir}\n")

    for dataset_name in args.datasets:
        dataset_dir = args.processed_dir / dataset_name
        if not dataset_dir.exists():
            print(f"  [warn] {dataset_dir} not found -- skipping")
            continue

        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                disease = classify_class_name(class_dir.name)
                if disease is None:
                    skipped_classes.append(f"{dataset_name}/{split}/{class_dir.name}")
                    continue
                # Namespace the destination class subfolder by dataset to avoid
                # collisions when both datasets have a class like "blight/train/Tomato..."
                dst_class = f"{dataset_name}__{class_dir.name}"
                plan[disease][split].append((class_dir, dst_class))

    # -----------------------------------------------------------------------
    # Print per-disease counts BEFORE copying so the user can check imbalance
    # -----------------------------------------------------------------------
    print("=" * 72)
    print(f"{'Disease':<18} {'split':<6} {'classes':>8} {'images':>8}")
    print("-" * 72)

    disease_split_counts = defaultdict(dict)   # disease -> split -> image_count
    for disease in sorted(plan):
        for split in splits:
            entries = plan[disease][split]
            n_images = sum(
                sum(1 for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
                for src_dir, _ in entries
            )
            n_classes = len(entries)
            disease_split_counts[disease][split] = n_images
            print(f"  {disease:<16} {split:<6} {n_classes:>8} {n_images:>8}")
        print()

    # Flag imbalance: warn if any disease's total images is < 50% of the
    # largest disease's total.
    disease_totals = {
        d: sum(disease_split_counts[d].values())
        for d in disease_split_counts
    }
    max_total = max(disease_totals.values()) if disease_totals else 1
    print("Imbalance check (vs largest disease bucket):")
    for disease, total in sorted(disease_totals.items(), key=lambda kv: -kv[1]):
        ratio = total / max_total
        flag = "  <<< HEAVILY IMBALANCED -- consider oversampling/class weights" if ratio < 0.5 else ""
        print(f"  {disease:<18} total={total:>6}  ratio={ratio:.2f}{flag}")

    print("\nProceed with copy? [y/N] ", end="", flush=True)
    answer = input().strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted -- no files were copied.")
        return

    # -----------------------------------------------------------------------
    # Copy files preserving split assignment
    # -----------------------------------------------------------------------
    total_copied = 0
    for disease in sorted(plan):
        for split in splits:
            for src_dir, dst_class in plan[disease][split]:
                dst_dir = args.output_dir / disease / split / dst_class
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img_path in src_dir.iterdir():
                    if img_path.suffix.lower() in IMAGE_EXTS:
                        shutil.copy2(img_path, dst_dir / img_path.name)
                        total_copied += 1

    print(f"\nDone. {total_copied:,} images copied to {args.output_dir}")

    # -----------------------------------------------------------------------
    # Final counts after copy (sanity-check that nothing was lost)
    # -----------------------------------------------------------------------
    print("\nFinal counts (post-copy):")
    print(f"{'Disease':<18} {'train':>8} {'val':>6} {'test':>6} {'total':>8}")
    print("-" * 52)
    for disease in sorted(plan):
        counts = {
            s: count_images(args.output_dir / disease / s)
            for s in splits
        }
        total = sum(counts.values())
        print(f"  {disease:<16} {counts['train']:>8} {counts['val']:>6} {counts['test']:>6} {total:>8}")

    # -----------------------------------------------------------------------
    # Report skipped classes
    # -----------------------------------------------------------------------
    unique_skipped = sorted(set(skipped_classes))
    # Deduplicate across splits for readability
    skipped_names = sorted(set(p.split("/", 1)[-1].rsplit("/", 1)[-1] for p in unique_skipped))
    print(f"\nSkipped classes (matched no disease keyword): {len(skipped_names)}")
    for name in skipped_names[:30]:
        print(f"  {name}")
    if len(skipped_names) > 30:
        print(f"  ... and {len(skipped_names) - 30} more")


if __name__ == "__main__":
    main()
