"""
preprocess_intensity_spatial.py

Preprocessing-ONLY pipeline for already-split leaf image datasets.

PURPOSE
-------
Takes images that are ALREADY in a train/val/test split (e.g. from
dedupe_split.py or reorganize_by_disease.py) and applies DIP preprocessing
to each image IN-PLACE relative to its split, writing results to a parallel
output tree. It NEVER reassigns any image between splits.

PIPELINE (per image, in order)
--------------------------------
1. Median filter      -- denoises salt-and-pepper / sensor noise
                         (kernel 3x3 by default; raises to 5x5 for heavy noise)
2. CLAHE              -- Contrast Limited Adaptive Histogram Equalisation
                         applied to the L-channel in LAB space, so colour is
                         preserved while local contrast is boosted uniformly
                         (clipLimit 2.0, tileGridSize 8x8)
3. Unsharp mask       -- sharpens edges post-CLAHE to recover detail lost in
                         the denoising step (radius 1, amount 1.5, threshold 0)
4. Resize + pad       -- aspect-ratio-preserving resize to img_size x img_size
                         with white padding (default 224 for EfficientNet/MobileNet)

DEMO PANELS
-----------
Pass --demo-samples N to save a before/after comparison panel for N randomly
chosen images from EACH class (good for your report's figures section).
Panels are written to <output_dir>/_demo_panels/<split>/<class>/.

USAGE
-----
    # Preprocess a single disease folder:
    python preprocess_intensity_spatial.py \
        --input  data/processed_by_disease/rust \
        --output data/preprocessed/rust \
        --demo-samples 3

    # Run all diseases:
    for disease in rust blight powdery_mildew; do
        python preprocess_intensity_spatial.py \
            --input  data/processed_by_disease/$disease \
            --output data/preprocessed/$disease \
            --demo-samples 3
    done

    # On Windows PowerShell:
    foreach ($d in @("rust","blight","powdery_mildew")) {
        python preprocess_intensity_spatial.py `
            --input  "data/processed_by_disease/$d" `
            --output "data/preprocessed/$d" `
            --demo-samples 3
    }

Needs: opencv-python numpy Pillow tqdm
    pip install opencv-python numpy Pillow tqdm
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tqdm as tqdm_module

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# DIP pipeline steps
# ---------------------------------------------------------------------------

def apply_median_filter(img_bgr: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Step 1: Median filter for salt-and-pepper / sensor noise removal."""
    if ksize % 2 == 0:
        ksize += 1   # kernel must be odd
    return cv2.medianBlur(img_bgr, ksize)


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """
    Step 2: CLAHE on the L-channel (LAB colour space).
    Boosts local contrast without washing out colours or over-brightening
    uniform leaf regions (which plain histogram equalisation does).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_grid, tile_grid))
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(img_bgr: np.ndarray, radius: int = 1,
                        amount: float = 1.5) -> np.ndarray:
    """
    Step 3: Unsharp mask to recover edge sharpness lost during denoising.
    Uses Gaussian blur as the blur step (radius controls sigma).
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharpened = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    return sharpened


def resize_with_padding(img_bgr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Step 4: Aspect-ratio-preserving resize + white-pad to a square.
    Avoids shape distortion (matters for margin/shape-based lesion classification).
    """
    h, w = img_bgr.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh = max(int(h * scale), 1)
    nw = max(int(w * scale), 1)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    top  = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess_image(img_bgr: np.ndarray, img_size: int,
                     median_ksize: int = 3,
                     clahe_clip: float = 2.0,
                     clahe_tile: int = 8,
                     unsharp_radius: int = 1,
                     unsharp_amount: float = 1.5) -> np.ndarray:
    """Full pipeline: median -> CLAHE -> unsharp mask -> resize+pad."""
    img = apply_median_filter(img_bgr, ksize=median_ksize)
    img = apply_clahe(img, clip_limit=clahe_clip, tile_grid=clahe_tile)
    img = apply_unsharp_mask(img, radius=unsharp_radius, amount=unsharp_amount)
    img = resize_with_padding(img, target_size=img_size)
    return img


# ---------------------------------------------------------------------------
# Demo panel
# ---------------------------------------------------------------------------

def save_demo_panel(src_path: Path, dst_preprocessed: Path,
                    panel_dir: Path, img_size: int):
    """
    Save a side-by-side before/after panel:
      LEFT  -- original image (resized to img_size for visual parity)
      RIGHT -- preprocessed version
    """
    orig_bgr = cv2.imread(str(src_path))
    if orig_bgr is None:
        return
    orig_display  = resize_with_padding(orig_bgr, img_size)
    prepr_bgr     = cv2.imread(str(dst_preprocessed))
    if prepr_bgr is None:
        return

    # Add text labels
    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 200), 2, cv2.LINE_AA)
        return out

    left  = add_label(orig_display, "Before")
    right = add_label(prepr_bgr,    "After")

    divider = np.full((img_size, 4, 3), 180, dtype=np.uint8)
    panel = np.hstack([left, divider, right])

    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / f"panel_{src_path.stem}.jpg"
    cv2.imwrite(str(panel_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------

def process_split_class(src_class_dir: Path, dst_class_dir: Path,
                         demo_panel_dir: Path | None,
                         img_size: int, demo_samples: int,
                         median_ksize: int, clahe_clip: float,
                         clahe_tile: int, unsharp_radius: int,
                         unsharp_amount: float) -> tuple[int, int]:
    """
    Process all images in src_class_dir -> dst_class_dir.
    Returns (ok_count, fail_count).
    """
    dst_class_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [f for f in src_class_dir.iterdir()
                   if f.suffix.lower() in IMAGE_EXTS]
    if not image_paths:
        return 0, 0

    # Pick demo samples upfront (reproducible: sort then slice)
    demo_paths = set()
    if demo_panel_dir is not None and demo_samples > 0:
        demo_paths = set(sorted(image_paths)[:demo_samples])

    ok_count = fail_count = 0
    for src_path in image_paths:
        dst_path = dst_class_dir / src_path.name
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            fail_count += 1
            continue
        processed = preprocess_image(
            img_bgr, img_size,
            median_ksize=median_ksize,
            clahe_clip=clahe_clip,
            clahe_tile=clahe_tile,
            unsharp_radius=unsharp_radius,
            unsharp_amount=unsharp_amount,
        )
        cv2.imwrite(str(dst_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ok_count += 1

        if src_path in demo_paths and demo_panel_dir is not None:
            save_demo_panel(src_path, dst_path, demo_panel_dir, img_size)

    return ok_count, fail_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Source folder with train/val/test/<class>/ layout")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination folder (same layout, preprocessed images)")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Output square size in pixels (default: 224)")
    parser.add_argument("--demo-samples", type=int, default=0,
                        help="Save this many before/after demo panels per class "
                             "(0 = disabled). Panels saved under output/_demo_panels/")
    parser.add_argument("--median-ksize", type=int, default=3,
                        help="Median filter kernel size (must be odd, default 3)")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clip limit (default 2.0)")
    parser.add_argument("--clahe-tile", type=int, default=8,
                        help="CLAHE tile grid size (default 8)")
    parser.add_argument("--unsharp-radius", type=int, default=1,
                        help="Unsharp mask Gaussian radius/sigma (default 1)")
    parser.add_argument("--unsharp-amount", type=float, default=1.5,
                        help="Unsharp mask blend amount (default 1.5)")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    # Discover which splits are present in the input
    present_splits = [s for s in SPLITS if (args.input / s).exists()]
    if not present_splits:
        raise SystemExit(
            f"No train/val/test subfolders found under {args.input}.\n"
            f"Expected layout: <input>/train/<class>/, <input>/val/<class>/, ..."
        )

    demo_base = args.output / "_demo_panels" if args.demo_samples > 0 else None

    print(f"\nInput      : {args.input}")
    print(f"Output     : {args.output}")
    print(f"Image size : {args.img_size}x{args.img_size}")
    print(f"Pipeline   : median(k={args.median_ksize}) -> "
          f"CLAHE(clip={args.clahe_clip}, tile={args.clahe_tile}) -> "
          f"unsharp(r={args.unsharp_radius}, a={args.unsharp_amount}) -> "
          f"resize+pad({args.img_size})")
    print(f"Demo panels: {args.demo_samples} per class" if args.demo_samples > 0
          else "Demo panels: disabled")
    print()

    grand_ok = grand_fail = 0
    split_counts = {s: {"ok": 0, "fail": 0, "classes": 0, "thin_classes": []} for s in SPLITS}

    for split in present_splits:
        src_split = args.input / split
        dst_split = args.output / split
        class_dirs = sorted(p for p in src_split.iterdir() if p.is_dir())

        print(f"  [{split}]  {len(class_dirs)} class folders")
        for class_dir in tqdm_module.tqdm(class_dirs, desc=f"  {split:5s}", unit="class"):
            demo_dir = (demo_base / split / class_dir.name
                        if demo_base else None)
            ok, fail = process_split_class(
                class_dir, dst_split / class_dir.name, demo_dir,
                args.img_size, args.demo_samples,
                args.median_ksize, args.clahe_clip, args.clahe_tile,
                args.unsharp_radius, args.unsharp_amount,
            )
            split_counts[split]["ok"]     += ok
            split_counts[split]["fail"]   += fail
            split_counts[split]["classes"] += 1
            if ok < 10:
                split_counts[split]["thin_classes"].append((class_dir.name, ok))
            grand_ok   += ok
            grand_fail += fail

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Split':<8} {'classes':>8} {'images_ok':>10} {'images_fail':>12}")
    print("-" * 65)
    for split in SPLITS:
        c = split_counts[split]
        print(f"  {split:<6} {c['classes']:>8} {c['ok']:>10} {c['fail']:>12}")
    print("-" * 65)
    total_classes = sum(split_counts[s]["classes"] for s in SPLITS)
    print(f"  {'TOTAL':<6} {total_classes:>8} {grand_ok:>10} {grand_fail:>12}")

    # Thin-class warnings
    all_thin = [(split, cls, n)
                for split in SPLITS
                for cls, n in split_counts[split]["thin_classes"]]
    if all_thin:
        print("\n  *** THIN CLASS WARNING (< 10 images in a split) ***")
        for split, cls, n in all_thin:
            print(f"    [{split}] {cls} -> {n} images")
        print("  These classes are too small to trust metrics on.")
    else:
        print("\n  No thin classes (all splits have >= 10 images per class).")

    if demo_base:
        print(f"\n  Demo panels -> {demo_base}/")

    print(f"\nDone.  {grand_ok:,} images preprocessed -> {args.output}")


if __name__ == "__main__":
    main()
