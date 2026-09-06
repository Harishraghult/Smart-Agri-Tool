"""
train_yolo.py — Pest Detection (YOLOv8)

Trains a YOLOv8-nano model for pest detection and counting on agricultural
images. Uses transfer learning from COCO-pretrained weights.

WHY YOLOv8-NANO
----------------
- Real-time inference speed (important for field use on mobile/edge devices)
- Nano variant keeps model size small (~6MB) while still accurate enough
- Built-in support for bounding box detection + class + confidence
- The ultralytics library handles training loop, augmentation, and export

DATASETS
--------
- IP102: 102 pest species, ~75k images (primary)
- Agricultural Pests: 12 classes (supplementary, cleaner)

Both provide class-labeled images. For object detection, we need bounding
boxes. Strategy:
  1. If annotations exist (YOLO/VOC format), use them directly
  2. If only classification images, treat each image as a single full-image
     detection (the whole image IS the pest) — this trains a coarse detector
     that can be refined later with proper bbox annotations

USAGE
-----
    python training/pest_detector/train_yolo.py \\
        --data-dir data/processed/ip102 \\
        --output-dir models \\
        --epochs 50

    # Quick test:
    python training/pest_detector/train_yolo.py --epochs 1
"""

import argparse
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def create_yolo_dataset(data_dir, output_dir):
    """
    Convert a classification-style dataset (class subfolders) into YOLO
    detection format. Each image gets a full-image bounding box label.
    
    This is an approximation: we're telling YOLO "the entire image is this
    pest class". This works when images are tightly cropped to the pest
    (which IP102 and Agricultural Pests mostly are). For field-wide images
    with multiple pests, you'd need proper bbox annotations.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Discover classes
    classes = set()
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    classes.add(class_dir.name)

    classes = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"  Found {len(classes)} classes")

    # Create YOLO directory structure
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_idx = class_to_idx.get(class_dir.name)
            if class_idx is None:
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                # Copy image
                dst_img = img_out / f"{class_dir.name}_{img_path.name}"
                shutil.copy2(img_path, dst_img)

                # Create YOLO label (full-image bbox: center_x, center_y, w, h = 0.5, 0.5, 1.0, 1.0)
                lbl_path = lbl_out / f"{class_dir.name}_{img_path.stem}.txt"
                with open(lbl_path, "w") as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                count += 1

        print(f"  {split}: {count} images")

    # Write data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"  YOLO dataset config: {yaml_path}")
    return yaml_path, classes


def train_yolo(data_yaml, output_dir, epochs, batch_size, img_size):
    """Train YOLOv8-nano using the ultralytics library."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ultralytics not installed. Install with:")
        print("    pip install ultralytics")
        print("\n  Creating training script for manual execution...")
        script = f"""
# Run this after installing ultralytics:
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # pretrained nano
results = model.train(
    data='{data_yaml}',
    epochs={epochs},
    batch={batch_size},
    imgsz={img_size},
    project='{output_dir}',
    name='pest_detector',
    pretrained=True,
    patience=10,
    save=True,
    plots=True,
)
"""
        script_path = Path(output_dir) / "train_yolo_manual.py"
        script_path.write_text(script)
        print(f"  Manual training script: {script_path}")
        return

    print(f"\n  Starting YOLOv8-nano training...")
    model = YOLO("yolov8n.pt")  # pretrained COCO weights

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(output_dir),
        name="pest_detector",
        pretrained=True,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best model to standard location
    best_pt = Path(output_dir) / "pest_detector" / "weights" / "best.pt"
    if best_pt.exists():
        final = Path(output_dir) / "yolo_pest_detector.pt"
        shutil.copy2(best_pt, final)
        print(f"\n  Best model: {final}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/ip102"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert classification dataset to YOLO format
    yolo_dir = args.output_dir / "pest_yolo_dataset"
    print(f"\n=== Converting to YOLO format ===")
    print(f"  Source: {args.data_dir}")
    data_yaml, classes = create_yolo_dataset(args.data_dir, yolo_dir)

    # Step 2: Train
    print(f"\n=== Training YOLOv8-nano pest detector ===")
    print(f"  Classes: {len(classes)}")
    print(f"  Epochs: {args.epochs}")
    train_yolo(data_yaml, args.output_dir, args.epochs, args.batch_size, args.img_size)


if __name__ == "__main__":
    main()
