"""
train_yolo.py — Weed Detection (YOLOv8)

Trains a YOLOv8-nano model for weed vs. crop classification/detection.

DATASETS
--------
- DeepWeeds: 8 weed species + negative class, field photos
- CropAndWeed: Crop-vs-weed segmentation (used for detection here)

Same architecture rationale as pest detector — YOLOv8-nano for speed
and small model size suitable for field deployment.

USAGE
-----
    python training/weed_detector/train_yolo.py \\
        --data-dir data/processed/deepweeds \\
        --output-dir models \\
        --epochs 50
"""

import argparse
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def create_yolo_dataset(data_dir, output_dir):
    """Convert classification dataset to YOLO detection format."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    classes = set()

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    classes.add(class_dir.name)

    classes = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"  Found {len(classes)} classes: {classes[:10]}...")

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
                dst_img = img_out / f"{class_dir.name}_{img_path.name}"
                shutil.copy2(img_path, dst_img)
                lbl_path = lbl_out / f"{class_dir.name}_{img_path.stem}.txt"
                with open(lbl_path, "w") as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                count += 1
        print(f"  {split}: {count} images")

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
    return yaml_path, classes


def train_yolo(data_yaml, output_dir, epochs, batch_size, img_size):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ultralytics not installed. Install: pip install ultralytics")
        script = f"""
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='{data_yaml}', epochs={epochs}, batch={batch_size},
                       imgsz={img_size}, project='{output_dir}', name='weed_detector',
                       pretrained=True, patience=10, save=True)
"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "train_weed_manual.py").write_text(script)
        return

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml), epochs=epochs, batch=batch_size,
        imgsz=img_size, project=str(output_dir), name="weed_detector",
        pretrained=True, patience=10, save=True, verbose=True,
    )
    best_pt = Path(output_dir) / "weed_detector" / "weights" / "best.pt"
    if best_pt.exists():
        final = Path(output_dir) / "yolo_weed_detector.pt"
        shutil.copy2(best_pt, final)
        print(f"\n  Best model: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/deepweeds"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir = args.output_dir / "weed_yolo_dataset"

    print(f"\n=== Converting to YOLO format ===")
    data_yaml, classes = create_yolo_dataset(args.data_dir, yolo_dir)

    print(f"\n=== Training YOLOv8-nano weed detector ===")
    train_yolo(data_yaml, args.output_dir, args.epochs, args.batch_size, args.img_size)


if __name__ == "__main__":
    main()
