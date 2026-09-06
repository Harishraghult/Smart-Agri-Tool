"""
train.py — Ripeness Classifier (EfficientNet-B0)

IMPORTANT: This is a SEPARATE model track from the disease classifier.
Per project requirements, ripeness models use FRESH weights, NOT the
disease CNN weights. Different data, different visual features, different
classification task.

Two-stage training:
  Stage 1: Train on Fruits-360 (learn general fruit visual features)
  Stage 2: Fine-tune on Banana Ripeness (specialize to ripeness staging)

Also includes HSV/Lab color-space feature extraction for computing a
ripeness index alongside the CNN classification.

USAGE
-----
    python training/ripeness_classifier/train.py \\
        --fruits360-dir data/processed/fruits360 \\
        --banana-dir data/processed/banana_ripeness \\
        --output-dir models \\
        --epochs-stage1 10 --epochs-stage2 15

    # Smoke test:
    python training/ripeness_classifier/train.py --epochs-stage1 1 --epochs-stage2 1
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_dataloaders(dataset_dir, batch_size, num_workers=2):
    train_dir = Path(dataset_dir) / "train"
    val_dir = Path(dataset_dir) / "val"
    if not train_dir.exists():
        raise SystemExit(f"Expected {train_dir} to exist.")

    train_ds = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(val_dir, transform=EVAL_TRANSFORM) if val_dir.exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes, pretrained=True):
    """Fresh EfficientNet-B0 — NOT reusing disease classifier weights."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def replace_head(model, num_classes):
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if is_train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_stage(model, train_loader, val_loader, epochs, lr, ckpt_path, log_path, class_names):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_acc = 0.0
    log_rows = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = 0, 0
        if val_loader:
            val_loss, val_acc = run_epoch(model, val_loader, criterion)
            scheduler.step(val_acc)
        elapsed = time.time() - t0

        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        check_acc = val_acc if val_loader else train_acc
        if check_acc > best_acc:
            best_acc = check_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": check_acc, "epoch": epoch,
            }, ckpt_path)
            print(f"    -> saved to {ckpt_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)

    return best_acc


# ─── Color-space ripeness features ────────────────────────────────────────

def compute_ripeness_index(image_rgb):
    """
    Compute a ripeness index from HSV/Lab color features.
    Returns a dict with:
      - hsv_mean_h/s/v: mean hue, saturation, value
      - lab_mean_a/b: mean a* (green-red), b* (blue-yellow)
      - green_ratio: fraction of green pixels
      - yellow_ratio: fraction of yellow pixels
      - brown_ratio: fraction of brown pixels
      - ripeness_score: 0-100 composite score (higher = more ripe)
    """
    import cv2

    if isinstance(image_rgb, str) or isinstance(image_rgb, Path):
        img = cv2.imread(str(image_rgb))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_rgb

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    total_pixels = h.size

    # Color ratios (HSV-based segmentation)
    green_mask = (h > 25) & (h < 85) & (s > 40)
    yellow_mask = (h > 15) & (h < 35) & (s > 50)
    brown_mask = (h > 5) & (h < 25) & (s > 30) & (v < 180)

    green_ratio = green_mask.sum() / total_pixels
    yellow_ratio = yellow_mask.sum() / total_pixels
    brown_ratio = brown_mask.sum() / total_pixels

    # Ripeness score: more yellow/brown = more ripe, more green = less ripe
    ripeness_score = min(100, max(0,
        (1 - green_ratio) * 50 + yellow_ratio * 30 + brown_ratio * 20
    ) * 100)

    return {
        "hsv_mean_h": float(h.mean()),
        "hsv_mean_s": float(s.mean()),
        "hsv_mean_v": float(v.mean()),
        "lab_mean_a": float(a_ch.mean()),
        "lab_mean_b": float(b_ch.mean()),
        "green_ratio": float(green_ratio),
        "yellow_ratio": float(yellow_ratio),
        "brown_ratio": float(brown_ratio),
        "ripeness_score": float(ripeness_score),
    }


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruits360-dir", type=Path, default=Path("data/processed/fruits360"))
    parser.add_argument("--banana-dir", type=Path, default=Path("data/processed/banana_ripeness"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs-stage1", type=int, default=10)
    parser.add_argument("--epochs-stage2", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    stage1_ckpt = args.output_dir / "ripeness_stage1_fruits360.pt"
    stage2_ckpt = args.output_dir / "ripeness_stage2_banana.pt"

    # --- Stage 1: Fruits-360 ---
    if not args.skip_stage1:
        print("=== Stage 1: Training on Fruits-360 ===")
        if args.fruits360_dir.exists():
            train_loader, val_loader, classes = build_dataloaders(args.fruits360_dir, args.batch_size)
            print(f"  {len(classes)} classes")
            model = build_model(len(classes), pretrained=True)
            best = train_stage(model, train_loader, val_loader, args.epochs_stage1,
                             args.lr_stage1, stage1_ckpt,
                             args.output_dir / "ripeness_stage1_log.csv", classes)
            print(f"Stage 1 best: {best:.4f}\n")
        else:
            print(f"  {args.fruits360_dir} not found, skipping stage 1")
            print(f"  Run: python scripts/prepare_fruit_data.py --dataset fruits360\n")

    # --- Stage 2: Banana Ripeness ---
    print("=== Stage 2: Fine-tuning on Banana Ripeness ===")
    if args.banana_dir.exists():
        train_loader, val_loader, classes = build_dataloaders(args.banana_dir, args.batch_size)
        print(f"  {len(classes)} classes: {classes}")

        if stage1_ckpt.exists():
            ckpt = torch.load(stage1_ckpt, map_location=DEVICE)
            model = build_model(len(ckpt["class_names"]), pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = replace_head(model, len(classes))
            print("  Loaded stage 1 weights, replaced head")
        else:
            model = build_model(len(classes), pretrained=True)
            print("  No stage 1 checkpoint, starting from ImageNet weights")

        best = train_stage(model, train_loader, val_loader, args.epochs_stage2,
                         args.lr_stage2, stage2_ckpt,
                         args.output_dir / "ripeness_stage2_log.csv", classes)
        print(f"Stage 2 best: {best:.4f}")
        print(f"\nFinal ripeness model: {stage2_ckpt}")
    else:
        print(f"  {args.banana_dir} not found")
        print(f"  Run: python scripts/prepare_fruit_data.py --dataset banana_ripeness")


if __name__ == "__main__":
    main()
