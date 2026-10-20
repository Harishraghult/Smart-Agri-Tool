"""
train.py — Wilting/Water-Stress Detector (Binary CNN)

Binary classifier: wilted/stressed vs. healthy leaves.
Used to detect water stress, which is then cross-referenced
against WeatherNext 3 rainfall forecasts to distinguish
drought stress from irrigation-system faults.

APPROACH
--------
- EfficientNet-B0 with binary output (healthy vs. wilted)
- Transfer learning from ImageNet weights
- Training data: healthy leaf images from PlantVillage +
  augmented/synthetic wilting samples (color shift, curl deformation)
- Confidence threshold: if softmax < 0.7, report "uncertain"

USAGE
-----
    python training/wilting_detector/train.py \\
        --data-dir data/processed/plantvillage \\
        --output-dir models \\
        --epochs 10
"""

import argparse
import csv
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Confidence threshold: below this, report "uncertain" rather than a
# potentially wrong binary prediction
CONFIDENCE_THRESHOLD = 0.7


class WiltingDataset(Dataset):
    """
    Creates a binary wilting dataset from existing healthy leaf images.
    Label 0 = healthy, Label 1 = wilted (synthetically augmented).
    
    Wilting augmentation: reduce green channel, increase yellowing,
    add slight blur (simulating water stress appearance changes).
    """

    def __init__(self, healthy_dirs, transform=None, augment_ratio=1.0):
        self.transform = transform
        self.samples = []  # (path, label, apply_wilt_aug)

        all_healthy = []
        for d in healthy_dirs:
            d = Path(d)
            if not d.exists():
                continue
            for img in d.rglob("*"):
                if img.suffix.lower() in IMAGE_EXTS:
                    all_healthy.append(img)

        # Real healthy samples
        for p in all_healthy:
            self.samples.append((p, 0, False))

        # Synthetic wilted samples (from same images, with augmentation)
        n_wilt = int(len(all_healthy) * augment_ratio)
        wilt_sources = random.choices(all_healthy, k=n_wilt) if all_healthy else []
        for p in wilt_sources:
            self.samples.append((p, 1, True))

        random.shuffle(self.samples)

    @staticmethod
    def apply_wilting_augmentation(img):
        """Simulate wilting: reduce saturation, shift to yellow/brown, add blur."""
        # Reduce green, boost yellow-brown
        r, g, b = img.split()
        g = g.point(lambda x: int(x * 0.6))  # reduce green
        r = r.point(lambda x: min(255, int(x * 1.2)))  # boost red/yellow
        img = Image.merge("RGB", (r, g, b))

        # Reduce saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.5)

        # Slight blur (wilted leaves often look less sharp)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Reduce brightness slightly
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, apply_wilt = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if apply_wilt:
            img = self.apply_wilting_augmentation(img)

        if self.transform:
            img = self.transform(img)

        return img, label


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def find_healthy_dirs(data_dir):
    """Find directories containing 'healthy' in their name."""
    data_dir = Path(data_dir)
    healthy_dirs = []
    for d in data_dir.rglob("*healthy*"):
        if d.is_dir():
            healthy_dirs.append(d)
    if not healthy_dirs:
        # Fallback: use all class dirs
        for split in ["train", "val"]:
            split_dir = data_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir() and "healthy" in class_dir.name.lower():
                        healthy_dirs.append(class_dir)
    return healthy_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Find healthy leaf directories
    healthy_dirs = find_healthy_dirs(args.data_dir)
    print(f"Found {len(healthy_dirs)} healthy leaf directories")

    if not healthy_dirs:
        print(f"No healthy leaf directories found in {args.data_dir}")
        return

    # Create datasets
    train_ds = WiltingDataset(healthy_dirs, transform=TRAIN_TRANSFORM, augment_ratio=1.0)
    print(f"Training samples: {len(train_ds)} (50% healthy, 50% synthetic wilt)")

    n_val = max(1, int(len(train_ds) * 0.15))
    n_train = len(train_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0

        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        log_rows.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = args.output_dir / "wilting_detector.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": ["healthy", "wilted"],
                "val_acc": val_acc, "epoch": epoch,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            }, ckpt_path)
            print(f"    -> saved to {ckpt_path}")

    # Save log
    log_path = args.output_dir / "wilting_detector_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Model: {args.output_dir / 'wilting_detector.pt'}")


if __name__ == "__main__":
    main()
