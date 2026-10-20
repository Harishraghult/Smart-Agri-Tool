"""
train_unet.py — Disease Segmentation (Lesion Area → Severity)

Trains a U-Net with a pretrained ResNet34 encoder for segmenting diseased
leaf regions (lesions). The lesion area percentage is then used to compute
a severity score.

WHY U-NET WITH PRETRAINED ENCODER
-----------------------------------
U-Net is the standard architecture for biomedical/agricultural segmentation.
Using a pretrained ResNet34 encoder (transfer learning) means we can train
with limited annotated data — PlantDoc Object Detection only provides bounding
boxes, which we convert to rough masks. A from-scratch U-Net would need
pixel-level annotations we don't have.

DATASET
-------
PlantDoc Object Detection Dataset (bounding box annotations).
Boxes are converted to rectangular masks for training. This is approximate
but sufficient for severity estimation (lesion area as % of leaf area).

USAGE
-----
    python training/disease_segmentation/train_unet.py \\
        --data-dir data/raw/plantdoc_objdet \\
        --output-dir models \\
        --epochs 20

    # Quick smoke test:
    python training/disease_segmentation/train_unet.py --epochs 1
"""

import argparse
import json
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256


# ─── U-Net Architecture ──────────────────────────────────────────────────

class UNetEncoder(nn.Module):
    """ResNet34 encoder for U-Net. Uses pretrained weights for transfer learning."""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x0 = self.layer0(x)   # /2
        x1 = self.layer1(self.pool0(x0))  # /4
        x2 = self.layer2(x1)  # /8
        x3 = self.layer3(x2)  # /16
        x4 = self.layer4(x3)  # /32
        return [x0, x1, x2, x3, x4]


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with pretrained ResNet34 encoder for lesion segmentation."""

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.encoder = UNetEncoder(pretrained=pretrained)
        self.decoder4 = UNetDecoderBlock(512, 256, 256)
        self.decoder3 = UNetDecoderBlock(256, 128, 128)
        self.decoder2 = UNetDecoderBlock(128, 64, 64)
        self.decoder1 = UNetDecoderBlock(64, 64, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        skips = self.encoder(x)  # [x0, x1, x2, x3, x4]
        d4 = self.decoder4(skips[4], skips[3])
        d3 = self.decoder3(d4, skips[2])
        d2 = self.decoder2(d3, skips[1])
        d1 = self.decoder1(d2, skips[0])
        out = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final(out)


# ─── Dataset ──────────────────────────────────────────────────────────────

class LesionSegmentationDataset(Dataset):
    """
    Loads images and creates binary masks from bounding box annotations.
    Expects PASCAL VOC or COCO format annotations alongside images.
    Falls back to generating masks from annotation files if available.
    """

    def __init__(self, image_dir, annotation_dir=None, img_size=256, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.samples = []

        image_dir = Path(image_dir)
        if not image_dir.exists():
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # Collect images
        for img_path in sorted(image_dir.rglob("*")):
            if img_path.suffix.lower() in image_exts:
                # Look for matching annotation
                ann_path = None
                if annotation_dir:
                    ann_dir = Path(annotation_dir)
                    # Try XML (PASCAL VOC) or JSON (COCO) or TXT (YOLO)
                    for ext in [".xml", ".json", ".txt"]:
                        candidate = ann_dir / f"{img_path.stem}{ext}"
                        if candidate.exists():
                            ann_path = candidate
                            break
                self.samples.append((img_path, ann_path))

    def _parse_annotation(self, ann_path, img_h, img_w):
        """Parse annotation file to get bounding boxes. Returns list of (x1,y1,x2,y2)."""
        if ann_path is None:
            return []
        boxes = []
        suffix = ann_path.suffix.lower()

        if suffix == ".xml":
            # PASCAL VOC format
            import xml.etree.ElementTree as ET
            tree = ET.parse(ann_path)
            for obj in tree.findall(".//object"):
                bbox = obj.find("bndbox")
                if bbox is not None:
                    x1 = int(float(bbox.find("xmin").text))
                    y1 = int(float(bbox.find("ymin").text))
                    x2 = int(float(bbox.find("xmax").text))
                    y2 = int(float(bbox.find("ymax").text))
                    boxes.append((x1, y1, x2, y2))

        elif suffix == ".txt":
            # YOLO format: class cx cy w h (normalized)
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - w/2) * img_w)
                        y1 = int((cy - h/2) * img_h)
                        x2 = int((cx + w/2) * img_w)
                        y2 = int((cy + h/2) * img_h)
                        boxes.append((x1, y1, x2, y2))

        return boxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size))
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)

        # Create mask from annotations
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        boxes = self._parse_annotation(ann_path, orig_h, orig_w)
        for (x1, y1, x2, y2) in boxes:
            # Scale to img_size
            sx1 = int(x1 * self.img_size / orig_w)
            sy1 = int(y1 * self.img_size / orig_h)
            sx2 = int(x2 * self.img_size / orig_w)
            sy2 = int(y2 * self.img_size / orig_h)
            mask[sy1:sy2, sx1:sx2] = 1.0

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return img_tensor, mask_tensor


# ─── Loss ─────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation. Dice handles class imbalance
    (healthy leaf area >> lesion area), BCE provides stable gradients."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        # BCE
        bce = F.binary_cross_entropy_with_logits(pred, target)
        # Dice
        smooth = 1e-5
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return bce + dice


# ─── Training ─────────────────────────────────────────────────────────────

def compute_iou(pred, target, threshold=0.5):
    """Compute IoU (Intersection over Union) for binary segmentation."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    if union == 0:
        return 1.0  # both empty
    return (intersection / union).item()


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_iou, n = 0, 0, 0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_iou += compute_iou(pred, masks) * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1), total_iou / max(n, 1)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_iou, n = 0, 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            pred = model(images)
            loss = criterion(pred, masks)
            total_loss += loss.item() * images.size(0)
            total_iou += compute_iou(pred, masks) * images.size(0)
            n += images.size(0)
    return total_loss / max(n, 1), total_iou / max(n, 1)


def compute_severity(mask_pred, threshold=0.5):
    """Convert segmentation mask to severity percentage.
    Severity = (lesion pixels / total leaf pixels) * 100"""
    binary = (mask_pred > threshold).float()
    lesion_area = binary.sum().item()
    total_area = binary.numel()
    return (lesion_area / total_area) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/raw/plantdoc_objdet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Find image and annotation directories
    img_dir = args.data_dir
    ann_dir = None
    for candidate in ["Annotations", "annotations", "labels", "Labels"]:
        if (args.data_dir / candidate).exists():
            ann_dir = args.data_dir / candidate
            break
    for candidate in ["JPEGImages", "images", "Images", "imgs"]:
        if (args.data_dir / candidate).exists():
            img_dir = args.data_dir / candidate
            break

    print(f"Images: {img_dir}")
    print(f"Annotations: {ann_dir}")

    # Create dataset
    full_dataset = LesionSegmentationDataset(img_dir, ann_dir, img_size=args.img_size)
    print(f"Total samples: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("No samples found! Check data directory structure.")
        return

    # Split into train/val
    n_val = max(1, int(len(full_dataset) * 0.15))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {n_train}, Val: {n_val}")

    # Build model
    model = UNet(num_classes=1, pretrained=True).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    best_iou = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_iou = val_epoch(model, val_loader, criterion)
        scheduler.step(val_iou)
        elapsed = time.time() - t0

        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f}  "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}  ({elapsed:.1f}s)")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_iou": train_iou,
            "val_loss": val_loss, "val_iou": val_iou,
        })

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = args.output_dir / "unet_lesion_segmentation.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou, "epoch": epoch,
                "img_size": args.img_size,
            }, ckpt_path)
            print(f"    -> new best (val_iou={val_iou:.4f}), saved to {ckpt_path}")

    # Save training log
    log_path = args.output_dir / "unet_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_iou", "val_loss", "val_iou"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nBest val IoU: {best_iou:.4f}")
    print(f"Model: {args.output_dir / 'unet_lesion_segmentation.pt'}")


if __name__ == "__main__":
    main()
