"""
Image-type router (EfficientNet-B0): decides which model should see an upload.

Classes:
  lab_leaf    - single leaf on a plain background (PlantVillage)  -> PlantVillage disease model
  field_leaf  - leaf photographed in the field (PlantDoc)         -> PlantDoc fine-tuned model
  fruit       - banana / fruit close-up (Banana Ripeness)          -> ripeness model only
  other       - insects, rangeland scenes, anything else           -> flagged as "not a leaf"

Why: the PlantVillage classifier is 99.5% accurate on lab-style photos but only ~12% on
field photos while staying >90% confident, so field photos must be routed elsewhere.

Each class is sampled from that dataset's own existing train/val/test split, so no
test image (or near-duplicate) leaks into training.

Usage:
    python training/image_router/train.py
Output: models/image_router.pt ({model_state_dict, class_names, val_acc, test_acc, epoch})
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
EXTS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["lab_leaf", "field_leaf", "fruit", "other"]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Must match EVAL_TRANSFORM in the backend services
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def images_in(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def deepweeds_split(root, split):
    with open(root / "labels" / f"{split}_subset0.csv", newline="") as f:
        return [root / "images" / row["Filename"] for row in csv.DictReader(f)]


def build_split_lists(data, per_class):
    """{split: [(path, label_idx)]} with each class capped at per_class[split]."""
    pv, pd = data / "processed" / "plantvillage", data / "processed" / "plantdoc"
    banana, pests = data / "raw" / "banana_ripeness", data / "processed" / "agricultural_pests"
    deepweeds = data / "raw" / "deepweeds"

    rng = random.Random(42)
    pv_train = images_in(pv / "train")
    rng.shuffle(pv_train)
    n_val = per_class["val"]
    sources = {
        "train": {
            "lab_leaf": pv_train[n_val:],
            "field_leaf": images_in(pd / "train"),
            "fruit": images_in(banana / "train"),
            "other": images_in(pests / "train") + deepweeds_split(deepweeds, "train"),
        },
        "val": {
            "lab_leaf": pv_train[:n_val],  # PlantVillage has no val split on disk
            "field_leaf": images_in(pd / "val"),
            "fruit": images_in(banana / "valid"),
            "other": images_in(pests / "val") + deepweeds_split(deepweeds, "val"),
        },
        "test": {
            "lab_leaf": images_in(pv / "test"),
            "field_leaf": images_in(pd / "test"),
            "fruit": images_in(banana / "test"),
            "other": images_in(pests / "test") + deepweeds_split(deepweeds, "test"),
        },
    }
    out = {}
    for split, by_class in sources.items():
        items = []
        for label, paths in by_class.items():
            paths = list(paths)
            rng.shuffle(paths)
            items += [(p, CLASS_NAMES.index(label)) for p in paths[:per_class[split]]]
        out[split] = items
    return out


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--eval-per-class", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_class = {"train": args.train_per_class, "val": args.eval_per_class, "test": args.eval_per_class}
    splits = build_split_lists(args.data_dir, per_class)
    loaders = {
        s: DataLoader(ListDataset(items, TRAIN_TRANSFORM if s == "train" else EVAL_TRANSFORM),
                      batch_size=args.batch_size, shuffle=(s == "train"), num_workers=args.workers,
                      pin_memory=True, persistent_workers=args.workers > 0)
        for s, items in splits.items()
    }
    for s, items in splits.items():
        counts = [sum(1 for _, y in items if y == i) for i in range(len(CLASS_NAMES))]
        print(f"{s}: {dict(zip(CLASS_NAMES, counts))}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Class weights: field_leaf has only ~2k train images
    counts = torch.tensor([sum(1 for _, y in splits["train"] if y == i) for i in range(len(CLASS_NAMES))],
                          dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(DEVICE) / len(CLASS_NAMES))
    ckpt_path = args.output_dir / "image_router.pt"
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate(model, loaders["val"])
        print(f"Epoch {epoch}/{args.epochs} val_acc={val_acc:.4f} ({time.time() - start:.0f}s)", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                        "val_acc": val_acc, "epoch": epoch}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt["test_acc"] = evaluate(model, loaders["test"])
    torch.save(ckpt, ckpt_path)
    print(f"Best val_acc={best_val:.4f} test_acc={ckpt['test_acc']:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
