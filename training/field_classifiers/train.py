"""
Field image classifiers (EfficientNet-B0 transfer learning) for the Field Monitoring module.

Two tasks, served by backend/app/services/field_service.py:

  pest  - 12-class Agricultural Pests dataset (ants, bees, beetle, ... weevil).
          Uses the leakage-safe pHash split already produced on disk (train/val/test folders).
          A 13th "no_pest" class is sampled from PlantVillage leaf photos so that a plain leaf
          is not forced into one of the 12 pest species.
  weed  - DeepWeeds (8 weed species + "Negative"), using the official subset0 train/val/test CSVs.

Usage:
    python training/field_classifiers/train.py --task pest
    python training/field_classifiers/train.py --task weed

Output: models/<task>_classifier.pt  ({model_state_dict, class_names, val_acc, test_acc, epoch})
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

DEEPWEEDS_CLASSES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
    "Rubber vine", "Siam weed", "Snake weed", "Negative",
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Must match EVAL_TRANSFORM in the backend services
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CsvImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform):
        self.images_dir = Path(images_dir)
        self.transform = transform
        with open(csv_path, newline="") as f:
            self.samples = [(row["Filename"], int(row["Label"])) for row in csv.DictReader(f)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = Image.open(self.images_dir / name).convert("RGB")
        return self.transform(img), label


class ListImageDataset(Dataset):
    def __init__(self, paths, label, transform):
        self.paths, self.label, self.transform = paths, label, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), self.label


def sample_negatives(neg_dir, label, per_split, seed=42):
    """Leaf photos (no pest) from PlantVillage train/test folders, split train/val/test."""
    rng = random.Random(seed)
    exts = {".jpg", ".jpeg", ".png"}
    pool_train = sorted(p for p in (neg_dir / "train").rglob("*") if p.suffix.lower() in exts)
    pool_test = sorted(p for p in (neg_dir / "test").rglob("*") if p.suffix.lower() in exts)
    rng.shuffle(pool_train)
    rng.shuffle(pool_test)
    n_tr, n_va, n_te = per_split
    return {
        "train": ListImageDataset(pool_train[:n_tr], label, TRAIN_TRANSFORM),
        "val": ListImageDataset(pool_train[n_tr:n_tr + n_va], label, EVAL_TRANSFORM),
        "test": ListImageDataset(pool_test[:n_te], label, EVAL_TRANSFORM),
    }


def rgb_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def build_datasets(args):
    if args.task == "pest":
        splits = {
            s: datasets.ImageFolder(args.data_dir / s, TRAIN_TRANSFORM if s == "train" else EVAL_TRANSFORM,
                                    loader=rgb_loader)
            for s in ("train", "val", "test")
        }
        class_names = list(splits["train"].classes)
        if args.negatives_dir and args.negatives_dir.exists():
            per_split = tuple(len(splits[s]) // len(class_names) for s in ("train", "val", "test"))
            negs = sample_negatives(args.negatives_dir, len(class_names), per_split)
            splits = {s: ConcatDataset([splits[s], negs[s]]) for s in splits}
            class_names.append("no_pest")
        return splits, class_names
    splits = {
        s: CsvImageDataset(args.labels_dir / f"{s}_subset0.csv", args.images_dir,
                           TRAIN_TRANSFORM if s == "train" else EVAL_TRANSFORM)
        for s in ("train", "val", "test")
    }
    return splits, DEEPWEEDS_CLASSES


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
    parser.add_argument("--task", choices=["pest", "weed"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/agricultural_pests"))
    parser.add_argument("--negatives-dir", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/deepweeds/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/raw/deepweeds/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits, class_names = build_datasets(args)
    loaders = {
        s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                      num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        for s, ds in splits.items()
    }
    print(f"Task={args.task} classes={len(class_names)} "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} device={DEVICE}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt_path = args.output_dir / f"{args.task}_classifier.pt"
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        scheduler.step()
        val_acc = evaluate(model, loaders["val"])
        print(f"Epoch {epoch}/{args.epochs} loss={running / seen:.4f} val_acc={val_acc:.4f} "
              f"({time.time() - start:.0f}s)", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names,
                        "val_acc": val_acc, "epoch": epoch}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt["test_acc"] = evaluate(model, loaders["test"])
    torch.save(ckpt, ckpt_path)
    print(f"Best val_acc={best_val:.4f} test_acc={ckpt['test_acc']:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
