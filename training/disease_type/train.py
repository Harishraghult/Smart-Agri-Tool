"""
Crop-independent disease-type classifier (EfficientNet-B0): answers "early blight", "rust",
"powdery mildew", ... whatever the crop, including crops the crop-specific models do not support.

Data: data/processed/disease_types/manifest.csv from scripts/prepare_disease_types.py
(PlantVillage + PlantDoc + PlantDoc leaf crops + PlantWild v1/v2; leakage-checked splits).
Types and their farmer-facing info: backend/app/knowledge_base/disease_types.json.

Recipe: init from the PlantWild-trained field disease model (disease features that already work on
farm photos), new head, 288 px, TrivialAugment + RandomErasing, label smoothing, class-balanced
sampling (types range from ~60 to ~13k training images), AdamW + one-cycle schedule.

Usage:
    python training/disease_type/train.py
Output: models/disease_type_classifier.pt
        ({model_state_dict, class_names, img_size, val_acc, test_acc, test_acc_by_source, epoch})
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ManifestDataset(Dataset):
    def __init__(self, rows, class_idx, transform):
        self.rows, self.class_idx, self.transform = rows, class_idx, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(Image.open(r["path"]).convert("RGB")), self.class_idx[r["type"]]


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
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/disease_types/manifest.csv"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("models/disease_classifier_field.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    class_names = sorted({r["type"] for r in rows})
    idx = {c: i for i, c in enumerate(class_names)}
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    counts = Counter(r["type"] for r in by_split["train"])
    print(f"{len(class_names)} types | train {len(by_split['train'])} | val {len(by_split['val'])} | test {len(by_split['test'])}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.TrivialAugmentWide(), transforms.ToTensor(), NORM, transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), NORM])

    weights = [1.0 / counts[r["type"]] ** 0.5 for r in by_split["train"]]
    sampler = WeightedRandomSampler(weights, num_samples=args.samples_per_epoch, replacement=True)
    train_loader = DataLoader(ManifestDataset(by_split["train"], idx, train_tf), batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(ManifestDataset(by_split["val"], idx, eval_tf), batch_size=64, num_workers=args.workers)

    model = models.efficientnet_b0(weights=None)
    init = torch.load(args.init_checkpoint, map_location="cpu")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(init["class_names"]))
    model.load_state_dict(init["model_state_dict"])
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * len(train_loader), pct_start=0.06)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    ckpt_path = args.output_dir / "disease_type_classifier.pt"
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{args.epochs} val_acc={val_acc:.4f} ({time.time() - start:.0f}s)", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names,
                        "img_size": args.img_size, "val_acc": val_acc, "epoch": epoch}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    by_source = defaultdict(list)
    for r in by_split["test"]:
        by_source[r["source"]].append(r)
    ckpt["test_acc"] = evaluate(model, DataLoader(ManifestDataset(by_split["test"], idx, eval_tf), batch_size=64,
                                                  num_workers=args.workers))
    ckpt["test_acc_by_source"] = {
        s: evaluate(model, DataLoader(ManifestDataset(rs, idx, eval_tf), batch_size=64, num_workers=args.workers))
        for s, rs in sorted(by_source.items())
    }
    torch.save(ckpt, ckpt_path)
    print(f"Best val_acc={best_val:.4f} test_acc={ckpt['test_acc']:.4f} by source={ckpt['test_acc_by_source']} -> {ckpt_path}")


if __name__ == "__main__":
    main()
