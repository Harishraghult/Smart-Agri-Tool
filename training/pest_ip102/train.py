"""
Expanded pest classifier (EfficientNet-B0): 102 IP102 crop-pest species + the 12 Agricultural Pests
classes + "no_pest" (115 labels). The app reports the exact label and its pest group (sum of the
probabilities of all labels in that group), from backend/app/knowledge_base/pest_types.json.

Data: data/processed/pests_ip102/manifest.csv (scripts/prepare_pests_ip102.py).
Init from the current 13-class pest model (insect features), new head; 224 px, TrivialAugment,
label smoothing, inverse-sqrt class-balanced sampling, AdamW + one-cycle.

Usage:
    python training/pest_ip102/train.py
Output: <output-dir>/pest_classifier.pt
        ({model_state_dict, class_names, img_size, val_acc, test_acc, test_group_acc, test_by_source, epoch})
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ManifestDataset(Dataset):
    def __init__(self, rows, idx, transform):
        self.rows, self.idx, self.transform = rows, idx, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(Image.open(r["path"]).convert("RGB")), self.idx[r["label"]]


def predict_all(model, loader):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model(x.to(DEVICE)).softmax(1).float().cpu())
            ys.append(y)
    return torch.cat(probs), torch.cat(ys)


def scores(probs, ys, group_matrix):
    species = (probs.argmax(1) == ys).float().mean().item()
    group_probs = probs @ group_matrix  # [n, groups]
    group = (group_probs.argmax(1) == group_matrix[ys].argmax(1)).float().mean().item()
    return species, group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/pests_ip102/manifest.csv")
    parser.add_argument("--pest-types", type=Path, default=ROOT / "backend/app/knowledge_base/pest_types.json")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/pest_classifier.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--samples-per-epoch", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    pest_types = json.load(open(args.pest_types, encoding="utf-8"))
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    class_names = sorted({r["label"] for r in rows})
    idx = {c: i for i, c in enumerate(class_names)}
    group_names = sorted({pest_types["labels"][c]["group"] for c in class_names})
    group_matrix = torch.zeros(len(class_names), len(group_names))
    for c, i in idx.items():
        group_matrix[i, group_names.index(pest_types["labels"][c]["group"])] = 1.0

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    counts = Counter(r["label"] for r in by_split["train"])
    print(f"{len(class_names)} labels / {len(group_names)} groups | train {len(by_split['train'])} | "
          f"val {len(by_split['val'])} | test {len(by_split['test'])}", flush=True)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.4, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), transforms.ToTensor(), NORM, transforms.RandomErasing(p=0.2),
    ])
    eval_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), NORM])
    weights = [1.0 / counts[r["label"]] ** 0.5 for r in by_split["train"]]
    train_loader = DataLoader(ManifestDataset(by_split["train"], idx, train_tf), batch_size=args.batch_size,
                              sampler=WeightedRandomSampler(weights, args.samples_per_epoch, replacement=True),
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    eval_loader = lambda rs: DataLoader(ManifestDataset(rs, idx, eval_tf), batch_size=96, num_workers=args.workers)
    val_loader = eval_loader(by_split["val"])

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
    ckpt_path = args.output_dir / "pest_classifier.pt"
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
        val_species, val_group = scores(*predict_all(model, val_loader), group_matrix)
        print(f"Epoch {epoch}/{args.epochs} val species {val_species:.4f} group {val_group:.4f} "
              f"({time.time() - start:.0f}s)", flush=True)
        if val_species > best_val:
            best_val = val_species
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names,
                        "img_size": args.img_size, "val_acc": val_species, "val_group_acc": val_group,
                        "epoch": epoch}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    by_source = defaultdict(list)
    for r in by_split["test"]:
        by_source[r["source"]].append(r)
    ckpt["test_acc"], ckpt["test_group_acc"] = scores(*predict_all(model, eval_loader(by_split["test"])), group_matrix)
    ckpt["test_by_source"] = {s: scores(*predict_all(model, eval_loader(rs)), group_matrix)
                              for s, rs in sorted(by_source.items())}
    torch.save(ckpt, ckpt_path)
    print(f"Best val={best_val:.4f} | test species={ckpt['test_acc']:.4f} group={ckpt['test_group_acc']:.4f} "
          f"| by source (species, group)={ckpt['test_by_source']} -> {ckpt_path}")


if __name__ == "__main__":
    main()
