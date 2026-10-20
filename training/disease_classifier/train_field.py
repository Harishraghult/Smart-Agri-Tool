"""
Field-photo disease classifier v2 (EfficientNet-B0), served for photos the image router
classifies as field_leaf.

Changes vs. the original stage-2 PlantDoc fine-tune (63% val, still improving when it stopped):
  * Data: PlantDoc whole photos + ~6.7k single-leaf crops from the PlantDoc detection boxes
    (scripts/prepare_plantdoc_crops.py; val/test duplicates already removed), plus a healthy-potato
    class ("Potato_leaf") the classification set lacks, topped up with PlantVillage Potato___healthy.
  * Init from the PlantVillage model (disease features), new 29-class head.
  * 288 px input, TrivialAugment + RandomErasing, label smoothing, class-balanced sampling
    (some classes have 2 training photos), AdamW + cosine schedule, 25 epochs.
  * Optional PlantWild in-the-wild photos mapped onto these classes (scripts/prepare_plantwild.py,
    CC-BY-NC-ND: non-commercial use), which also adds 5 classes. Its val split joins model selection;
    its test split is reported separately.
  * Validation / test are the untouched PlantDoc val/test whole photos (what users upload).

Usage:
    python training/disease_classifier/train_field.py
Output: <output-dir>/disease_classifier_field.pt
        ({model_state_dict, class_names, img_size, val_acc, test_acc, test_acc_tta,
          test_acc_plantwild, epoch}); test_acc is always the PlantDoc test set, for comparability
"""

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".jpg", ".jpeg", ".png"}
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def images_by_class(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) for d in root.iterdir() if d.is_dir()}


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def evaluate(model, loader, tta=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).softmax(1)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3])).softmax(1)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantdoc-dir", type=Path, default=Path("data/processed/plantdoc"))
    parser.add_argument("--crops-dir", type=Path, default=Path("data/processed/plantdoc_crops/train"))
    parser.add_argument("--plantvillage-dir", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--plantwild-dir", type=Path, default=Path("data/processed/plantwild_mapped"),
                        help="pass a non-existent path to train without PlantWild")
    parser.add_argument("--init-checkpoint", type=Path, default=Path("checkpoints/model_stage1_plantvillage.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        NORM,
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), NORM])

    whole = {s: images_by_class(args.plantdoc_dir / s) for s in ("train", "val", "test")}
    crops = images_by_class(args.crops_dir)
    wild = {s: images_by_class(args.plantwild_dir / s) for s in ("train", "val", "test")}
    class_names = sorted(set(whole["train"]) | set(crops) | set(wild["train"]))
    idx = {c: i for i, c in enumerate(class_names)}

    train_items = [(p, idx[c]) for c, ps in whole["train"].items() for p in ps]
    train_items += [(p, idx[c]) for c, ps in crops.items() for p in ps]
    train_items += [(p, idx[c]) for c, ps in wild["train"].items() for p in ps]
    if "Potato_leaf" in idx:  # top up the healthy-potato class from PlantVillage's train split
        pv_potato = images_by_class(args.plantvillage_dir / "train").get("Potato___healthy", [])
        train_items += [(p, idx["Potato_leaf"]) for p in pv_potato]
    val_items = [(p, idx[c]) for c, ps in whole["val"].items() if c in idx for p in ps]
    val_items += [(p, idx[c]) for c, ps in wild["val"].items() if c in idx for p in ps]
    test_items = [(p, idx[c]) for c, ps in whole["test"].items() if c in idx for p in ps]
    wild_test_items = [(p, idx[c]) for c, ps in wild["test"].items() if c in idx for p in ps]

    counts = Counter(y for _, y in train_items)
    print(f"{len(class_names)} classes | train {len(train_items)} | val {len(val_items)} | "
          f"test PlantDoc {len(test_items)} + PlantWild {len(wild_test_items)}")
    print("smallest train classes:", sorted(((class_names[k], v) for k, v in counts.items()), key=lambda t: t[1])[:5])

    # Inverse-sqrt frequency: rare classes are seen more often without fully oversampling 2-image classes
    weights = [1.0 / counts[y] ** 0.5 for _, y in train_items]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
    train_loader = DataLoader(ListDataset(train_items, train_tf), batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(ListDataset(val_items, eval_tf), batch_size=64, num_workers=args.workers)
    test_loader = DataLoader(ListDataset(test_items, eval_tf), batch_size=64, num_workers=args.workers)
    wild_test_loader = (DataLoader(ListDataset(wild_test_items, eval_tf), batch_size=64, num_workers=args.workers)
                        if wild_test_items else None)

    model = models.efficientnet_b0(weights=None)
    init = torch.load(args.init_checkpoint, map_location="cpu")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(init["class_names"]))
    model.load_state_dict(init["model_state_dict"])
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    ckpt_path = args.output_dir / "disease_classifier_field.pt"
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
    ckpt["test_acc"] = evaluate(model, test_loader)
    ckpt["test_acc_tta"] = evaluate(model, test_loader, tta=True)
    if wild_test_loader:
        ckpt["test_acc_plantwild"] = evaluate(model, wild_test_loader)
    torch.save(ckpt, ckpt_path)
    print(f"Best val_acc={best_val:.4f} test_acc(PlantDoc)={ckpt['test_acc']:.4f} "
          f"test_acc_tta={ckpt['test_acc_tta']:.4f} test_acc(PlantWild)={ckpt.get('test_acc_plantwild')} -> {ckpt_path}")


if __name__ == "__main__":
    main()
