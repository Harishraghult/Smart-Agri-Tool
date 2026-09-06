"""
train.py

Two-stage transfer learning for the leaf disease/species classifier:

  Stage 1: train an EfficientNet-B0 head (then unfreeze + fine-tune) on
           data/processed/plantvillage/{train,val}  (clean, lab-condition images)
  Stage 2: continue training the SAME model on
           data/processed/plantdoc/{train,val}      (messy, field-condition images)

Why two stages instead of merging the folders: PlantVillage teaches the
model clean symptom patterns fast (large dataset, low label noise).
PlantDoc then adapts those patterns to real-world lighting/backgrounds/angles.
Merging them into one dataset from the start would let the much larger,
easier PlantVillage set dominate training and the model would still fail
on field photos -- the exact problem PlantDoc exists to catch.

NOTE: PlantVillage and PlantDoc use different class taxonomies (different
label sets), so this script trains two separate classification heads on
the same backbone -- it does NOT try to force them into one shared label
space. Stage 2 re-initializes only the final classification layer to match
PlantDoc's class count, then fine-tunes the whole network from the Stage 1
weights. This gives you two checkpoints:
  - model_stage1_plantvillage.pt  (evaluate on plantvillage/test)
  - model_stage2_plantdoc.pt      (evaluate on plantdoc/test -- your main
                                    "does it work in the field" number)

USAGE
-----
    python scripts/train.py \\
        --plantvillage-dir data/processed/plantvillage \\
        --plantdoc-dir data/processed/plantdoc \\
        --output-dir checkpoints \\
        --epochs-stage1 10 --epochs-stage2 15 \\
        --batch-size 32

Run with --epochs-stage1 1 --epochs-stage2 1 first just to confirm the
pipeline runs end to end on your machine before committing to a full run.

Needs: pip install torch torchvision
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224  # EfficientNet-B0's native input size

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_dataloaders(dataset_dir: Path, batch_size: int, num_workers: int = 2):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected {train_dir} and {val_dir} to exist. "
            f"Did you run dedupe_split.py for this dataset yet?"
        )

    # allow_empty=True: some very thin classes (e.g. PlantDoc's
    # Tomato_two_spotted_spider_mites_leaf with only 2 total images) can end
    # up with zero images in val/test after the leakage-safe cluster split.
    # Without this flag, ImageFolder raises on an empty class folder instead
    # of just treating it as "no samples in this split" -- which is the
    # correct behavior here, not a real error.
    train_ds = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM, allow_empty=True)
    val_ds = datasets.ImageFolder(val_dir, transform=EVAL_TRANSFORM, allow_empty=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def replace_classifier_head(model, num_classes: int):
    """Swap only the final layer to match a new dataset's class count,
    keeping every other learned weight (used for Stage 1 -> Stage 2)."""
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over `loader`. If optimizer is given, trains; otherwise
    evaluates (no gradient updates)."""
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
    return total_loss / total, correct / total


def train_stage(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    checkpoint_path,
    log_path,
    class_names,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    log_rows = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        print(
            f"  epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"    -> new best (val_acc={val_acc:.4f}), saved to {checkpoint_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )
        writer.writeheader()
        writer.writerows(log_rows)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantvillage-dir", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--plantdoc-dir", type=Path, default=Path("data/processed/plantdoc"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs-stage1", type=int, default=10)
    parser.add_argument("--epochs-stage2", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)  # lower: fine-tuning, not learning from scratch
    parser.add_argument("--skip-stage1", action="store_true", help="Skip straight to stage 2 using an existing stage-1 checkpoint")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    stage1_ckpt = args.output_dir / "model_stage1_plantvillage.pt"
    stage2_ckpt = args.output_dir / "model_stage2_plantdoc.pt"

    # ---------- Stage 1: PlantVillage ----------
    if not args.skip_stage1:
        print("=== Stage 1: training on PlantVillage ===")
        train_loader, val_loader, class_names = build_dataloaders(
            args.plantvillage_dir, args.batch_size
        )
        print(f"  {len(class_names)} classes, "
              f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images")

        model = build_model(num_classes=len(class_names), pretrained=True)
        best_acc = train_stage(
            model, train_loader, val_loader,
            epochs=args.epochs_stage1, lr=args.lr_stage1,
            checkpoint_path=stage1_ckpt,
            log_path=args.output_dir / "stage1_log.csv",
            class_names=class_names,
        )
        print(f"Stage 1 best val_acc: {best_acc:.4f}\n")
    else:
        if not stage1_ckpt.exists():
            raise SystemExit(f"--skip-stage1 given but {stage1_ckpt} does not exist yet.")
        print(f"Skipping stage 1, loading existing checkpoint {stage1_ckpt}\n")

    # ---------- Stage 2: fine-tune on PlantDoc ----------
    print("=== Stage 2: fine-tuning on PlantDoc ===")
    train_loader, val_loader, plantdoc_classes = build_dataloaders(
        args.plantdoc_dir, args.batch_size
    )
    print(f"  {len(plantdoc_classes)} classes, "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images")

    checkpoint = torch.load(stage1_ckpt, map_location=DEVICE)
    model = build_model(num_classes=len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = replace_classifier_head(model, num_classes=len(plantdoc_classes))

    best_acc = train_stage(
        model, train_loader, val_loader,
        epochs=args.epochs_stage2, lr=args.lr_stage2,
        checkpoint_path=stage2_ckpt,
        log_path=args.output_dir / "stage2_log.csv",
        class_names=plantdoc_classes,
    )
    print(f"Stage 2 best val_acc: {best_acc:.4f}")
    print(f"\nDone. Final field-ready model: {stage2_ckpt}")
    print("Next: run evaluate.py against the held-out test/ folders for both datasets.")


if __name__ == "__main__":
    main()