"""
evaluate.py

Runs a trained checkpoint against a held-out test/ folder and reports
overall accuracy plus per-class precision/recall/F1 -- the numbers you
actually want in your report, not just a single accuracy figure.

USAGE
-----
    python scripts/evaluate.py \\
        --checkpoint checkpoints/model_stage1_plantvillage.pt \\
        --test-dir data/processed/plantvillage/test

    python scripts/evaluate.py \\
        --checkpoint checkpoints/model_stage2_plantdoc.pt \\
        --test-dir data/processed/plantdoc/test

Run both. The PlantDoc test number is your real "does this work outside
the lab" result -- lead with that one when you present.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()

    return model, class_names


def evaluate(model, test_dir: Path, class_names, batch_size=32):
    # allow_empty=True: thin classes (e.g. PlantDoc's 2-image spider-mite
    # class) can have zero samples in test after the split -- treat that as
    # "can't evaluate this class" rather than a hard crash.
    test_ds = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORM, allow_empty=True)

    # sanity check: the test folder's classes must match the checkpoint's
    if test_ds.classes != class_names:
        print(
            "[warn] test folder class list doesn't exactly match the "
            "checkpoint's class list -- results below map by index, "
            "double check test_ds.classes vs class_names if this looks off."
        )

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(test_ds)} images ({len(loader)} batches)...")

    num_classes = len(class_names)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, t in zip(preds.tolist(), labels.tolist()):
                if p == t:
                    tp[t] += 1
                else:
                    fp[p] += 1
                    fn[t] += 1

    overall_acc = correct / total
    print(f"\nOverall test accuracy: {overall_acc:.4f}  ({correct}/{total})\n")

    print(f"{'class':40s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>8s}")
    for i, name in enumerate(class_names):
        support = tp[i] + fn[i]
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        flag = "  <- low support, treat cautiously" if support < 10 else ""
        print(f"{name:40s} {precision:10.3f} {recall:10.3f} {f1:10.3f} {support:8d}{flag}")

    return overall_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.test_dir.exists():
        raise SystemExit(f"Test directory not found: {args.test_dir}")

    model, class_names = load_model(args.checkpoint)
    print(f"Loaded checkpoint with {len(class_names)} classes on {DEVICE}")
    evaluate(model, args.test_dir, class_names, args.batch_size)


if __name__ == "__main__":
    main()