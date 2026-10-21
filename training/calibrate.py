"""
Temperature scaling: makes a model's confidence match how often it is actually right.

The pest, weed and disease classifiers were trained with label smoothing, which makes them
systematically under-confident (pest: answers given at 80-90% were right 97% of the time). The app's
"not confident" cut-off (60%) then hides correct answers, e.g. a clear beetle photo at 59.7%.

Fits one scalar T on the model's VALIDATION split (logits / T, minimising negative log-likelihood),
saves it in the checkpoint as "temperature", and reports on the TEST split, before and after:
accuracy (unchanged by design), expected calibration error, and how many correct answers fall
below the 60% cut-off.

Usage:
    python training/calibrate.py --task pest
    python training/calibrate.py --task weed
    python training/calibrate.py --task disease_type
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
THRESHOLD = 0.6


class Items(Dataset):
    def __init__(self, items, size):
        self.items = items
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), NORM])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def split_items(task, class_names):
    """{'val': [(path, idx)], 'test': [...]} built exactly as the training script built them."""
    idx = {c: i for i, c in enumerate(class_names)}
    if task in ("pest", "weed"):
        sys.path.insert(0, str(ROOT / "training" / "field_classifiers"))
        import train as fc  # noqa: E402

        args = argparse.Namespace(
            task=task, data_dir=ROOT / "data/processed/agricultural_pests", data_root=ROOT / "data",
            images_dir=ROOT / "data/raw/deepweeds/images", labels_dir=ROOT / "data/raw/deepweeds/labels",
            negative_sources=(["plantvillage", "plantdoc", "deepweeds_negative"] if task == "pest"
                              else ["plantvillage", "plantdoc"]))
        splits, names = fc.build_datasets(args)
        assert list(names) == list(class_names), "class order differs from the checkpoint"
        out = {}
        for s in ("val", "test"):
            items = []
            for ds in splits[s].datasets:
                if hasattr(ds, "samples") and isinstance(ds.samples[0][0], str) and not hasattr(ds, "images_dir"):
                    items += [(p, y) for p, y in ds.samples]  # ImageFolder
                elif hasattr(ds, "images_dir"):
                    items += [(ds.images_dir / n, y) for n, y in ds.samples]  # DeepWeeds CSV
                else:
                    items += [(p, ds.label) for p in ds.paths]  # negatives
            out[s] = items
        return out
    if task == "disease_type":
        rows = list(csv.DictReader(open(ROOT / "data/processed/disease_types/manifest.csv", encoding="utf-8")))
        return {s: [(r["path"], idx[r["type"]]) for r in rows if r["split"] == s] for s in ("val", "test")}
    raise ValueError(task)


def logits_for(model, items, size):
    loader = DataLoader(Items(items, size), batch_size=64, num_workers=3)
    outs, ys = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            outs.append(model(x.to(DEVICE)).float().cpu())
            ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def report(logits, y, T):
    p = (logits / T).softmax(1)
    conf, pred = p.max(1)
    correct = (pred == y).float()
    ece = 0.0
    for lo in torch.arange(0, 1, 0.1):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            ece += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    hidden = ((conf < THRESHOLD) & (correct == 1)).sum().item()
    return (f"acc {correct.mean():.3f} | calibration error {float(ece):.3f} | correct answers hidden by the "
            f"{THRESHOLD:.0%} cut-off: {hidden}/{int(correct.sum())} | wrong answers shown as confident: "
            f"{int(((conf >= THRESHOLD) & (correct == 0)).sum())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pest", "weed", "disease_type"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    ckpt_path = args.checkpoint or ROOT / "models" / {
        "pest": "pest_classifier.pt", "weed": "weed_classifier.pt",
        "disease_type": "disease_type_classifier.pt"}[args.task]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    names = ckpt["class_names"]
    size = ckpt.get("img_size", 224)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)

    items = split_items(args.task, names)
    val_logits, val_y = logits_for(model, items["val"], size)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(val_logits / log_t.exp(), val_y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.exp())
    test_logits, test_y = logits_for(model, items["test"], size)
    print(f"{args.task}: fitted temperature T={T:.3f} on {len(val_y)} val photos; test n={len(test_y)}")
    print(f"  before: {report(test_logits, test_y, 1.0)}")
    print(f"  after : {report(test_logits, test_y, T)}")
    ckpt["temperature"] = T
    torch.save(ckpt, ckpt_path)
    print(f"  saved temperature into {ckpt_path}")


if __name__ == "__main__":
    main()
