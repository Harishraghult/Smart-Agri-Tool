"""
predict_visualize.py

Single-image inference on a trained checkpoint (model_stage1_plantvillage.pt
or model_stage2_plantdoc.pt from train.py), with a Grad-CAM based CONTOUR
visualization showing *where* on the leaf the model is basing its decision --
this is what actually reads as "pest localization" even though the
classifiers in this pipeline are trained for whole-image classification,
not detection/segmentation. See the note at the bottom of this file for why
that distinction matters and what to say about it in your report.

WHAT THIS PRODUCES
-------------------
A single figure with 3 panels, saved as a PNG:
  1. Original image
  2. Grad-CAM heatmap (filled contour, seaborn colormap) overlaid on the image
  3. Same heatmap as pure line-contour (matplotlib contour) over the image,
     which is the "contoured output" look most people mean by that phrase

Plus the predicted class, confidence, and top-3 alternatives printed to
stdout and annotated on the figure.

USAGE
-----
    python predict_visualize.py \\
        --checkpoint checkpoints/model_stage2_plantdoc.pt \\
        --image path/to/leaf.jpg

    python predict_visualize.py \\
        --checkpoint checkpoints/model_stage2_plantdoc.pt \\
        --image path/to/leaf.jpg \\
        --output-dir predictions --topk 5

Needs: torch, torchvision, matplotlib, seaborn, opencv-python (cv2), Pillow
(pip install matplotlib seaborn opencv-python)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ─── Model loading (mirrors evaluate.py so checkpoints are interchangeable) ─

def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()

    return model, class_names


# ─── Grad-CAM ───────────────────────────────────────────────────────────
# Hooks the last conv block of EfficientNet-B0 (model.features[-1]).
# This is the standard "where is the network looking" localization method
# for a classifier that was never given bounding boxes or masks to train on.

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> one weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, output


def predict_and_explain(model, image_path: Path, class_names, topk=3):
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = INFER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    cam_engine = GradCAM(model, target_layer=model.features[-1])

    # First forward pass (no grad) just to get the predicted class
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())

    # Second forward+backward pass (grad needed) to build the CAM for
    # the predicted class specifically
    cam, _ = cam_engine(input_tensor, pred_idx)

    top_probs, top_idxs = probs.topk(min(topk, len(class_names)))
    top_results = [
        (class_names[i], float(p)) for p, i in zip(top_probs.tolist(), top_idxs.tolist())
    ]

    # Resize CAM up to the displayed image resolution
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    return display_img, cam_resized, top_results


# ─── Visualization ──────────────────────────────────────────────────────

def render_figure(display_img, cam, top_results, save_path: Path, image_name: str):
    sns.set_theme(style="white")
    pred_label, pred_conf = top_results[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(display_img)
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(display_img)
    filled = axes[1].contourf(cam, levels=12, cmap="inferno", alpha=0.55)
    fig.colorbar(filled, ax=axes[1], fraction=0.046, pad=0.04, label="model attention")
    axes[1].set_title("Grad-CAM (filled contour)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(display_img)
    contour_levels = np.linspace(cam.max() * 0.3, cam.max() * 0.95, 6) if cam.max() > 0 else 6
    lines = axes[2].contour(cam, levels=contour_levels, cmap="cool", linewidths=1.8)
    axes[2].clabel(lines, inline=True, fontsize=7, fmt="%.2f")
    axes[2].set_title("Attention contours (line)", fontsize=11)
    axes[2].axis("off")

    top_str = "  |  ".join(f"{name}: {conf:.1%}" for name, conf in top_results)
    fig.suptitle(
        f"{image_name}\nPrediction: {pred_label}  ({pred_conf:.1%} confidence)",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.01, f"Top-{len(top_results)}: {top_str}", ha="center", fontsize=9, color="dimgray")

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="e.g. checkpoints/model_stage2_plantdoc.pt")
    parser.add_argument("--image", type=Path, required=True,
                         help="Path to a single leaf image to diagnose")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = load_model(args.checkpoint)
    print(f"Loaded checkpoint with {len(class_names)} classes on {DEVICE}")

    display_img, cam, top_results = predict_and_explain(
        model, args.image, class_names, topk=args.topk
    )

    print(f"\nImage: {args.image.name}")
    print("Top predictions:")
    for name, conf in top_results:
        print(f"  {name:40s}  {conf:.2%}")

    save_path = args.output_dir / f"{args.image.stem}_gradcam.png"
    render_figure(display_img, cam, top_results, save_path, args.image.name)
    print(f"\nSaved visualization -> {save_path}")


if __name__ == "__main__":
    main()


# ─── Note for your report ──────────────────────────────────────────────
#
# Grad-CAM localizes based on gradients of the CLASSIFICATION head, not a
# trained detection/segmentation objective. That means the "contour" you
# get is "the region of the leaf that most influenced this label", which
# is usually where the lesion/pest damage is (since that's the
# discriminative region for the model) but is not a guaranteed, pixel-
# accurate pest boundary the way a trained segmentation mask would be.
# If you need an actual bounding box / mask around the pest itself rather
# than an attention proxy, that requires a detector (YOLO/Faster-RCNN) or
# segmentation model (U-Net/Mask R-CNN) trained on labeled pest
# bounding-boxes/masks -- which PlantVillage/PlantDoc do not provide out
# of the box. Document this distinction explicitly; it's exactly the kind
# of nuance that separates a strong methodology section from a shaky one.