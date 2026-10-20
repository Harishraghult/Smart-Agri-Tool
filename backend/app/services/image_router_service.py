import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.config import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Below this the router cannot tell what the image shows (e.g. noise, abstract images)
ROUTER_MIN_CONFIDENCE = 0.6

IMAGE_TYPE_LABELS = {
    "lab_leaf": "single leaf on a plain background",
    "field_leaf": "leaf photographed in the field",
    "fruit": "fruit",
    "other": "not a plant leaf or fruit (insect, landscape, or other object)",
}


class ImageRouterService:
    """Classifies an upload as lab_leaf / field_leaf / fruit / other (training/image_router)."""

    def __init__(self):
        self.model = None
        self.class_names = list(IMAGE_TYPE_LABELS)
        self.metrics = {}
        self._load_model()

    def _load_model(self):
        path = settings.IMAGE_ROUTER_PATH
        if not path.exists():
            return
        try:
            ckpt = torch.load(path, map_location=DEVICE)
            self.class_names = ckpt["class_names"]
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_names))
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(DEVICE).eval()
            self.model = model
            self.metrics = {"val_acc": ckpt.get("val_acc"), "test_acc": ckpt.get("test_acc")}
            print(f"Loaded image router from {path}")
        except Exception as e:
            print(f"Image router load note: {e}")

    def classify(self, image: Image.Image):
        """Return {"type", "confidence", "description"}, or None when no router is trained."""
        if self.model is None:
            return None
        with torch.no_grad():
            probs = torch.softmax(self.model(EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        label = self.class_names[idx.item()]
        return {"type": label, "confidence": round(float(conf), 3), "description": IMAGE_TYPE_LABELS.get(label, label),
                "uncertain": float(conf) < ROUTER_MIN_CONFIDENCE}


UNCLEAR_IMAGE_WARNING = ("The image is unclear (the system can't tell what it shows). "
                         "Results below are unreliable; retake a sharp, well-lit photo.")


image_router_service = ImageRouterService()
