"""
Crop-independent disease type ("early blight", "rust", "powdery mildew", ...), two ways:

  A. "aggregated": sum the crop-specific model's class probabilities per disease type
     (e.g. P(tomato early blight) + P(potato early blight) = P(early blight)). No extra model; only
     covers the 14 supported crops.
  B. "model": a dedicated disease-type classifier (training/disease_type/train.py) trained on
     PlantVillage + PlantDoc + PlantWild, which also covers crops the app does not support.

Types, farmer-facing info and the label -> type mapping: knowledge_base/disease_types.json.
"""
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.config import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Measured on held-out photos (README): B is the best default (84.8% overall, 81.5% on unsupported crops
# with Auto-detect); A+B is best on lab-style / PlantDoc-like photos of supported crops; A is weakest.
METHODS = {
    "model": "Disease-type model (B, recommended): EfficientNet-B0 trained directly on disease types across 30+ crops",
    "combined": "Combined (A+B): average of the disease-type model and the crop-specific model's scores",
    "aggregated": "Aggregated (A): crop-specific model's scores added up per disease type",
}


class DiseaseTypeService:
    def __init__(self):
        with open(settings.DISEASE_TYPES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.types = data["types"]
        self.label_to_type = data["label_to_type"]
        self.model = None
        self.class_names = []
        self.transform = None
        self.metrics = {}
        self._load_model()

    def _load_model(self):
        path = settings.DISEASE_TYPE_MODEL_PATH
        if not path.exists():
            return
        try:
            ckpt = torch.load(path, map_location=DEVICE)
            self.class_names = ckpt["class_names"]
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_names))
            model.load_state_dict(ckpt["model_state_dict"])
            self.model = model.to(DEVICE).eval()
            size = ckpt.get("img_size", 224)
            self.transform = transforms.Compose([
                transforms.Resize((size, size)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.metrics = {"val_acc": ckpt.get("val_acc"), "test_acc": ckpt.get("test_acc"),
                            "test_acc_by_source": ckpt.get("test_acc_by_source")}
            print(f"Loaded disease-type model from {path} ({len(self.class_names)} types)")
        except Exception as e:
            print(f"Disease-type model load note: {e}")

    def available_methods(self) -> dict:
        methods = dict(METHODS)
        if self.model is None:
            methods.pop("model")
            methods.pop("combined")
        return methods

    def default_method(self) -> str:
        return "model" if self.model is not None else "aggregated"

    def info(self, disease_type: str) -> dict:
        return self.types.get(disease_type, {})

    def _result(self, scores: dict, method: str, source: str) -> dict:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_type, top_conf = ranked[0]
        return {
            "type": top_type,
            "display_name": self.types[top_type]["display_name"],
            "confidence": round(top_conf, 4),
            "low_confidence": top_conf < settings.CONFIDENCE_THRESHOLD,
            "top_candidates": [
                {"type": t, "display_name": self.types[t]["display_name"], "confidence": round(c, 4),
                 "percentage": f"{c * 100:.1f}%"}
                for t, c in ranked[:3] if c > 0
            ],
            "method": method,
            "model": source,
        }

    def model_scores(self, image: Image.Image) -> dict:
        with torch.no_grad():
            probs = torch.softmax(self.model(self.transform(image).unsqueeze(0).to(DEVICE)), dim=1)[0]
        return {t: float(p) for t, p in zip(self.class_names, probs.tolist())}

    def aggregated_scores(self, probs: torch.Tensor, class_names: list) -> dict:
        scores = {}
        for name, p in zip(class_names, probs.tolist()):
            t = self.label_to_type.get(name)
            if t:
                scores[t] = scores.get(t, 0.0) + float(p)
        total = sum(scores.values()) or 1.0
        return {t: s / total for t, s in scores.items()}

    def classify(self, image: Image.Image) -> dict | None:
        """Method B. None if the disease-type model is not trained/loaded."""
        if self.model is None:
            return None
        return self._result(self.model_scores(image), "model", METHODS["model"].split(": ", 1)[1])

    def aggregate(self, probs: torch.Tensor, class_names: list, source_model: str) -> dict:
        """Method A: sum crop-specific class probabilities per disease type."""
        return self._result(self.aggregated_scores(probs, class_names), "aggregated",
                            f"aggregated from the {source_model}")

    def combine(self, image: Image.Image, probs: torch.Tensor, class_names: list, source_model: str) -> dict:
        """A+B: average both distributions (A only covers the supported crops' disease types)."""
        a = self.aggregated_scores(probs, class_names)
        b = self.model_scores(image)
        scores = {t: (a.get(t, 0.0) + b.get(t, 0.0)) / 2 for t in set(a) | set(b)}
        return self._result(scores, "combined",
                            f"disease-type model (EfficientNet-B0) averaged with the {source_model}")


disease_type_service = DiseaseTypeService()
