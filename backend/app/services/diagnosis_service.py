import json
import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

from app.config import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Default PlantVillage 38 class list
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


class DiagnosisService:
    def __init__(self):
        self.disease_model = None
        self.unet_model = None
        self.class_names = PLANT_VILLAGE_CLASSES
        self.cause_lookup = {}
        self.metrics = {}
        self._load_cause_lookup()
        self._load_models()

    def model_status(self) -> dict:
        return {
            "disease_classifier": self.disease_model is not None,
            "unet_segmentation": self.unet_model is not None,
            "cause_category_lookup": bool(self.cause_lookup),
        }

    def _load_cause_lookup(self):
        if settings.CAUSE_LOOKUP_PATH.exists():
            with open(settings.CAUSE_LOOKUP_PATH, "r") as f:
                self.cause_lookup = json.load(f)

    def _load_models(self):
        # Load Disease Classifier (Check primary path or checkpoints directory)
        candidate_paths = [
            settings.DISEASE_MODEL_PATH,
            settings.BASE_DIR / "checkpoints" / "model_stage1_plantvillage.pt",
            settings.BASE_DIR / "checkpoints" / "model_stage2_plantdoc.pt",
        ]
        
        for path in candidate_paths:
            if path.exists():
                try:
                    ckpt = torch.load(path, map_location=DEVICE)
                    self.class_names = ckpt.get("class_names", PLANT_VILLAGE_CLASSES)
                    model = models.efficientnet_b0(weights=None)
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_names))
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.to(DEVICE).eval()
                    self.disease_model = model
                    self.metrics = {"val_acc": ckpt.get("val_acc"), "checkpoint": path.name}
                    print(f"Loaded disease classification model from {path.name} with {len(self.class_names)} classes.")
                    break
                except Exception as e:
                    print(f"Error loading disease model from {path}: {e}")

        # Load U-Net Segmentation
        if settings.UNET_SEGMENTATION_PATH.exists():
            try:
                import sys
                if str(settings.BASE_DIR) not in sys.path:
                    sys.path.insert(0, str(settings.BASE_DIR))
                from training.disease_segmentation.train_unet import UNet
                ckpt = torch.load(settings.UNET_SEGMENTATION_PATH, map_location=DEVICE)
                model = UNet(num_classes=1, pretrained=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(DEVICE).eval()
                self.unet_model = model
            except Exception as e:
                print(f"Error loading U-Net model: {e}")

    def predict(self, image_bytes: bytes) -> dict:
        import cv2
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        top3_predictions = []

        # 1. Classification
        if self.disease_model is not None:
            with torch.no_grad():
                outputs = self.disease_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
                predicted_class = self.class_names[pred_idx.item()]
                confidence = float(conf.item())
                
                # Get top-3
                topk_conf, topk_indices = torch.topk(probs, min(3, len(self.class_names)))
                for k_conf, k_idx in zip(topk_conf.tolist(), topk_indices.tolist()):
                    top3_predictions.append({
                        "class": self.class_names[k_idx],
                        "confidence": round(float(k_conf), 4),
                        "percentage": f"{k_conf * 100:.1f}%"
                    })
        else:
            # Fallback/Demo prediction if weights not trained yet
            predicted_class = "Tomato___Early_blight"
            confidence = 0.942
            top3_predictions = [
                {"class": "Tomato___Early_blight", "confidence": 0.942, "percentage": "94.2%"},
                {"class": "Tomato___Septoria_leaf_spot", "confidence": 0.041, "percentage": "4.1%"},
                {"class": "Tomato___healthy", "confidence": 0.012, "percentage": "1.2%"}
            ]

        # Parse crop and condition
        parts = predicted_class.split("___")
        crop_type = parts[0].replace("_", " ").replace(",", "") if len(parts) > 0 else "Unknown Crop"
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown Condition"

        # 2. U-Net Severity Segmentation or Dynamic Vision Lesion Calculation
        severity_score = 0.0
        if self.unet_model is not None:
            with torch.no_grad():
                seg_out = self.unet_model(img_tensor)
                seg_prob = torch.sigmoid(seg_out)[0, 0]
                lesion_pixels = (seg_prob > 0.5).sum().item()
                total_pixels = seg_prob.numel()
                severity_score = round((lesion_pixels / total_pixels) * 100, 2)
        else:
            # Dynamic lesion surface area estimation from actual leaf pixels
            if "healthy" in condition.lower():
                severity_score = 0.0
            else:
                try:
                    img_np = np.array(image)
                    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                    
                    # Foreground leaf mask (excluding extreme background)
                    leaf_mask = (s > 25) & (v > 30) & (v < 240)
                    leaf_area = int(leaf_mask.sum())
                    
                    if leaf_area > 100:
                        # Brown necrosis, fungal lesions, yellow halo chlorosis
                        brown_lesions = (h < 25) & (s > 35) & (v < 185) & leaf_mask
                        yellow_chlorosis = (h >= 18) & (h <= 36) & (s > 65) & leaf_mask
                        lesion_pixels = int(brown_lesions.sum() + yellow_chlorosis.sum())
                        calc_pct = round((lesion_pixels / leaf_area) * 100, 1)
                        severity_score = max(2.5, min(78.0, calc_pct))
                    else:
                        severity_score = 19.5
                except Exception:
                    severity_score = 19.5

        # 3. Knowledge Base Lookup
        kb_info = self.cause_lookup.get(predicted_class, {})
        cause_category = kb_info.get("cause_category", "pathogen" if "healthy" not in condition.lower() else "none")
        pathogen_type = kb_info.get("pathogen_type", "fungus" if cause_category == "pathogen" else "none")
        pathogen_name = kb_info.get("pathogen_name", "Alternaria solani" if "Early_blight" in predicted_class else None)
        symptoms = kb_info.get("symptoms", ["Dark concentric spots on leaves", "Lower leaf yellowing"])
        remedies = kb_info.get("remedies", [
            "Apply copper-based fungicide preventatively",
            "Remove and destroy heavily infected lower leaves",
            "Ensure proper spacing for ventilation",
            "Avoid overhead irrigation to reduce leaf moisture"
        ])

        return {
            "predicted_class": predicted_class,
            "crop_type": crop_type,
            "condition": condition,
            "confidence": confidence,
            "top_candidates": top3_predictions,
            "cause_category": cause_category, # pathogen, pest, deficiency, none
            "pathogen_type": pathogen_type,
            "pathogen_name": pathogen_name,
            "severity_score": severity_score, # percentage
            "symptoms": symptoms,
            "remedies": remedies,
            "is_healthy": "healthy" in condition.lower(),
            "low_confidence": confidence < settings.CONFIDENCE_THRESHOLD,
            "inference_engine": {
                "classifier": "efficientnet_b0" if self.disease_model is not None else "demo",
                "severity": "unet" if self.unet_model is not None else "hsv_heuristic",
            }
        }


diagnosis_service = DiagnosisService()
