import json
import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

from app.config import settings
from app.services.image_utils import load_rgb_image
from app.services.image_router_service import image_router_service, UNCLEAR_IMAGE_WARNING

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

# PlantDoc (field photo) class names -> equivalent PlantVillage class, so the knowledge base,
# healthy detection and UI labels are identical whichever model answered.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple_Scab_Leaf": "Apple___Apple_scab", "Apple_leaf": "Apple___healthy",
    "Apple_rust_leaf": "Apple___Cedar_apple_rust", "Bell_pepper_leaf": "Pepper,_bell___healthy",
    "Bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot", "Blueberry_leaf": "Blueberry___healthy",
    "Cherry_leaf": "Cherry_(including_sour)___healthy",
    "Corn_Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn_rust_leaf": "Corn_(maize)___Common_rust_",
    "Peach_leaf": "Peach___healthy", "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight", "Raspberry_leaf": "Raspberry___healthy",
    "Soyabean_leaf": "Soybean___healthy", "Squash_Powdery_mildew_leaf": "Squash___Powdery_mildew",
    "Strawberry_leaf": "Strawberry___healthy", "Tomato_Early_blight_leaf": "Tomato___Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot", "Tomato_leaf": "Tomato___healthy",
    "Tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot", "Tomato_leaf_late_blight": "Tomato___Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": "Tomato___Leaf_Mold",
    "Tomato_two_spotted_spider_mites_leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape_leaf": "Grape___healthy", "grape_leaf_black_rot": "Grape___Black_rot",
}


def _load_efficientnet(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval(), ckpt


class DiagnosisService:
    def __init__(self):
        self.disease_model = None
        self.field_model = None  # (model, class_names) PlantDoc fine-tune for field photos
        self.unet_model = None
        self.class_names = PLANT_VILLAGE_CLASSES
        self.cause_lookup = {}
        self.metrics = {}
        self._load_cause_lookup()
        self._load_models()

    def model_status(self) -> dict:
        return {
            "disease_classifier": self.disease_model is not None,
            "disease_field_classifier": self.field_model is not None,
            "image_router": image_router_service.model is not None,
            "unet_segmentation": self.unet_model is not None,
            "cause_category_lookup": bool(self.cause_lookup),
        }

    def _load_cause_lookup(self):
        if settings.CAUSE_LOOKUP_PATH.exists():
            with open(settings.CAUSE_LOOKUP_PATH, "r") as f:
                self.cause_lookup = json.load(f)

    def _load_models(self):
        # PlantVillage classifier (lab-style photos)
        for path in [settings.DISEASE_MODEL_PATH, settings.BASE_DIR / "checkpoints" / "model_stage1_plantvillage.pt"]:
            if path.exists():
                try:
                    self.disease_model, ckpt = _load_efficientnet(path)
                    self.class_names = ckpt.get("class_names", PLANT_VILLAGE_CLASSES)
                    self.metrics = {"val_acc": ckpt.get("val_acc"), "checkpoint": path.name}
                    print(f"Loaded disease classification model from {path.name} with {len(self.class_names)} classes.")
                    break
                except Exception as e:
                    print(f"Error loading disease model from {path}: {e}")

        # PlantDoc fine-tune (field photos)
        if settings.DISEASE_FIELD_MODEL_PATH.exists():
            try:
                model, ckpt = _load_efficientnet(settings.DISEASE_FIELD_MODEL_PATH)
                self.field_model = (model, ckpt["class_names"])
                print(f"Loaded field disease model from {settings.DISEASE_FIELD_MODEL_PATH.name}")
            except Exception as e:
                print(f"Error loading field disease model: {e}")

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
        image = load_rgb_image(image_bytes)
        img_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        top3_predictions = []

        # 0. What kind of image is this? Field photos go to the PlantDoc model, because the
        #    PlantVillage model is ~12% accurate (yet >90% confident) on them.
        image_type = image_router_service.classify(image)
        input_warning = None
        use_field_model = self.field_model is not None and image_type is not None and image_type["type"] != "lab_leaf"
        if image_type and image_type["uncertain"]:
            input_warning = UNCLEAR_IMAGE_WARNING
        elif image_type and image_type["type"] in ("fruit", "other"):
            input_warning = (f"This doesn't look like a leaf photo (it looks like {image_type['description']}). "
                             "The diagnosis below is unreliable; upload a close-up of a single leaf.")

        # 1. Classification
        if use_field_model or self.disease_model is not None:
            model, names = self.field_model if use_field_model else (self.disease_model, self.class_names)
            to_pv = PLANTDOC_TO_PLANTVILLAGE if use_field_model else {}
            with torch.no_grad():
                probs = torch.softmax(model(img_tensor), dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            predicted_class = to_pv.get(names[pred_idx.item()], names[pred_idx.item()])
            confidence = float(conf.item())
            topk_conf, topk_indices = torch.topk(probs, min(3, len(names)))
            for k_conf, k_idx in zip(topk_conf.tolist(), topk_indices.tolist()):
                top3_predictions.append({
                    "class": to_pv.get(names[k_idx], names[k_idx]),
                    "confidence": round(float(k_conf), 4),
                    "percentage": f"{k_conf * 100:.1f}%"
                })
            classifier_engine = "efficientnet_b0_plantdoc" if use_field_model else "efficientnet_b0_plantvillage"
        else:
            # No trained weights: placeholder output, always flagged low-confidence below
            predicted_class = "Tomato___Early_blight"
            confidence = 0.0
            top3_predictions = [{"class": predicted_class, "confidence": 0.0, "percentage": "0.0%"}]
            classifier_engine = "demo"

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
            "low_confidence": confidence < settings.CONFIDENCE_THRESHOLD or input_warning is not None,
            "image_type": image_type,
            "preprocessing": image.info.get("preprocessing", []),
            "input_warning": input_warning,
            "inference_engine": {
                "classifier": classifier_engine,
                "severity": "unet" if self.unet_model is not None else "hsv_heuristic",
            }
        }


diagnosis_service = DiagnosisService()
