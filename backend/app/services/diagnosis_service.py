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
from app.services.species_service import species_service
from app.services.disease_type_service import disease_type_service, METHODS as DISEASE_TYPE_METHODS

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
    "Potato_leaf": "Potato___healthy",  # added by the v2 field model (train_field.py)
    # added with the PlantWild data (scripts/prepare_plantwild.py)
    "Apple_black_rot": "Apple___Black_rot", "Cherry_powdery_mildew": "Cherry_(including_sour)___Powdery_mildew",
    "Citrus_greening": "Orange___Haunglongbing_(Citrus_greening)", "Corn_leaf": "Corn_(maize)___healthy",
    "Strawberry_leaf_scorch": "Strawberry___Leaf_scorch",
}

# Crops the disease models know, as shown to users -> PlantVillage class-name prefix
SUPPORTED_CROPS = {
    "Apple": "Apple", "Blueberry": "Blueberry", "Cherry": "Cherry_(including_sour)",
    "Corn (Maize)": "Corn_(maize)", "Grape": "Grape", "Orange": "Orange", "Peach": "Peach",
    "Bell Pepper": "Pepper,_bell", "Potato": "Potato", "Raspberry": "Raspberry", "Soybean": "Soybean",
    "Squash": "Squash", "Strawberry": "Strawberry", "Tomato": "Tomato",
}
_CROP_ALIASES = {"corn": "Corn (Maize)", "maize": "Corn (Maize)", "pepper": "Bell Pepper",
                 "capsicum": "Bell Pepper", "soyabean": "Soybean"}


# Which disease model to use. Accuracies are measured on held-out test photos (see README).
MODEL_CHOICES = {
    "auto": "Auto: the image router sends lab-style photos to the Lab model and farm photos to the Farm model",
    "lab": "Lab model (PlantVillage): single leaf on a plain background",
    "farm": "Farm model (PlantDoc): real farm photos",
    "farm_bioclip": "Farm model + BioCLIP-2: BioCLIP-2 identifies the plant species first",
}


def available_model_choices() -> dict:
    """Choices that actually work on this server (farm_bioclip needs the BioCLIP-2 species identifier)."""
    choices = dict(MODEL_CHOICES)
    if species_service.model is None:
        choices.pop("farm_bioclip")
    else:
        choices["auto"] += " (with BioCLIP-2 crop identification)"
    return choices


# Human-readable names of the models behind each answer (shown in the UI)
ENGINE_NAMES = {
    "efficientnet_b0_plantvillage": "Lab model (EfficientNet-B0, PlantVillage)",
    "efficientnet_b0_plantdoc": "Farm model (EfficientNet-B0, PlantDoc + PlantWild)",
    "demo": "no trained model",
}


class InvalidModelChoiceError(ValueError):
    """Unknown disease-model choice; message is safe to show."""


class UnsupportedCropError(ValueError):
    """The user's crop is not covered by the disease models; message is safe to show."""


def resolve_crop(crop_type):
    """User crop name -> (display name, PlantVillage prefix); (None, None) for auto-detect."""
    if not crop_type or crop_type.strip().lower() in ("", "auto", "auto-detect"):
        return None, None
    key = crop_type.strip().lower()
    for display, prefix in SUPPORTED_CROPS.items():
        if key in (display.lower(), prefix.lower()):
            return display, prefix
    if key in _CROP_ALIASES:
        display = _CROP_ALIASES[key]
        return display, SUPPORTED_CROPS[display]
    raise UnsupportedCropError(
        f"'{crop_type.strip()}' is not supported yet, so a diagnosis would be a guess. "
        f"Supported crops: {', '.join(SUPPORTED_CROPS)}.")


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
        self.field_transform = EVAL_TRANSFORM
        self.field_tta = False
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
            "species_identifier": species_service.model is not None,
            "disease_type_classifier": disease_type_service.model is not None,
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
                size = ckpt.get("img_size", IMG_SIZE)
                self.field_transform = transforms.Compose([
                    transforms.Resize((size, size)), transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                # Flip test-time augmentation only if it measurably helped on the test split
                self.field_tta = ckpt.get("test_acc_tta", 0) > ckpt.get("test_acc", 1)
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

    def _severity(self, image, img_tensor, healthy):
        """Lesion area %: U-Net if trained, else HSV colour segmentation of brown/chlorotic pixels."""
        import cv2
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
            if healthy:
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
        return severity_score

    def _class_probs(self, model, transform, image, tta):
        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            if tta:
                probs = (probs + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)[0]) / 2
        return probs

    @staticmethod
    def _restrict_to_crop(probs, pv_names, crop_prefix):
        """P(class | crop): zero out other crops and renormalise. None if the model lacks the crop."""
        mask = torch.tensor([n.split("___")[0] == crop_prefix for n in pv_names], device=probs.device)
        if not mask.any():
            return None
        masked = probs * mask
        return masked / masked.sum().clamp_min(1e-12)

    def predict(self, image_bytes: bytes, crop_type: str = None, model_choice: str = "auto",
                disease_type_method: str = None) -> dict:
        model_choice = (model_choice or "auto").strip().lower()
        disease_type_method = (disease_type_method or disease_type_service.default_method()).strip().lower()
        if disease_type_method not in disease_type_service.available_methods():
            raise InvalidModelChoiceError(
                f"Unknown or unavailable disease-type method '{disease_type_method}'. "
                f"Choose one of: {', '.join(disease_type_service.available_methods())}.")
        if model_choice not in MODEL_CHOICES:
            raise InvalidModelChoiceError(f"Unknown model '{model_choice}'. Choose one of: {', '.join(MODEL_CHOICES)}.")
        if model_choice in ("farm", "farm_bioclip") and self.field_model is None:
            raise InvalidModelChoiceError("The Farm model is not available on this server.")
        if model_choice == "farm_bioclip" and species_service.model is None:
            raise InvalidModelChoiceError("BioCLIP-2 is turned off on this server (SPECIES_ID_ENABLED=false); "
                                          "choose the Farm model instead.")
        # Unsupported / unlisted crop: answer with the disease type only (needs the disease-type model)
        unsupported_crop = None
        try:
            crop_display, crop_prefix = resolve_crop(crop_type)
        except UnsupportedCropError:
            if disease_type_service.model is None:
                raise
            crop_display, crop_prefix = None, None
            unsupported_crop = "Unlisted crop" if crop_type.strip().lower() in ("other", "unlisted") else crop_type.strip()
        image = load_rgb_image(image_bytes)
        img_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        top3_predictions = []

        # 0. What kind of image is this? Field photos go to the PlantDoc model, because the
        #    PlantVillage model is ~12% accurate (yet >90% confident) on them.
        image_type = image_router_service.classify(image)
        input_warning = None
        if model_choice == "auto":
            use_field_model = self.field_model is not None and image_type is not None and image_type["type"] != "lab_leaf"
        else:
            use_field_model = model_choice in ("farm", "farm_bioclip")
        if image_type and image_type["uncertain"]:
            input_warning = UNCLEAR_IMAGE_WARNING
        elif image_type and image_type["type"] in ("fruit", "other"):
            input_warning = (f"This doesn't look like a leaf photo (it looks like {image_type['description']}). "
                             "The diagnosis below is unreliable; upload a close-up of a single leaf.")

        # 0b. Auto-detect: identify the plant species. A confident unsupported plant (pear, betel, ...)
        #     is reported instead of being forced into a wrong disease class; a confident supported
        #     crop restricts the diagnosis to it (farm photos 75.2% -> 79.4%).
        crop_detected = None
        # Field photos only: on lab-style photos BioCLIP-2 named the crop right just 71% of the time,
        # while the PlantVillage model gets it ~99% right. Also skipped for exposure-corrected photos,
        # whose altered colours made it misidentify a brightened tomato leaf as strawberry.
        colour_altered = any(step.endswith("_correction") for step in image.info.get("preprocessing", []))
        is_field_photo = image_type is not None and image_type["type"] == "field_leaf"
        use_species = model_choice == "farm_bioclip" or (model_choice == "auto" and is_field_photo)
        if crop_prefix is None and input_warning is None and use_species and not colour_altered:
            crop_detected = species_service.identify(image)
            if crop_detected and crop_detected["confident"]:
                if not crop_detected["supported"] and disease_type_service.model is not None:
                    unsupported_crop = crop_detected["species"]
                elif not crop_detected["supported"]:
                    raise UnsupportedCropError(
                        f"This looks like {crop_detected['species'].lower()} ({crop_detected['scientific_name']}, "
                        f"{crop_detected['confidence']:.0%} sure), which isn't supported yet, so a diagnosis would "
                        f"be a guess. Supported crops: {', '.join(SUPPORTED_CROPS)}. If it is actually one of "
                        "these, select it in the Crop list.")
                else:
                    crop_prefix = SUPPORTED_CROPS[crop_detected["species"]]

        if unsupported_crop is not None:
            return self._disease_type_only(image, img_tensor, unsupported_crop, image_type, input_warning,
                                           model_choice, crop_detected)

        # 1. Classification, restricted to the user's crop when given (field photos: 64.8% -> 80.8%)
        candidates = []
        if use_field_model:
            model, names = self.field_model
            candidates.append(("efficientnet_b0_plantdoc", model, self.field_transform,
                               [PLANTDOC_TO_PLANTVILLAGE.get(n, n) for n in names], self.field_tta))
        if self.disease_model is not None:
            candidates.append(("efficientnet_b0_plantvillage", self.disease_model, EVAL_TRANSFORM,
                               list(self.class_names), False))

        probs = None
        for engine, model, transform, pv_names, tta in candidates:
            probs = self._class_probs(model, transform, image, tta)
            if crop_prefix:
                probs = self._restrict_to_crop(probs, pv_names, crop_prefix)
                if probs is None:  # e.g. Orange: only the PlantVillage model knows it
                    continue
            classifier_engine = engine
            break

        if probs is not None:
            conf, pred_idx = torch.max(probs, dim=0)
            predicted_class = pv_names[pred_idx.item()]
            confidence = float(conf.item())
            topk_conf, topk_indices = torch.topk(probs, min(3, len(pv_names)))
            for k_conf, k_idx in zip(topk_conf.tolist(), topk_indices.tolist()):
                if k_conf <= 0:
                    continue
                top3_predictions.append({
                    "class": pv_names[k_idx],
                    "confidence": round(float(k_conf), 4),
                    "percentage": f"{k_conf * 100:.1f}%"
                })
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

        # 2. Lesion severity
        severity_score = self._severity(image, img_tensor, healthy="healthy" in condition.lower())

        # 3. Crop-independent disease type (A: aggregate this model's scores, B: dedicated model)
        if disease_type_method == "model":
            disease_type = disease_type_service.classify(image)
        elif disease_type_method == "combined" and probs is not None:
            disease_type = disease_type_service.combine(image, probs, pv_names, ENGINE_NAMES[classifier_engine])
        elif probs is not None:
            disease_type = disease_type_service.aggregate(probs, pv_names, ENGINE_NAMES[classifier_engine])
        else:
            disease_type = None

        # 4. Knowledge Base Lookup
        kb_info = self.cause_lookup.get(predicted_class, {})
        cause_category = kb_info.get("cause_category", "pathogen" if "healthy" not in condition.lower() else "none")
        pathogen_type = kb_info.get("pathogen_type", "fungus" if cause_category == "pathogen" else "none")
        pathogen_name = kb_info.get("pathogen_name")
        # No curated entry: say so rather than borrowing another disease's symptoms and treatments
        symptoms = kb_info.get("symptoms", [])
        remedies = kb_info.get("remedies", [
            "No curated treatment for this diagnosis yet",
            "Confirm with your local agricultural extension officer before applying any chemicals",
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
            "crop_selected": crop_display,
            "model_choice": model_choice,
            "crop_detected": crop_detected,
            "input_warning": input_warning,
            "disease_type": disease_type,
            "disease_type_only": False,
            "unsupported_crop": None,
            "models_used": {
                "crop_specific_diagnosis": ENGINE_NAMES[classifier_engine],
                "disease_type": (disease_type or {}).get("model"),
            },
            "inference_engine": {
                "classifier": classifier_engine,
                "severity": "unet" if self.unet_model is not None else "hsv_heuristic",
            }
        }

    def _disease_type_only(self, image, img_tensor, crop_name, image_type, input_warning, model_choice,
                           crop_detected) -> dict:
        """Crop the crop-specific models don't know: report the disease type from model B only."""
        dt = disease_type_service.classify(image)
        info = disease_type_service.info(dt["type"])
        healthy = dt["type"] == "healthy"
        return {
            "predicted_class": None,
            "crop_type": crop_name,
            "condition": dt["display_name"],
            "confidence": dt["confidence"],
            "top_candidates": [],
            "cause_category": info.get("cause_category", "pathogen"),
            "pathogen_type": info.get("pathogen_type"),
            "pathogen_name": None,
            "severity_score": self._severity(image, img_tensor, healthy=healthy),
            "symptoms": info.get("symptoms", []),
            "remedies": info.get("management", []),
            "is_healthy": healthy,
            "low_confidence": dt["low_confidence"] or input_warning is not None,
            "image_type": image_type,
            "preprocessing": image.info.get("preprocessing", []),
            "crop_selected": None,
            "model_choice": model_choice,
            "crop_detected": crop_detected,
            "input_warning": input_warning,
            "disease_type": dt,
            "disease_type_only": True,
            "unsupported_crop": crop_name,
            "models_used": {"crop_specific_diagnosis": None, "disease_type": dt["model"]},
            "inference_engine": {
                "classifier": "disease_type_model",
                "severity": "unet" if self.unet_model is not None else "hsv_heuristic",
            },
        }


diagnosis_service = DiagnosisService()
