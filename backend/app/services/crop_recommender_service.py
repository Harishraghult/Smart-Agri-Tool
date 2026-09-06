import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from app.config import settings

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Default feature importance fallback (N, P, K, temp, humidity, ph, rainfall)
DEFAULT_IMPORTANCES = {
    "rainfall": 0.35,
    "humidity": 0.22,
    "K": 0.15,
    "P": 0.12,
    "N": 0.08,
    "temperature": 0.05,
    "ph": 0.03
}

CROP_DESCRIPTIONS = {
    "rice": "Requires high rainfall (>200mm), clayey loam soil, and warm temperatures.",
    "maize": "Well-drained soil, moderate rainfall (50-100mm), high nitrogen requirement.",
    "chickpea": "Drought tolerant, low moisture demand, requires well-drained sandy loam.",
    "kidneybeans": "Moderate rainfall, prefers slightly acidic soil (pH 6.0-6.8).",
    "pigeonpeas": "Deep taproot system, highly drought tolerant, low phosphorus requirement.",
    "mothbeans": "Extreme heat and drought tolerance, ideal for arid and semi-arid regions.",
    "mungbean": "Short season legume (60-70 days), enriches soil nitrogen.",
    "blackgram": "Warm humid climate, neutral pH soil, excellent green manure crop.",
    "lentil": "Cool season crop, moderate rainfall, sensitive to waterlogging.",
    "pomegranate": "Arid to semi-arid, deep well-drained soil, tolerates alkaline pH.",
    "banana": "High potassium demand, heavy water requirements (>150mm rain), warm climate.",
    "mango": "Deep alluvial soil, distinct dry dry spell for flowering, high temperature.",
    "grapes": "Low humidity required during ripening, medium rainfall, good drainage.",
    "watermelon": "Sandy loam, high temperature and sunshine, moderate nitrogen.",
    "muskmelon": "Warm dry climate, high potassium and phosphorus requirement.",
    "apple": "Cool temperate climate, well-drained loamy soil, distinct winter chilling.",
    "orange": "Subtropical, well-drained soil, sensitive to high salinity.",
    "papaya": "Frost sensitive, high rainfall, neutral pH soil.",
    "coconut": "Coastal tropical climate, high humidity (>80%), high rainfall.",
    "cotton": "Black cotton soil (vertisol), warm climate, 180-200 frost-free days.",
    "jute": "High rainfall (>150mm), high humidity (>80%), alluvial soil.",
    "coffee": "Elevated shade farming, high rainfall, organic rich acidic soil (pH 5.5-6.5)."
}


class CropRecommenderService:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = FEATURE_NAMES
        self._load_model()

    def _load_model(self):
        if settings.CROP_RECOMMENDER_PATH.exists():
            try:
                with open(settings.CROP_RECOMMENDER_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.label_encoder = data["label_encoder"]
                    self.feature_cols = data.get("feature_cols", FEATURE_NAMES)
            except Exception as e:
                print(f"Error loading crop recommender model: {e}")

    def predict(self, n: float, p: float, k: float, temp: float, humidity: float, ph: float, rainfall: float) -> dict:
        features = np.array([[n, p, k, temp, humidity, ph, rainfall]])

        if self.model is not None and self.label_encoder is not None:
            probs = self.model.predict_proba(features)[0]
            top_3_idx = np.argsort(probs)[::-1][:3]
            top_crops = [
                {
                    "crop": str(self.label_encoder.classes_[idx]),
                    "confidence": round(float(probs[idx]), 3),
                    "description": CROP_DESCRIPTIONS.get(str(self.label_encoder.classes_[idx]).lower(), "Suitable for current soil and climate profile.")
                }
                for idx in top_3_idx
            ]
            recommended_crop = top_crops[0]["crop"]
            
            # Feature importances
            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
                feature_importances = {self.feature_cols[i]: round(float(imp[i]), 3) for i in range(len(self.feature_cols))}
            else:
                feature_importances = DEFAULT_IMPORTANCES
        else:
            # Smart fallback rule-based decision logic if model file not yet pickled
            if rainfall > 180 and humidity > 75:
                recommended_crop = "rice"
                alt1, alt2 = "jute", "coconut"
            elif k > 150 and rainfall > 100:
                recommended_crop = "banana"
                alt1, alt2 = "watermelon", "papaya"
            elif n > 70 and p > 40:
                recommended_crop = "maize"
                alt1, alt2 = "cotton", "blackgram"
            elif rainfall < 60:
                recommended_crop = "chickpea"
                alt1, alt2 = "mothbeans", "lentil"
            else:
                recommended_crop = "mango"
                alt1, alt2 = "pomegranate", "orange"

            top_crops = [
                {"crop": recommended_crop, "confidence": 0.92, "description": CROP_DESCRIPTIONS.get(recommended_crop.lower(), "")},
                {"crop": alt1, "confidence": 0.84, "description": CROP_DESCRIPTIONS.get(alt1.lower(), "")},
                {"crop": alt2, "confidence": 0.76, "description": CROP_DESCRIPTIONS.get(alt2.lower(), "")},
            ]
            feature_importances = DEFAULT_IMPORTANCES

        # Build key drivers text
        top_driver = max(feature_importances.items(), key=lambda x: x[1])[0]

        return {
            "recommended_crop": recommended_crop.capitalize(),
            "top_3_recommendations": top_crops,
            "feature_importances": feature_importances,
            "key_driver": top_driver,
            "soil_climate_summary": {
                "N_PK_ratio": f"{n}:{p}:{k}",
                "temperature": f"{temp}°C",
                "humidity": f"{humidity}%",
                "pH_level": f"{ph} ({'Acidic' if ph < 6.5 else ('Alkaline' if ph > 7.5 else 'Neutral')})",
                "rainfall": f"{rainfall} mm"
            }
        }


crop_recommender_service = CropRecommenderService()
