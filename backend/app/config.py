import os
from pathlib import Path

# Project root (contains models/, checkpoints/, training/). Override with PROJECT_ROOT in Docker.
BASE_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
APP_DIR = Path(__file__).resolve().parent

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        class BaseSettings:
            pass


def _first_existing(*candidates: Path) -> Path:
    """Return the first candidate that exists, else the first (canonical) path."""
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


MODELS = BASE_DIR / "models"


class Settings(BaseSettings):
    BASE_DIR: Path = BASE_DIR
    PROJECT_NAME: str = "Smart Crop Monitoring & Pest Intelligence System API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Model Artifact Paths (each falls back to the raw training-run output location)
    MODELS_DIR: Path = MODELS
    DISEASE_MODEL_PATH: Path = _first_existing(
        MODELS / "disease_classifier.pt",
        BASE_DIR / "checkpoints" / "model_stage1_plantvillage.pt",
    )
    # PlantDoc fine-tune: used for field photos, where the PlantVillage model is unreliable
    DISEASE_FIELD_MODEL_PATH: Path = _first_existing(
        MODELS / "disease_classifier_field.pt",
        BASE_DIR / "checkpoints" / "model_stage2_plantdoc.pt",
    )
    # Crop-independent disease type ("early blight", "rust", ...): training/disease_type/train.py
    DISEASE_TYPE_MODEL_PATH: Path = MODELS / "disease_type_classifier.pt"
    # Routes uploads to the right model: lab_leaf / field_leaf / fruit / other
    IMAGE_ROUTER_PATH: Path = MODELS / "image_router.pt"
    UNET_SEGMENTATION_PATH: Path = MODELS / "unet_lesion_segmentation.pt"
    # Stage-2 banana ripeness model only (the Fruits-360 stage-1 checkpoint predicts fruit type, not ripeness)
    RIPENESS_MODEL_PATH: Path = MODELS / "ripeness_stage2_banana.pt"
    # Only custom-trained detectors: stock COCO yolov8n.pt has no pest/weed classes.
    PEST_YOLO_PATH: Path = _first_existing(
        MODELS / "yolo_pest_detector.pt",
        MODELS / "pest_detector" / "weights" / "best.pt",
    )
    WEED_YOLO_PATH: Path = _first_existing(
        MODELS / "yolo_weed_detector.pt",
        MODELS / "weed_detector" / "weights" / "best.pt",
        BASE_DIR / "runs" / "detect" / "models" / "weed_detector" / "weights" / "best.pt",
    )
    # Whole-image species classifiers (training/field_classifiers/train.py), used when no YOLO is trained
    PEST_CLASSIFIER_PATH: Path = MODELS / "pest_classifier.pt"
    WEED_CLASSIFIER_PATH: Path = MODELS / "weed_classifier.pt"
    WILTING_MODEL_PATH: Path = MODELS / "wilting_detector.pt"
    # Trained on the real Kaggle Crop Recommendation CSV (synthetic-data model kept in crop_recommender_synthetic/)
    CROP_RECOMMENDER_PATH: Path = MODELS / "crop_recommender.pkl"

    # BioCLIP-2 plant species identification for crop auto-detect. Off by default: with the PlantWild-trained
    # farm model it adds only ~2 points, but it is a ~1.7 GB download and slow without a GPU. Set
    # SPECIES_ID_ENABLED=true to turn it back on (it also rejects unsupported plants such as pear or betel).
    SPECIES_ID_ENABLED: bool = False

    # Predictions below this confidence are flagged "low confidence" instead of shown as fact
    CONFIDENCE_THRESHOLD: float = 0.60

    # Live weather (Open-Meteo, no API key). Set WEATHER_LIVE=false to use the offline sample forecast.
    WEATHER_LIVE: bool = True

    # Knowledge Base
    CAUSE_LOOKUP_PATH: Path = APP_DIR / "knowledge_base" / "cause_category_lookup.json"
    DISEASE_TYPES_PATH: Path = APP_DIR / "knowledge_base" / "disease_types.json"
    PEST_TYPES_PATH: Path = APP_DIR / "knowledge_base" / "pest_types.json"

    # Database
    DATABASE_URL: str = f"sqlite:///{APP_DIR.parent}/agri_system.db"

    # Security / Auth
    SECRET_KEY: str = "smart-agri-secret-key-change-in-production-2026"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days

    # External APIs
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "demo_weather_key")

settings = Settings()
