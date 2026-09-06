import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        class BaseSettings:
            pass

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Crop Monitoring & Pest Intelligence System API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model Artifact Paths
    MODELS_DIR: Path = BASE_DIR / "models"
    DISEASE_MODEL_PATH: Path = BASE_DIR / "models" / "disease_classifier.pt"
    UNET_SEGMENTATION_PATH: Path = BASE_DIR / "models" / "unet_lesion_segmentation.pt"
    RIPENESS_MODEL_PATH: Path = BASE_DIR / "models" / "ripeness_stage2_banana.pt"
    PEST_YOLO_PATH: Path = BASE_DIR / "models" / "yolo_pest_detector.pt"
    WEED_YOLO_PATH: Path = BASE_DIR / "models" / "yolo_weed_detector.pt"
    WILTING_MODEL_PATH: Path = BASE_DIR / "models" / "wilting_detector.pt"
    CROP_RECOMMENDER_PATH: Path = BASE_DIR / "models" / "crop_recommender.pkl"
    
    # Knowledge Base
    CAUSE_LOOKUP_PATH: Path = BASE_DIR / "backend" / "app" / "knowledge_base" / "cause_category_lookup.json"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/backend/agri_system.db"
    
    # Security / Auth
    SECRET_KEY: str = "smart-agri-secret-key-change-in-production-2026"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    # External APIs
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "demo_weather_key")

settings = Settings()
