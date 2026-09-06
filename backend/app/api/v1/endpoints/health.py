from fastapi import APIRouter
from app.config import settings

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """
    Check API health and status of loaded model artifacts.
    """
    models_status = {
        "disease_classifier": settings.DISEASE_MODEL_PATH.exists(),
        "unet_segmentation": settings.UNET_SEGMENTATION_PATH.exists(),
        "ripeness_classifier": settings.RIPENESS_MODEL_PATH.exists(),
        "pest_yolo": settings.PEST_YOLO_PATH.exists(),
        "weed_yolo": settings.WEED_YOLO_PATH.exists(),
        "wilting_detector": settings.WILTING_MODEL_PATH.exists(),
        "crop_recommender": settings.CROP_RECOMMENDER_PATH.exists(),
        "cause_category_lookup": settings.CAUSE_LOOKUP_PATH.exists(),
    }
    
    all_ready = any(models_status.values()) # at least fallback ready

    return {
        "status": "online",
        "version": settings.VERSION,
        "models": models_status,
        "system_ready": True
    }
