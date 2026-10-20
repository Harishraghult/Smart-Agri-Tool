from fastapi import APIRouter
from app.config import settings
from app.services.diagnosis_service import diagnosis_service
from app.services.ripeness_service import ripeness_service
from app.services.field_service import field_service
from app.services.crop_recommender_service import crop_recommender_service

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """
    Check API health and which trained model artifacts are actually loaded in memory.
    Modules without a loaded model fall back to image/rule-based heuristics.
    """
    models_status = {
        **diagnosis_service.model_status(),
        **ripeness_service.model_status(),
        **field_service.model_status(),
        **crop_recommender_service.model_status(),
    }

    return {
        "status": "online",
        "version": settings.VERSION,
        "models": models_status,
        "models_loaded": sum(models_status.values()),
        "models_total": len(models_status),
        "system_ready": True
    }
