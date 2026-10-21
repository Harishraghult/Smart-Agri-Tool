from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.services.disease_type_service import disease_type_service
from app.services.diagnosis_service import (
    diagnosis_service, SUPPORTED_CROPS, available_model_choices, UnsupportedCropError, InvalidModelChoiceError,
)
from app.db.session import get_db
from app.models.db_models import DiagnosisLog
from app.services.image_utils import ImageValidationError

router = APIRouter()

@router.get("/crops")
async def supported_crops():
    """Crops the disease models can diagnose (for the crop selector)."""
    return {"crops": list(SUPPORTED_CROPS)}


@router.get("/models")
async def disease_models():
    """Disease-model choices for the model selector."""
    return {
        "models": [{"id": k, "description": v} for k, v in available_model_choices().items()],
        "default": "auto",
        "disease_type_methods": [{"id": k, "description": v}
                                 for k, v in disease_type_service.available_methods().items()],
        "disease_type_default": disease_type_service.default_method(),
    }


@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    model: Optional[str] = Form("auto"),
    disease_type_method: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Diagnose plant diseases from an uploaded leaf image.
    Returns species, condition, confidence, cause_category, severity_score %, symptoms, and remedies.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = diagnosis_service.predict(image_bytes, crop_type, model, disease_type_method)
    except (UnsupportedCropError, InvalidModelChoiceError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log diagnosis to DB
    try:
        log = DiagnosisLog(
            image_filename=file.filename,
            crop_type=result.get("crop_type", crop_type),
            predicted_class=result["predicted_class"] or f"disease_type:{result['disease_type']['type']}",
            confidence=result["confidence"],
            cause_category=result["cause_category"],
            pathogen_name=result.get("pathogen_name"),
            severity_score=result["severity_score"],
            remedies=result["remedies"]
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        result["log_id"] = log.id
    except Exception as e:
        print(f"DB Logging Note: {e}")

    return result
