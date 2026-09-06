from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.services.diagnosis_service import diagnosis_service
from app.db.session import get_db
from app.models.db_models import DiagnosisLog

router = APIRouter()

@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Diagnose plant diseases from an uploaded leaf image.
    Returns species, condition, confidence, cause_category, severity_score %, symptoms, and remedies.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = diagnosis_service.predict(image_bytes)

    # Log diagnosis to DB
    try:
        log = DiagnosisLog(
            image_filename=file.filename,
            crop_type=result.get("crop_type", crop_type),
            predicted_class=result["predicted_class"],
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
