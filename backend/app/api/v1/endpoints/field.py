from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.field_service import field_service
from app.db.session import get_db
from app.models.db_models import FieldAnalysisLog

router = APIRouter()

@router.post("/analyze")
async def analyze_field(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Field Intelligence: detects pest species & counts (YOLO), weed density (YOLO), and wilting/water stress (CNN).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = field_service.analyze(image_bytes)

    # Log to DB
    try:
        log = FieldAnalysisLog(
            pest_count=result["pest_summary"]["pest_count"],
            weed_count=result["weed_summary"]["weed_count"],
            wilting_detected=1 if result["water_stress"]["is_wilted"] else 0,
            irrigation_alert=result["water_stress"]["status"]
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"DB Logging Note: {e}")

    return result
