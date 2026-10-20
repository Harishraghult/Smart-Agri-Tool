from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ripeness_service import ripeness_service
from app.services.image_utils import ImageValidationError

router = APIRouter()

@router.post("/predict")
async def predict_ripeness(file: UploadFile = File(...)):
    """
    Assess fruit ripeness stage, quality grade, HSV ripeness index, and storage recommendation.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = ripeness_service.predict(image_bytes)
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
