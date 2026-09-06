from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.crop_recommender_service import crop_recommender_service
from app.db.session import get_db
from app.models.db_models import CropRecommendationLog

router = APIRouter()

class CropRecommendationRequest(BaseModel):
    N: float = Field(..., example=90.0, description="Nitrogen content (mg/kg)")
    P: float = Field(..., example=42.0, description="Phosphorus content (mg/kg)")
    K: float = Field(..., example=43.0, description="Potassium content (mg/kg)")
    temperature: float = Field(..., example=20.87, description="Temperature in °C")
    humidity: float = Field(..., example=82.0, description="Relative humidity in %")
    ph: float = Field(..., example=6.5, description="Soil pH value (0-14)")
    rainfall: float = Field(..., example=202.9, description="Rainfall in mm")

@router.post("/recommend")
async def recommend_crop(req: CropRecommendationRequest, db: Session = Depends(get_db)):
    """
    Recommend optimal crops from tabular soil nutrients & climate data with feature importance explanations.
    """
    result = crop_recommender_service.predict(
        n=req.N,
        p=req.P,
        k=req.K,
        temp=req.temperature,
        humidity=req.humidity,
        ph=req.ph,
        rainfall=req.rainfall
    )

    # Log to DB
    try:
        log = CropRecommendationLog(
            n=req.N, p=req.P, k=req.K,
            temperature=req.temperature, humidity=req.humidity,
            ph=req.ph, rainfall=req.rainfall,
            recommended_crop=result["recommended_crop"],
            top_alternatives=result["top_3_recommendations"]
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"DB Logging Note: {e}")

    return result
