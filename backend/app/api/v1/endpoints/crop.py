from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.crop_recommender_service import crop_recommender_service
from app.db.session import get_db
from app.models.db_models import CropRecommendationLog

router = APIRouter()

from typing import Optional
from pydantic import BaseModel, Field, model_validator

class CropRecommendationRequest(BaseModel):
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    n: Optional[float] = None
    p: Optional[float] = None
    k: Optional[float] = None
    temperature: float = Field(22.0, example=20.87, description="Temperature in °C")
    humidity: float = Field(70.0, example=82.0, description="Relative humidity in %")
    ph: float = Field(6.5, example=6.5, description="Soil pH value (0-14)")
    rainfall: float = Field(100.0, example=202.9, description="Rainfall in mm")

    @model_validator(mode="after")
    def reconcile_npk(self):
        if self.N is None:
            self.N = self.n if self.n is not None else 50.0
        if self.P is None:
            self.P = self.p if self.p is not None else 50.0
        if self.K is None:
            self.K = self.k if self.k is not None else 50.0
        return self

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
