from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional

from app.services.weather_advisory_service import weather_advisory_service

router = APIRouter()

class AdvisoryRequest(BaseModel):
    latitude: float = 12.9716
    longitude: float = 77.5946
    crop_type: str = "Tomato"
    is_wilted: Optional[bool] = False

@router.post("/recommend")
async def get_crop_advisory(req: AdvisoryRequest):
    """
    Get WeatherNext 3 forecast, disease outbreak risk warnings, irrigation cross-reference, and spray windows.
    """
    return weather_advisory_service.generate_advisory(
        lat=req.latitude,
        lon=req.longitude,
        crop_type=req.crop_type,
        is_wilted=req.is_wilted
    )

@router.get("/weather-forecast")
async def get_weather(lat: float = 12.9716, lon: float = 77.5946):
    """
    Direct WeatherNext 3 7-day forecast endpoint.
    """
    return weather_advisory_service.fetch_weathernext_forecast(lat, lon)
