import requests
from typing import Optional
from app.config import settings

import time

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 30 * 60

# WMO weather interpretation codes -> short condition label
WMO_CONDITIONS = {
    0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle", 61: "Light Rain", 63: "Rain",
    65: "Heavy Rain", 66: "Freezing Rain", 67: "Freezing Rain", 71: "Light Snow", 73: "Snow",
    75: "Heavy Snow", 80: "Scattered Showers", 81: "Showers", 82: "Heavy Showers",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


class WeatherAdvisoryService:
    def __init__(self):
        self._cache = {}

    def _fetch_open_meteo(self, lat: float, lon: float) -> dict:
        """
        Live 7-day forecast from Open-Meteo (free, no API key).
        NOTE: WeatherNext 3 (Google) needs Earth Engine / Maps Platform access that is not
        configured here, so Open-Meteo is the documented fallback weather source.
        """
        key = (round(lat, 2), round(lon, 2))
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

        resp = requests.get(OPEN_METEO_URL, timeout=8, params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 7,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "wind_speed_10m_max,relative_humidity_2m_mean,weather_code",
        })
        resp.raise_for_status()
        data = resp.json()
        d = data["daily"]
        daily_forecast = [
            {
                "day": "Today" if i == 0 else f"Day {i + 1}",
                "date": d["time"][i],
                "temp_max": d["temperature_2m_max"][i],
                "temp_min": d["temperature_2m_min"][i],
                "humidity": round(d["relative_humidity_2m_mean"][i] or 0),
                "rainfall_mm": d["precipitation_sum"][i] or 0.0,
                "wind_kmh": d["wind_speed_10m_max"][i],
                "condition": WMO_CONDITIONS.get(d["weather_code"][i], "Unknown"),
            }
            for i in range(len(d["time"]))
        ]
        current = data["current"]
        result = {
            "current_temp_c": current["temperature_2m"],
            "current_humidity_pct": current["relative_humidity_2m"],
            "current_wind_kmh": current["wind_speed_10m"],
            "daily_forecast": daily_forecast,
        }
        self._cache[key] = (time.time(), result)
        return result

    def fetch_weathernext_forecast(self, lat: float = 12.9716, lon: float = 77.5946) -> dict:
        """
        Returns 7-day meteorological forecast including rainfall, humidity, and wind speed.
        Live data from Open-Meteo; falls back to an offline sample forecast if unreachable.
        """
        if settings.WEATHER_LIVE:
            try:
                live = self._fetch_open_meteo(lat, lon)
                daily_forecast = live["daily_forecast"]
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "source": "open-meteo",
                    "current_temp_c": live["current_temp_c"],
                    "current_humidity_pct": live["current_humidity_pct"],
                    "current_wind_kmh": live["current_wind_kmh"],
                    "total_7day_rainfall_mm": round(sum(x["rainfall_mm"] for x in daily_forecast), 1),
                    "avg_7day_humidity_pct": round(sum(x["humidity"] for x in daily_forecast) / len(daily_forecast), 1),
                    "daily_forecast": daily_forecast,
                }
            except Exception as e:
                print(f"Open-Meteo unavailable, using offline sample forecast: {e}")

        # Offline sample forecast (not location-specific)
        daily_forecast = [
            {"day": "Today", "temp_max": 29.5, "temp_min": 21.0, "humidity": 78, "rainfall_mm": 2.5, "wind_kmh": 11.0, "condition": "Partly Cloudy"},
            {"day": "Day 2", "temp_max": 31.0, "temp_min": 22.5, "humidity": 82, "rainfall_mm": 0.0, "wind_kmh": 9.5, "condition": "Sunny"},
            {"day": "Day 3", "temp_max": 30.0, "temp_min": 21.5, "humidity": 85, "rainfall_mm": 12.0, "wind_kmh": 14.0, "condition": "Scattered Showers"},
            {"day": "Day 4", "temp_max": 28.5, "temp_min": 20.0, "humidity": 88, "rainfall_mm": 24.5, "wind_kmh": 18.5, "condition": "Heavy Rain"},
            {"day": "Day 5", "temp_max": 27.0, "temp_min": 19.5, "humidity": 90, "rainfall_mm": 8.0, "wind_kmh": 13.0, "condition": "Light Rain"},
            {"day": "Day 6", "temp_max": 29.0, "temp_min": 20.5, "humidity": 75, "rainfall_mm": 0.0, "wind_kmh": 10.0, "condition": "Clear"},
            {"day": "Day 7", "temp_max": 30.5, "temp_min": 21.0, "humidity": 70, "rainfall_mm": 0.0, "wind_kmh": 8.0, "condition": "Sunny"},
        ]

        total_7day_rainfall = sum(d["rainfall_mm"] for d in daily_forecast)
        avg_humidity = sum(d["humidity"] for d in daily_forecast) / len(daily_forecast)

        return {
            "latitude": lat,
            "longitude": lon,
            "source": "offline-sample",
            "current_temp_c": daily_forecast[0]["temp_max"],
            "current_humidity_pct": daily_forecast[0]["humidity"],
            "current_wind_kmh": daily_forecast[0]["wind_kmh"],
            "total_7day_rainfall_mm": round(total_7day_rainfall, 1),
            "avg_7day_humidity_pct": round(avg_humidity, 1),
            "daily_forecast": daily_forecast
        }

    def generate_advisory(self, lat: float, lon: float, crop_type: str, is_wilted: bool = False) -> dict:
        weather = self.fetch_weathernext_forecast(lat, lon)
        rainfall_7d = weather["total_7day_rainfall_mm"]
        curr_humidity = weather["current_humidity_pct"]
        curr_wind = weather["current_wind_kmh"]
        curr_temp = weather["current_temp_c"]

        # 1. Irrigation & Water Stress Analysis (Wilting x Weather Cross-Reference)
        if is_wilted and rainfall_7d < 10.0:
            irrigation_status = "URGENT_IRRIGATION_REQUIRED"
            irrigation_advice = f"Field shows wilting and only {rainfall_7d}mm rain expected in 7 days. Irrigate immediately (apply 25-30mm water)."
        elif is_wilted and rainfall_7d >= 10.0:
            irrigation_status = "IRRIGATION_SYSTEM_FAULT_OR_PATHOLOGY"
            irrigation_advice = f"Field shows wilting despite {rainfall_7d}mm incoming rain. Inspect drip lines for blockages or check roots for fungal vascular wilt (Fusarium/Verticillium)."
        elif not is_wilted and rainfall_7d > 30.0:
            irrigation_status = "HOLD_IRRIGATION"
            irrigation_advice = f"Soil moisture adequate and heavy rain ({rainfall_7d}mm) expected. Pause automated irrigation to prevent waterlogging."
        else:
            irrigation_status = "NORMAL_SCHEDULE"
            irrigation_advice = "Maintain standard drip irrigation schedule."

        # 2. Disease Outbreak Risk Engine
        disease_risks = []
        if curr_humidity > 80 and (20.0 <= curr_temp <= 28.0):
            disease_risks.append({
                "disease": "Late Blight / Downy Mildew",
                "risk_level": "HIGH",
                "trigger": f"High humidity ({curr_humidity}%) combined with warm temperature ({curr_temp}°C)",
                "precaution": "Apply preventative copper hydroxide or chlorothalonil spray before upcoming rain."
            })
        if rainfall_7d > 20.0:
            disease_risks.append({
                "disease": "Bacterial Spot & Root Rot",
                "risk_level": "MEDIUM-HIGH",
                "trigger": f"Forecasted heavy rainfall ({rainfall_7d}mm)",
                "precaution": "Ensure field drainage channels are clear of debris."
            })
        if not disease_risks:
            disease_risks.append({
                "disease": "General Foliar Pathogens",
                "risk_level": "LOW",
                "trigger": "Favorable weather conditions",
                "precaution": "Continue standard crop monitoring."
            })

        # 3. Chemical Spraying Window Advisory
        today_rain = weather["daily_forecast"][0]["rainfall_mm"]
        if curr_wind < 15.0 and today_rain < 2.0:
            spray_window = "OPTIMAL"
            spray_advice = f"Wind speed {curr_wind} km/h is low and rain risk is minimal today. Good conditions for pesticide/foliar application."
        elif curr_wind >= 15.0:
            spray_window = "UNFAVORABLE (High Wind)"
            spray_advice = f"Wind speed {curr_wind} km/h exceeds 15 km/h limit. High risk of chemical drift; postpone spraying."
        else:
            spray_window = "UNFAVORABLE (Rain Expected)"
            spray_advice = "Rain expected today; chemical application will be washed off."

        return {
            "location": {"lat": lat, "lon": lon},
            "crop_type": crop_type,
            "weather_summary": weather,
            "irrigation_advisory": {
                "status": irrigation_status,
                "advice": irrigation_advice
            },
            "disease_outbreak_risks": disease_risks,
            "spraying_advisory": {
                "window": spray_window,
                "advice": spray_advice
            }
        }


weather_advisory_service = WeatherAdvisoryService()
