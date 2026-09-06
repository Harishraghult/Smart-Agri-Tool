import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app

client = TestClient(app)


def create_dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color=(100, 180, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_status():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["system_ready"] is True


def test_disease_diagnosis_endpoint():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/diagnosis/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "severity_score" in data
    assert "cause_category" in data
    assert "remedies" in data


def test_ripeness_endpoint():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/ripeness/predict",
        files={"file": ("fruit.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "ripeness_stage" in data
    assert "quality_grade" in data
    assert "ripeness_index" in data


def test_field_intelligence_endpoint():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/field/analyze",
        files={"file": ("field.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "field_health_score" in data
    assert "pest_summary" in data
    assert "weed_summary" in data
    assert "water_stress" in data


def test_weather_advisory_endpoint():
    payload = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "crop_type": "Tomato",
        "is_wilted": False
    }
    response = client.post("/api/v1/advisory/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "weather_summary" in data
    assert "irrigation_advisory" in data
    assert "spraying_advisory" in data


def test_crop_recommender_endpoint():
    payload = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.87,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }
    response = client.post("/api/v1/crop/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_crop" in data
    assert "top_3_recommendations" in data
    assert "feature_importances" in data


def test_chatbot_endpoint():
    payload = {
        "session_id": "test_session",
        "message": "How do I control tomato early blight?",
        "history": []
    }
    response = client.post("/api/v1/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "bot_response" in data
    assert "suggested_followups" in data
