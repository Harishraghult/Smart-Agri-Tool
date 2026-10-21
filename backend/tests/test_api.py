import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app

client = TestClient(app)


def create_dummy_image_bytes():
    # Textured (not solid) so it passes the blank-image check
    import numpy as np
    pixels = np.random.default_rng(0).integers(60, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
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


# --- End-to-end prediction checks on real held-out test images (frontend/public/samples) ---

from pathlib import Path

from app.services.diagnosis_service import diagnosis_service
from app.services.field_service import field_service
from app.services.crop_recommender_service import crop_recommender_service

SAMPLES = Path(__file__).resolve().parents[2] / "frontend" / "public" / "samples"


def _post_sample(url, name):
    with open(SAMPLES / name, "rb") as f:
        return client.post(url, files={"file": (name, f.read(), "image/jpeg")}).json()


@pytest.mark.skipif(diagnosis_service.disease_model is None, reason="disease classifier not trained")
@pytest.mark.parametrize("name,expected", [
    ("tomato_early_blight.jpg", "Tomato___Early_blight"),
    ("potato_late_blight.jpg", "Potato___Late_blight"),
    ("tomato_healthy.jpg", "Tomato___healthy"),
])
def test_disease_prediction_on_real_leaf(name, expected):
    data = _post_sample("/api/v1/diagnosis/predict", name)
    assert data["predicted_class"] == expected
    assert data["low_confidence"] is False


@pytest.mark.skipif(field_service.pest_classifier is None, reason="pest classifier not trained")
def test_pest_identification_on_real_photo():
    data = _post_sample("/api/v1/field/analyze", "field_pest_grasshopper.jpg")
    assert data["pest_summary"]["identification"]["label"] == "grasshopper"


@pytest.mark.skipif(field_service.weed_classifier is None, reason="weed classifier not trained")
def test_weed_identification_on_real_photo():
    data = _post_sample("/api/v1/field/analyze", "field_weed_lantana.jpg")
    assert data["weed_summary"]["identification"]["label"] == "Lantana"


@pytest.mark.skipif(crop_recommender_service.model is None, reason="crop recommender not trained")
def test_crop_recommender_uses_trained_model():
    data = client.post("/api/v1/crop/recommend", json={
        "N": 90, "P": 42, "K": 43, "temperature": 20.88, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
    }).json()
    assert data["inference_engine"] == "xgboost"
    assert data["recommended_crop"] == "Rice"


def test_status_reports_loaded_models():
    data = client.get("/api/v1/status").json()
    assert data["models"]["disease_classifier"] == (diagnosis_service.disease_model is not None)
    assert data["models_loaded"] == sum(data["models"].values())


def test_dashboard_stats():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert len(data["trend_7d"]) == 7
    assert "model_metrics" in data


def test_chatbot_grounded_on_latest_scan():
    _post_sample("/api/v1/diagnosis/predict", "tomato_early_blight.jpg")
    data = client.post("/api/v1/chat/message", json={"message": "What should I do about my last scan?"}).json()
    assert "Early blight" in data["bot_response"]
    assert data.get("grounded_on_scan_id") is not None


from app.services.ripeness_service import ripeness_service


@pytest.mark.skipif(ripeness_service.model is None, reason="ripeness classifier not trained")
@pytest.mark.parametrize("name,expected", [
    ("banana_unripe.jpg", "Unripe (Green)"),
    ("banana_ripe.jpg", "Ripe (Yellow)"),
    ("banana_rotten.jpg", "Decayed"),
])
def test_ripeness_prediction_on_real_banana(name, expected):
    data = _post_sample("/api/v1/ripeness/predict", name)
    assert data["inference_engine"] == "efficientnet_b0"
    assert data["ripeness_stage"] == expected
