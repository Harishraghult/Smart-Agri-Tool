"""Edge cases: bad uploads, wrong image types, degraded photos and out-of-range inputs."""
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.diagnosis_service import diagnosis_service
from app.services.image_router_service import image_router_service

client = TestClient(app, raise_server_exceptions=False)
SAMPLES = Path(__file__).resolve().parents[2] / "frontend" / "public" / "samples"
IMAGE_ENDPOINTS = ["/api/v1/diagnosis/predict", "/api/v1/ripeness/predict", "/api/v1/field/analyze"]
needs_models = pytest.mark.skipif(
    diagnosis_service.disease_model is None or image_router_service.model is None,
    reason="disease classifier / image router not trained")


def encode(img, fmt="JPEG", **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def leaf():
    return Image.open(SAMPLES / "tomato_early_blight.jpg").convert("RGB")


def post(endpoint, data, ctype="image/jpeg"):
    return client.post(endpoint, files={"file": ("upload", data, ctype)})


UNUSABLE_UPLOADS = {
    "empty": lambda: b"",
    "text disguised as jpeg": lambda: b"definitely not an image",
    "truncated jpeg": lambda: encode(Image.new("RGB", (224, 224), (90, 160, 90)))[:200],
    "1x1 pixel": lambda: encode(Image.new("RGB", (1, 1), (90, 160, 90))),
    "extreme strip": lambda: encode(Image.new("RGB", (4000, 40), (90, 160, 90))),
    "solid colour": lambda: encode(Image.new("RGB", (224, 224), (40, 120, 40))),
    "over 15 MB": lambda: b"0" * (16 * 1024 * 1024),
}


@pytest.mark.parametrize("endpoint", IMAGE_ENDPOINTS)
@pytest.mark.parametrize("name", list(UNUSABLE_UPLOADS))
def test_unusable_uploads_rejected_with_400(endpoint, name):
    r = post(endpoint, UNUSABLE_UPLOADS[name]())
    assert r.status_code == 400, f"{name}: {r.status_code} {r.text[:200]}"
    assert isinstance(r.json()["detail"], str)


@pytest.mark.parametrize("endpoint", IMAGE_ENDPOINTS)
def test_non_image_content_type_rejected(endpoint):
    assert post(endpoint, b"hello", "text/plain").status_code == 400


@needs_models
@pytest.mark.parametrize("fmt,img_fn", [
    ("PNG", lambda: leaf()),
    ("WEBP", lambda: leaf()),
    ("JPEG", lambda: leaf().convert("CMYK")),
    ("PNG", lambda: leaf().convert("RGBA")),
    ("JPEG", lambda: leaf().resize((6000, 6000))),
])
def test_other_formats_and_sizes_still_diagnosed(fmt, img_fn):
    data = post("/api/v1/diagnosis/predict", encode(img_fn(), fmt)).json()
    assert data["predicted_class"] == "Tomato___Early_blight"


@needs_models
def test_exif_rotated_photo_is_uprighted():
    img = leaf()
    exif = img.getexif()
    exif[0x0112] = 6
    data = post("/api/v1/diagnosis/predict", encode(img, exif=exif)).json()
    assert data["predicted_class"] == "Tomato___Early_blight"
    assert "exif_orientation" in data["preprocessing"]


@needs_models
def test_dark_photo_is_corrected_and_diagnosed():
    dark = Image.fromarray((np.asarray(leaf()) * 0.15).astype(np.uint8))
    data = post("/api/v1/diagnosis/predict", encode(dark)).json()
    assert "low_light_correction" in data["preprocessing"]
    assert data["predicted_class"] == "Tomato___Early_blight"


@needs_models
def test_normal_photo_is_not_altered():
    data = post("/api/v1/diagnosis/predict", encode(leaf())).json()
    assert data["preprocessing"] == []


@needs_models
@pytest.mark.parametrize("endpoint,sample", [
    ("/api/v1/diagnosis/predict", "banana_ripe.jpg"),
    ("/api/v1/diagnosis/predict", "field_pest_grasshopper.jpg"),
    ("/api/v1/ripeness/predict", "tomato_healthy.jpg"),
])
def test_wrong_image_type_is_warned_and_low_confidence(endpoint, sample):
    data = post(endpoint, (SAMPLES / sample).read_bytes()).json()
    assert data["input_warning"]
    assert data["low_confidence"] is True


@needs_models
def test_random_noise_is_flagged():
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    data = post("/api/v1/diagnosis/predict", encode(noise)).json()
    assert data["low_confidence"] is True


@pytest.mark.parametrize("payload", [
    {"N": -5}, {"ph": 20}, {"humidity": 150}, {"temperature": -80}, {"rainfall": 1e9}, {"N": "lots"}, {"N": "NaN"},
])
def test_impossible_crop_inputs_rejected(payload):
    assert client.post("/api/v1/crop/recommend", json=payload).status_code == 422


def test_crop_input_outside_training_range_is_flagged():
    data = client.post("/api/v1/crop/recommend", json={
        "N": 300, "P": 42, "K": 43, "temperature": 20, "humidity": 80, "ph": 6.5, "rainfall": 200}).json()
    assert data["low_confidence"] is True
    assert data["out_of_range_features"][0]["feature"] == "N"


@pytest.mark.parametrize("payload", [{"latitude": 200, "longitude": 0}, {"latitude": 0, "longitude": -500}])
def test_invalid_coordinates_rejected(payload):
    assert client.post("/api/v1/advisory/recommend", json=payload).status_code == 422
    assert client.get("/api/v1/advisory/weather-forecast", params={
        "lat": payload["latitude"], "lon": payload["longitude"]}).status_code == 422


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_bad_chat_messages_rejected(message):
    r = client.post("/api/v1/chat/message", json={"message": message})
    assert r.status_code == 422
    assert len(r.content) < 2000  # long input is not echoed back


def test_unicode_chat_message_ok():
    assert client.post("/api/v1/chat/message", json={"message": "தக்காளி இலை நோய் 🍅"}).status_code == 200


# --- Crop selection ---

def test_supported_crops_listed():
    crops = client.get("/api/v1/diagnosis/crops").json()["crops"]
    assert "Potato" in crops and "Tomato" in crops


@pytest.mark.parametrize("crop", ["Pear", "Betel", "Cassava"])
def test_unsupported_crop_rejected_with_reason(crop):
    r = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                    data={"crop_type": crop})
    assert r.status_code == 422
    assert "not supported" in r.json()["detail"]


@needs_models
@pytest.mark.parametrize("crop", ["Potato", "potato", "Pepper,_bell", "maize"])
def test_selected_crop_restricts_diagnosis(crop):
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                       data={"crop_type": crop}).json()
    prefix = {"potato": "Potato", "pepper,_bell": "Pepper,_bell", "maize": "Corn_(maize)"}[crop.lower()]
    assert data["predicted_class"].startswith(prefix)
    assert all(c["class"].startswith(prefix) for c in data["top_candidates"])
    assert data["crop_selected"]


@needs_models
def test_correct_crop_keeps_correct_diagnosis():
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                       data={"crop_type": "Tomato"}).json()
    assert data["predicted_class"] == "Tomato___Early_blight"


# --- Species identification (BioCLIP-2) for crop auto-detect ---

from app.services.species_service import species_service  # noqa: E402

needs_species = pytest.mark.skipif(species_service.model is None or diagnosis_service.disease_model is None,
                                   reason="BioCLIP-2 species identifier not available")


@needs_species
def test_lab_photo_not_overridden_by_species_id():
    # BioCLIP-2 is only consulted for field photos; lab photos keep the 99%-accurate PlantVillage model
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")}).json()
    assert data["crop_detected"] is None
    assert data["predicted_class"] == "Tomato___Early_blight"



# --- Disease model selection ---

def test_model_choices_listed():
    ids = [m["id"] for m in client.get("/api/v1/diagnosis/models").json()["models"]]
    assert ids == ["auto", "lab", "farm", "farm_bioclip"]


def test_unknown_model_choice_rejected():
    r = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                    data={"model": "gpt"})
    assert r.status_code == 422


@needs_models
@pytest.mark.parametrize("choice,engine", [
    ("lab", "efficientnet_b0_plantvillage"),
    ("farm", "efficientnet_b0_plantdoc"),
    ("farm_bioclip", "efficientnet_b0_plantdoc"),
    ("auto", "efficientnet_b0_plantvillage"),  # lab-style sample photo -> router picks Lab
])
def test_model_choice_selects_model(choice, engine):
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                       data={"model": choice, "crop_type": "Tomato"}).json()
    assert data["model_choice"] == choice
    assert data["inference_engine"]["classifier"] == engine


# --- Knowledge base coverage ---

def test_every_predictable_class_has_knowledge_base_entry():
    from app.services.diagnosis_service import PLANT_VILLAGE_CLASSES, PLANTDOC_TO_PLANTVILLAGE
    kb = diagnosis_service.cause_lookup
    reachable = set(PLANT_VILLAGE_CLASSES) | set(PLANTDOC_TO_PLANTVILLAGE.values())
    if diagnosis_service.field_model is not None:
        _, names = diagnosis_service.field_model
        unmapped = [n for n in names if n not in PLANTDOC_TO_PLANTVILLAGE]
        assert not unmapped, f"field-model classes without a PlantVillage mapping: {unmapped}"
    missing = sorted(c for c in reachable if c not in kb)
    assert not missing, f"no knowledge-base entry for: {missing}"
