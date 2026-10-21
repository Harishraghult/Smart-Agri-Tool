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


@pytest.mark.parametrize("crop", ["Pear", "Betel", "Cassava", "other"])
def test_unsupported_crop_gets_disease_type_or_clear_rejection(crop):
    from app.services.disease_type_service import disease_type_service
    r = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                    data={"crop_type": crop})
    if disease_type_service.model is None:
        assert r.status_code == 422
        assert "not supported" in r.json()["detail"]
    else:  # crop-specific models can't help, so the disease-type model answers on its own
        data = r.json()
        assert r.status_code == 200 and data["disease_type_only"] is True
        assert data["predicted_class"] is None and data["disease_type"]["method"] == "model"


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
    expected = ["auto", "lab", "farm"] + (["farm_bioclip"] if species_service.model is not None else [])
    assert ids == expected


def test_bioclip_choice_rejected_when_disabled():
    if species_service.model is not None:
        pytest.skip("BioCLIP-2 is enabled")
    r = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                    data={"model": "farm_bioclip"})
    assert r.status_code == 422
    assert "turned off" in r.json()["detail"]


def test_unknown_model_choice_rejected():
    r = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                    data={"model": "gpt"})
    assert r.status_code == 422


@needs_models
@pytest.mark.parametrize("choice,engine", [
    ("lab", "efficientnet_b0_plantvillage"),
    ("farm", "efficientnet_b0_plantdoc"),
    pytest.param("farm_bioclip", "efficientnet_b0_plantdoc",
                 marks=pytest.mark.skipif(species_service.model is None, reason="BioCLIP-2 disabled")),
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



# --- Crop-independent disease type ---

from app.services.disease_type_service import disease_type_service  # noqa: E402


@needs_models
def test_disease_type_aggregated_from_crop_model():
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                       data={"disease_type_method": "aggregated"}).json()
    dt = data["disease_type"]
    assert dt["type"] == "early_blight" and dt["method"] == "aggregated"
    assert "Lab model" in dt["model"] and data["models_used"]["disease_type"] == dt["model"]


@pytest.mark.skipif(disease_type_service.model is None, reason="disease-type model not trained")
def test_disease_type_model_on_supported_crop():
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                       data={"disease_type_method": "model"}).json()
    assert data["disease_type"]["type"] == "early_blight" and data["disease_type"]["method"] == "model"
    assert data["predicted_class"] == "Tomato___Early_blight"  # crop-specific answer unchanged


def test_disease_type_methods_listed():
    data = client.get("/api/v1/diagnosis/models").json()
    ids = [m["id"] for m in data["disease_type_methods"]]
    assert "aggregated" in ids
    assert ("model" in ids) == (disease_type_service.model is not None)


def test_every_disease_type_has_farmer_info():
    for t, info in disease_type_service.types.items():
        assert info["display_name"] and "management" in info, t


@pytest.mark.skipif(disease_type_service.model is None, reason="disease-type model not trained")
def test_disease_type_combined_method_and_default():
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")},
                       data={"disease_type_method": "combined"}).json()
    assert data["disease_type"]["method"] == "combined" and data["disease_type"]["type"] == "early_blight"
    menu = client.get("/api/v1/diagnosis/models").json()
    assert menu["disease_type_default"] == "model"
    assert [m["id"] for m in menu["disease_type_methods"]] == ["model", "combined", "aggregated"]


# --- Single plant + disease answer ---

@needs_models
def test_single_plant_and_disease_answer():
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")}).json()
    assert data["plant"] == "Tomato" and data["disease"] == "Early blight"
    assert all(c["plant"] and c["disease"] for c in data["top_candidates"])


@needs_models
def test_healthy_leaf_reports_healthy():
    img = Image.open(SAMPLES / "tomato_healthy.jpg").convert("RGB")
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(img), "image/jpeg")}).json()
    assert data["plant"] == "Tomato" and data["disease"] == "Healthy" and data["is_healthy"]


@pytest.mark.skipif(disease_type_service.model is None, reason="disease-type model not trained")
def test_joint_answer_is_default_when_disease_type_model_loaded():
    data = client.post("/api/v1/diagnosis/predict", files={"file": ("x", encode(leaf()), "image/jpeg")}).json()
    assert data["answer_method"] == "joint"


# --- Pest / weed confidence calibration ---

from app.services.field_service import field_service  # noqa: E402


@pytest.mark.skipif(field_service.pest_classifier is None, reason="pest classifier not trained")
def test_pest_and_weed_models_are_calibrated():
    for clf in (field_service.pest_classifier, field_service.weed_classifier):
        model, _ = clf
        assert 0.3 < model.temperature < 1.0  # under-confident models get sharpened, not flattened


@pytest.mark.skipif(field_service.pest_classifier is None, reason="pest classifier not trained")
def test_uncertain_pest_is_flagged_not_reported_clean():
    data = post("/api/v1/field/analyze", (SAMPLES / "field_pest_grasshopper.jpg").read_bytes()).json()
    p = data["pest_summary"]
    if p["identification"]["low_confidence"] and p["identification"]["kind"] == "pest":
        assert p["risk_level"] == "Uncertain" and p["possible"]
    else:
        assert p["possible"] is None
    assert p["pest_count"] == 1 and "Grasshopper" in p["pest_detected_list"][0]["group"]
    assert p["management"]  # group treatment advice is returned


@pytest.mark.skipif(field_service.pest_classifier is None, reason="pest classifier not trained")
def test_every_pest_class_has_name_and_group():
    _, names = field_service.pest_classifier
    labels, groups = field_service.pest_types["labels"], field_service.pest_types["groups"]
    for n in names:
        key = field_service._pest_key(n)
        assert key in labels and labels[key]["group"] in groups, n


@pytest.mark.skipif(field_service.pest_classifier is None, reason="pest classifier not trained")
def test_plain_leaf_has_no_pest():
    data = post("/api/v1/field/analyze", (SAMPLES / "tomato_healthy.jpg").read_bytes()).json()
    assert data["pest_summary"]["pest_count"] == 0
