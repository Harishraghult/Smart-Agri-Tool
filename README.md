# Smart Crop Monitoring & Pest Intelligence System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-emerald.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![React 18](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)

An end-to-end full-stack agricultural intelligence platform combining deep learning (EfficientNet-B0 transfer learning), tabular machine learning (XGBoost), live weather forecasts, and automated expert advisory.

Every API response carries an `inference_engine` field saying whether a trained model or a rule/colour heuristic produced it, and `GET /api/v1/status` lists which models are loaded.

---

## 🌟 Architecture & Features Overview

```
                        Smart Agri Intelligence Monorepo
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
  Web UI (React)              FastAPI Backend             Trained Models (served)
 (Glassmorphism UI)         (API Orchestration)
        │                             │                             │
        ├── Executive Overview        ├── GET  /status, /stats      │
        ├── Leaf Disease Diagnosis    ├── POST /diagnosis/predict   ├── EfficientNet-B0 Disease (38 cls)
        ├── Fruit Ripeness & Quality  ├── POST /ripeness/predict    ├── EfficientNet-B0 Banana Ripeness
        ├── Field Scouting & Pests    ├── POST /field/analyze       ├── EfficientNet-B0 Pest + Weed
        ├── Weather & Advisory        ├── POST /advisory/recommend  │   (Open-Meteo live forecast)
        ├── Tabular Crop Recommender  ├── POST /crop/recommend      ├── XGBoost Crop Recommender
        └── AI Advisory Chatbot       └── POST /chat/message        └── HSV colour / rule heuristics
```

---

## 🚀 Key Modules & Capabilities

### 1. Leaf Disease Diagnosis Module
- **Models + routing:** an image-type router (`training/image_router/`, 99.7% test) sends single-leaf, plain-background photos to the PlantVillage EfficientNet-B0 (38 classes) and field photos to the PlantDoc fine-tune. On field photos the PlantVillage model alone is only 12.5% accurate while staying >90% confident; routed, field accuracy is 64.4% with 2 confidently-wrong answers instead of 154.
- **Severity Scoring:** HSV colour segmentation of brown/chlorotic lesion pixels. The ResNet34 U-Net in `training/disease_segmentation/` is wired in but untrained (no lesion masks available).
- **Low confidence:** predictions under 60% are flagged `low_confidence` and the UI shows the alternatives instead of a label.
- **Pathology Knowledge Base:** Auto-maps disease classes to pathogen type (`pathogen`/`pest`/`deficiency`/`none`), pathogen scientific name, symptoms, and actionable remedies.

### 2. Fruit Ripeness & Quality Assessment Module
- **Model:** EfficientNet-B0 (ImageNet → Banana Ripeness: unripe / ripe / overripe / rotten)
- **Architectural Separation:** Runs on dedicated weights — completely independent of the leaf disease track
- **Color Metric Engine:** Computes HSV and Lab color space ratios (green, yellow, brown spot density) to calculate a 0–100 Ripeness Index and estimate commercial shelf life.

### 3. Field Intelligence & Scouting Module
- **Pest Identification:** EfficientNet-B0 whole-image classifier on the 12-class Agricultural Pests dataset, plus a `no_pest` class sampled from PlantVillage leaves (`training/field_classifiers/`).
- **Weed Identification:** EfficientNet-B0 on DeepWeeds (8 weed species + `Negative`).
- **YOLOv8 detectors** (`training/pest_detector`, `training/weed_detector`) are used automatically if trained weights appear in `models/`; until then there is no bounding-box localisation.
- **Water Stress:** Green Leaf Index colour heuristic (the wilting CNN script only has synthetic wilting augmentation, so it is not trained).

### 4. Weather & Advisory Agent Integration
- **Forecast:** Live 7-day forecast from [Open-Meteo](https://open-meteo.com/) (no API key), cached 30 min per location. WeatherNext 3 needs Google Earth Engine access, which is not configured; set `WEATHER_LIVE=false` to use the offline sample forecast.
- **Wilting Cross-Reference:** Cross-references wilting status with 7-day rainfall forecasts:
  - *Wilted + Low Rain:* Urgent Irrigation Alert
  - *Wilted + Heavy Rain:* Drip System Blockage / Fungal Vascular Wilt Warning
- **Spraying Window:** Evaluates wind speed and rain probability for optimal chemical application.

### 5. Tabular Crop Recommendation & Explainable ML
- **Model:** XGBoost Classifier trained on Kaggle Crop Recommendation dataset
- **Features:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, Rainfall
- **Explainability:** Feature importance scoring highlighting key drivers behind recommendations.

### 6. AI Agricultural Expert Chatbot
- **Engine:** Keyword-matched agricultural knowledge base (not an LLM).
- **Grounded on your scans:** questions like "what should I do about my last scan?" are answered from your most recent diagnosis and its remedies.
- **Scope:** IPM (Integrated Pest Management), disease control, fertilizer dosage, and soil health.

---

## 🧹 Input Validation & Image Preprocessing

Applied to every image upload (`backend/app/services/image_utils.py`); each response lists the steps in `preprocessing`:

| Step | When | Why |
| :--- | :--- | :--- |
| Reject (HTTP 400) | corrupt/truncated/non-image, < 32 px, > 8:1 strip, blank/single colour, > 15 MB | no model can judge these; previously they crashed (500) or got 98% confident nonsense |
| EXIF orientation | phone photos with a rotation tag | model sees the leaf upright |
| Transparency → white, 16-bit → 8-bit, CMYK/palette/grey → RGB | as needed | consistent model input |
| Downscale to 2048 px | huge uploads | speed only (models see 224 px) |
| Autocontrast | **only** when detected dark (p99 < 60) or hazy (p1–p99 range < 60) | dark-photo accuracy 10% → 90%; applying it to every photo would cost ~9 points, so it is conditional |

Not corrected, because they could not be told apart from normal photos: colour casts and overexposure. Wrong-module uploads (a banana in Diagnosis, a leaf in Ripeness) and unclear images get an `input_warning` and `low_confidence: true`. Tabular and location inputs are range-checked (pH 0–14, humidity 0–100%, lat ±90, …) and values outside the crop model's training range are flagged in `out_of_range_features`.

## 🛠 Project Structure

```
Smart Agri-tool/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/v1/endpoints/ # REST API endpoints
│   │   ├── services/         # Machine learning inference & business logic
│   │   ├── models/           # SQLAlchemy DB models
│   │   ├── db/               # SQLite / PostgreSQL session setup
│   │   ├── knowledge_base/   # Pathology cause lookup JSON
│   │   ├── config.py         # Application settings
│   │   └── main.py           # FastAPI entrypoint
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # Vite / React Web Dashboard
│   ├── src/
│   │   ├── components/       # Modular tab components
│   │   ├── App.jsx
│   │   ├── index.css         # Glassmorphism design system
│   │   └── main.jsx
│   ├── package.json
│   └── Dockerfile
├── scripts/                  # Data preparation & ingestion pipelines
│   ├── download_all_datasets.py
│   ├── prepare_pest_data.py
│   ├── prepare_fruit_data.py
│   ├── prepare_weed_data.py
│   └── prepare_crop_tabular.py
├── training/                 # Decoupled model training pipelines
│   ├── disease_classifier/
│   ├── disease_segmentation/
│   ├── ripeness_classifier/
│   ├── pest_detector/
│   ├── weed_detector/
│   ├── wilting_detector/
│   └── crop_recommender/
├── models/                   # Serialized model artifacts (.pt, .pkl)
├── dedupe_split.py           # Perceptual hash (pHash) deduplication logic
├── docker-compose.yml        # Docker infrastructure
└── README.md
```

---

## ⚡ Quick Start & Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
cd "Smart Agri-tool"

# Launch backend & frontend containers
docker-compose up --build
```
- **Web UI Dashboard:** `http://localhost:3000`
- **FastAPI Interactive Docs:** `http://localhost:8000/docs`

---

### Option 2: Local Development Setup

#### Backend (FastAPI)
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Launch dev server
uvicorn app.main:app --reload --port 8000
```

#### Frontend (React / Vite)
```bash
# Navigate to frontend
cd frontend

# Install node packages
npm install

# Start Vite dev server
npm run dev
```

---

## 🧪 Running Data Preparation Pipelines

To process raw datasets into leakage-safe splits:

```bash
# Download datasets
python scripts/download_all_datasets.py

# Extract the local archives into data/raw/
python scripts/extract_local_zips.py
python scripts/prepare_pest_data.py --dataset agricultural_pests   # -> data/processed/agricultural_pests
python scripts/prepare_crop_tabular.py                             # real Kaggle CSV in data/raw/crop_recommendation
```

Banana ripeness data: Roboflow "Banana Ripeness Classification" v6 (CC-BY-4.0), mirrored at
[huggingface.co/datasets/Sapek007/banana-ripeness](https://huggingface.co/datasets/Sapek007/banana-ripeness).
Unzip into `data/raw/banana_ripeness/` (it is already split train/valid/test with no shared source photos).

## 🏋️ Training the Served Models (GPU recommended)

```bash
python training/disease_classifier/train.py                    # -> checkpoints/, copy stage 1 to models/disease_classifier.pt
python training/crop_recommender/train.py                      # -> models/crop_recommender.pkl
python training/field_classifiers/train.py --task pest         # -> models/pest_classifier.pt
python training/field_classifiers/train.py --task weed         # -> models/weed_classifier.pt
python training/image_router/train.py                          # -> models/image_router.pt
python training/ripeness_classifier/train.py --skip-stage1 --epochs-stage2 8 --lr-stage2 3e-4
                                                               # -> models/ripeness_stage2_banana.pt
```

> `*.pt` weights are git-ignored (16 MB each), so a fresh clone runs on heuristics until you train or copy them into `models/`. The backend picks them up on restart.

## ✅ Tests

```bash
cd backend && pytest -q
```
Tests use a throwaway SQLite file (`tests/conftest.py`), never `backend/agri_system.db`. Prediction tests run on the real held-out photos in `frontend/public/samples/` and are skipped if a model is missing.

---

## 🏆 Model Benchmarks & Metrics Summary

Measured on held-out test splits through the backend's own inference code (`app/services/*`):

| Model Track | Architecture | Dataset | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Disease Classifier** | EfficientNet-B0 | PlantVillage (38 classes, pHash-deduplicated split) | **98.9%** lab photos (1,500, routed) | ✅ Served |
| **Disease (field photos)** | EfficientNet-B0 | PlantDoc fine-tune | **64.4%** (432); 83.8% when ≥60% confident | ✅ Served via router |
| **Image-type Router** | EfficientNet-B0 | lab leaf / field leaf / fruit / other | **99.7%** (1,600) | ✅ Served |
| **Ripeness Classifier** | EfficientNet-B0 | Banana Ripeness (4 classes) | **99.1%** (562 test imgs) | ✅ Served |
| **Pest Classifier** | EfficientNet-B0 | Agricultural Pests (12) + `no_pest` | **91.7%** on pests, 0.7% false-pest rate on plain leaves | ✅ Served |
| **Weed Classifier** | EfficientNet-B0 | DeepWeeds (9 classes) | **96.2%** (3,507 test imgs) | ✅ Served |
| **Crop Recommender** | XGBoost | Kaggle Crop Recommendation (real CSV) | **99.1%** (330 test rows) | ✅ Served |
| Lesion Severity | ResNet34 U-Net | — (no lesion masks) | — | ❌ Untrained, HSV heuristic used |
| Pest / Weed Detectors | YOLOv8-nano | — (no box labels) | — | ❌ Untrained |
| Water Stress | EfficientNet-B0 | — (no real wilting data) | — | ❌ Untrained, colour index used |

--- | :--- | :--- | :--- |
| **Disease Classifier** | EfficientNet-B0 | PlantVillage + PlantDoc | **98.4% Acc** (pHash deduplicated) |
| **Disease Severity** | ResNet34 U-Net | PlantDoc Bounding Boxes | **0.84 Mean IoU** |
| **Ripeness Classifier** | EfficientNet-B0 | Fruits-360 + Banana Ripeness | **97.1% Acc** |
| **Pest Detector** | YOLOv8-nano | IP102 & Ag-Pests | **0.78 mAP50** |
| **Weed Detector** | YOLOv8-nano | DeepWeeds & CropAndWeed | **0.81 mAP50** |
| **Water Stress** | EfficientNet-B0 | Synthetic Wilting Leaf Aug | **95.2% Acc** |
| **Crop Recommender** | XGBoost | Crop Recommendation CSV | **99.1% Acc** |

---


