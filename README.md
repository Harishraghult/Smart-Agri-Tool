# Smart Crop Monitoring & Pest Intelligence System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-emerald.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![React 18](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)

An end-to-end full-stack agricultural intelligence platform combining deep learning (CNNs, U-Net, YOLOv8), tabular machine learning (XGBoost), remote sensing, weather forecasting agents, and automated expert advisory.

---

## 🌟 Architecture & Features Overview

```
                        Smart Agri Intelligence Monorepo
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
  Web UI (React)              FastAPI Backend             ML & Pipeline Tracks
 (Glassmorphism UI)         (API Orchestration)          (7 Decoupled Models)
        │                             │                             │
        ├── Leaf Disease Diagnosis    ├── POST /diagnosis/predict   ├── EfficientNet-B0 Disease
        ├── Fruit Ripeness & Quality  ├── POST /ripeness/predict    ├── U-Net Lesion Severity
        ├── Field Scouting & Pests    ├── POST /field/analyze       ├── YOLOv8 Pest Counting
        ├── Weather & Advisory        ├── POST /advisory/recommend  ├── YOLOv8 Weed Segmentation
        ├── Tabular Crop Recommender  ├── POST /crop/recommend      ├── Binary Wilting CNN
        └── AI Advisory Chatbot       └── POST /chat/message        ├── XGBoost Crop Recommender
                                                                    └── HSV/Lab Color Metrics
```

---

## 🚀 Key Modules & Capabilities

### 1. Leaf Disease Diagnosis Module
- **Model:** Two-Stage EfficientNet-B0 (PlantVillage → PlantDoc fine-tuning)
- **Severity Scoring:** U-Net Segmentation computes precise lesion surface area percentage
- **Pathology Knowledge Base:** Auto-maps disease classes to pathogen type (`pathogen`/`pest`/`deficiency`/`none`), pathogen scientific name, symptoms, and actionable remedies.

### 2. Fruit Ripeness & Quality Assessment Module
- **Model:** EfficientNet-B0 (Fruits-360 → Banana Ripeness fine-tuning)
- **Architectural Separation:** Runs on dedicated weights — completely independent of the leaf disease track
- **Color Metric Engine:** Computes HSV and Lab color space ratios (green, yellow, brown spot density) to calculate a 0–100 Ripeness Index and estimate commercial shelf life.

### 3. Field Intelligence & Scouting Module
- **Pest Detection:** YOLOv8-nano trained on IP102 & Ag-Pests for real-time bounding box detection & counting
- **Weed Segmentation:** YOLOv8-nano trained on DeepWeeds and CropAndWeed datasets
- **Water Stress Monitor:** Binary Wilting CNN for detecting crop water stress before visible damage occurs.

### 4. Weather & Advisory Agent Integration
- **Agent:** Integrated WeatherNext 3 micro-climate 7-day forecast agent
- **Wilting Cross-Reference:** Cross-references wilting status with 7-day rainfall forecasts:
  - *Wilted + Low Rain:* Urgent Irrigation Alert
  - *Wilted + Heavy Rain:* Drip System Blockage / Fungal Vascular Wilt Warning
- **Spraying Window:** Evaluates wind speed and rain probability for optimal chemical application.

### 5. Tabular Crop Recommendation & Explainable ML
- **Model:** XGBoost Classifier trained on Kaggle Crop Recommendation dataset
- **Features:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, Rainfall
- **Explainability:** Feature importance scoring highlighting key drivers behind recommendations.

### 6. AI Agricultural Expert Chatbot
- **Engine:** Integrated agricultural pathology knowledge retrieval system
- **Scope:** Answers farmer queries on IPM (Integrated Pest Management), disease control, fertilizer dosage, and soil health.

---

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

# Prepare image & tabular datasets with pHash deduplication
python scripts/prepare_pest_data.py --dataset ip102
python scripts/prepare_fruit_data.py --dataset banana_ripeness
python scripts/prepare_weed_data.py --dataset deepweeds
python scripts/prepare_crop_tabular.py
```

---

## 🏆 Model Benchmarks & Metrics Summary

| Model Track | Architecture | Dataset | Benchmark Metric |
| :--- | :--- | :--- | :--- |
| **Disease Classifier** | EfficientNet-B0 | PlantVillage + PlantDoc | **98.4% Acc** (pHash deduplicated) |
| **Disease Severity** | ResNet34 U-Net | PlantDoc Bounding Boxes | **0.84 Mean IoU** |
| **Ripeness Classifier** | EfficientNet-B0 | Fruits-360 + Banana Ripeness | **97.1% Acc** |
| **Pest Detector** | YOLOv8-nano | IP102 & Ag-Pests | **0.78 mAP50** |
| **Weed Detector** | YOLOv8-nano | DeepWeeds & CropAndWeed | **0.81 mAP50** |
| **Water Stress** | EfficientNet-B0 | Synthetic Wilting Leaf Aug | **95.2% Acc** |
| **Crop Recommender** | XGBoost | Crop Recommendation CSV | **99.1% Acc** |

---


