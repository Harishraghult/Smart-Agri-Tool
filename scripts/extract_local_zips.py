"""
extract_local_zips.py

Extracts all local raw dataset zip files into distinct, structured folders under data/raw/
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

ZIP_MAPPINGS = [
    {
        "zip": BASE_DIR / "archive (3).zip",
        "target": RAW_DIR / "plantvillage",
        "desc": "PlantVillage Disease Dataset (38 classes, 54k images)"
    },
    {
        "zip": BASE_DIR / "PlantDoc-Dataset-master.zip",
        "target": RAW_DIR / "plantdoc_cropped",
        "desc": "PlantDoc Cropped Field Leaf Disease Dataset"
    },
    {
        "zip": BASE_DIR / "PlantDoc-Object-Detection-Dataset-master.zip",
        "target": RAW_DIR / "plantdoc_objdet",
        "desc": "PlantDoc Object Detection Lesion Dataset"
    },
    {
        "zip": BASE_DIR / "cropandweed-dataset-main.zip",
        "target": RAW_DIR / "cropandweed",
        "desc": "CropAndWeed Segmentation Dataset"
    },
    {
        "zip": BASE_DIR / "archive (4).zip",
        "target": RAW_DIR / "banana_ripeness",
        "desc": "Banana Ripeness / Fruits-360 Dataset"
    },
    {
        "zip": BASE_DIR / "images (1).zip",
        "target": RAW_DIR / "agricultural_pests",
        "desc": "Agricultural Pests Dataset"
    },
    {
        "zip": BASE_DIR / "IP102-master.zip",
        "target": RAW_DIR / "ip102",
        "desc": "IP102 Pest Dataset Annotations"
    }
]

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTarget Extract Root: {RAW_DIR}\n")

    for item in ZIP_MAPPINGS:
        zip_path = item["zip"]
        target_dir = item["target"]
        desc = item["desc"]

        print(f"==================================================")
        print(f" Processing: {desc}")
        print(f" Target: {target_dir}")
        print(f"==================================================")

        if not zip_path.exists():
            print(f"  [FAIL] Zip file not found: {zip_path.name}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        if any(target_dir.iterdir()):
            print(f"  [SKIP] Already extracted in {target_dir.name}")
            continue

        print(f"  Extracting {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.1f} MB)...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"  [OK] Successfully extracted to {target_dir.name}\n")
        except Exception as e:
            print(f"  [FAIL] Error extracting {zip_path.name}: {e}\n")

if __name__ == "__main__":
    main()
