"""
download_datasets.py

Downloads PlantVillage and PlantDoc into data/raw/ using the Kaggle API.

SETUP (one-time, do this before running):
1. pip install -r requirements.txt
2. Get a Kaggle API token:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New Token" -> downloads kaggle.json
3. Place the token at ~/.kaggle/kaggle.json (Linux/Mac) or
   C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)
   chmod 600 ~/.kaggle/kaggle.json   # Linux/Mac only
4. Run: python scripts/download_datasets.py

This script does NOT run inside Claude's sandbox (Kaggle is not reachable
from there) -- run it on your own machine.
"""

import os
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    # kaggle_slug : target_subfolder_name
    "abdallahalidev/plantvillage-dataset": "plantvillage",
    "nirmalsankalana/plantdoc-dataset": "plantdoc",
}


def check_kaggle_credentials():
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        raise SystemExit(
            "No Kaggle credentials found at ~/.kaggle/kaggle.json.\n"
            "Get one from https://www.kaggle.com/settings/account "
            "(Create New Token), save it there, then re-run this script."
        )


def download_and_unzip(slug: str, target_name: str):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    target_dir = RAW_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{target_name}] downloading {slug} ...")
    api.dataset_download_files(slug, path=str(target_dir), unzip=False, quiet=False)

    # unzip whatever .zip landed in target_dir
    for zip_path in target_dir.glob("*.zip"):
        print(f"[{target_name}] extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        zip_path.unlink()  # remove zip after extraction to save space

    print(f"[{target_name}] done -> {target_dir}")


def main():
    check_kaggle_credentials()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for slug, name in DATASETS.items():
        try:
            download_and_unzip(slug, name)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            print(
                f"  -> You can also download manually from "
                f"https://www.kaggle.com/datasets/{slug} and unzip into "
                f"{RAW_DIR / name}"
            )

    print("\nAll done. Next step: python scripts/inspect_raw.py")


if __name__ == "__main__":
    main()
