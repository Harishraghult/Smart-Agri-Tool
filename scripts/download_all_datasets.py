"""
download_all_datasets.py

Unified download script for all datasets required by the Smart Crop Monitoring
& Pest Intelligence System.

DATASETS DOWNLOADED (and why each was chosen)
---------------------------------------------
1.  PlantVillage           — Lab-condition baseline for disease classification.
                             Clean images, 38 classes, large (~54k images).
                             Kaggle: abdallahalidev/plantvillage-dataset

2.  PlantDoc (cropped)     — Field-condition generalization for disease classifier.
                             Real-world lighting/backgrounds. Smaller (~2.5k).
                             GitHub: pratikkayal/PlantDoc-Dataset

3.  PlantDoc (obj detect)  — Object-detection annotations (bounding boxes) for
                             training detection/segmentation models on field images.
                             GitHub: pratikkayal/PlantDoc-Object-Detection-Dataset

4.  New Plant Diseases     — Augmented/pre-split disease data. Overlaps with
                             PlantVillage but adds augmentation variety.
                             Kaggle: vipoooool/new-plant-diseases-dataset

5.  IP102                  — Large-scale pest classification/detection. 102 pest
                             species. Used for pest detector training.
                             GitHub: xpwu95/IP102

6.  Agricultural Pests     — Simpler/smaller pest classification dataset. Used as
                             supplementary pest data or for quick prototyping.
                             Kaggle: vencerlanz09/agricultural-pests-image-dataset

7.  Fruits-360             — 131 fruit/vegetable classes, uniform background.
                             Base classes for ripeness classifier.
                             Kaggle: moltean/fruits

8.  Banana Ripeness        — Banana ripeness stages (unripe/ripe/overripe).
                             Fine-tune ripeness classifier on real staging data.
                             Kaggle: mrmars1010/banana-ripeness-classification-dataset

9.  DeepWeeds              — Rangeland weed species from field photos.
                             Primary weed detection dataset (8 weed species + negative).
                             GitHub: AlexOlsen/DeepWeeds

10. CropAndWeed            — Crop-vs-weed segmentation dataset with pixel masks.
                             Used for weed segmentation training.
                             GitHub: cropandweed/cropandweed-dataset

11. Crop Recommendation    — Tabular dataset: N, P, K, temperature, humidity,
                             pH, rainfall → recommended crop label.
                             Kaggle: atharvaingle/crop-recommendation-dataset

12. Sentinel-2 via EE      — Remote sensing NDVI/NDMI (accessed live via API,
                             not downloaded here).

13. data.gov.in            — Regional crop history for India (manual download,
                             not automated here — requires browsing the portal).

14. WeatherNext 3          — Weather forecasting (API access, not a dataset download).

USAGE
-----
    python scripts/download_all_datasets.py
    python scripts/download_all_datasets.py --output-dir data/raw --skip-existing
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "raw"


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

KAGGLE_DATASETS = [
    {
        "name": "plantvillage",
        "slug": "abdallahalidev/plantvillage-dataset",
        "desc": "PlantVillage — lab-condition disease classification baseline",
    },
    {
        "name": "new_plant_diseases",
        "slug": "vipoooool/new-plant-diseases-dataset",
        "desc": "New Plant Diseases — augmented/pre-split disease data",
    },
    {
        "name": "agricultural_pests",
        "slug": "vencerlanz09/agricultural-pests-image-dataset",
        "desc": "Agricultural Pests — simpler pest classification dataset",
    },
    {
        "name": "fruits360",
        "slug": "moltean/fruits",
        "desc": "Fruits-360 — base fruit classes for ripeness classifier",
    },
    {
        "name": "banana_ripeness",
        "slug": "mrmars1010/banana-ripeness-classification-dataset",
        "desc": "Banana Ripeness — ripeness stage classification",
    },
    {
        "name": "crop_recommendation",
        "slug": "atharvaingle/crop-recommendation-dataset",
        "desc": "Crop Recommendation — tabular N/P/K/weather → crop label",
    },
]

GITHUB_REPOS = [
    {
        "name": "plantdoc_cropped",
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        "desc": "PlantDoc (cropped) — field-condition disease generalization",
    },
    {
        "name": "plantdoc_objdet",
        "url": "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git",
        "desc": "PlantDoc Object Detection — bbox annotations for field images",
    },
    {
        "name": "ip102",
        "url": "https://github.com/xpwu95/IP102.git",
        "desc": "IP102 — large-scale pest classification (102 species)",
    },
    {
        "name": "deepweeds",
        "url": "https://github.com/AlexOlsen/DeepWeeds.git",
        "desc": "DeepWeeds — rangeland weed species detection",
    },
    {
        "name": "cropandweed",
        "url": "https://github.com/cropandweed/cropandweed-dataset.git",
        "desc": "CropAndWeed — crop-vs-weed segmentation with pixel masks",
    },
]


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_kaggle(slug: str, output_dir: Path, name: str) -> bool:
    """Download and extract a Kaggle dataset. Returns True on success."""
    target = output_dir / name
    try:
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", slug, "-p", str(target), "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  ✓ Downloaded {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Kaggle download failed for {name}: {e.stderr.strip()}")
        print(f"    Manual download: https://www.kaggle.com/datasets/{slug}")
        return False
    except FileNotFoundError:
        print(f"  ✗ Kaggle CLI not found. Install: pip install kaggle")
        print(f"    Manual download: https://www.kaggle.com/datasets/{slug}")
        return False


def clone_github(url: str, output_dir: Path, name: str) -> bool:
    """Clone a GitHub repo. Returns True on success."""
    target = output_dir / name
    if target.exists() and any(target.iterdir()):
        print(f"  ⊘ {name} already exists, skipping clone")
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  ✓ Cloned {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Git clone failed for {name}: {e.stderr.strip()}")
        print(f"    Manual clone: git clone {url}")
        return False
    except FileNotFoundError:
        print(f"  ✗ Git not found. Install git, then retry.")
        print(f"    Manual clone: git clone {url}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download all datasets for the Smart Agri-Tool project."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Root directory for raw dataset downloads (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip datasets whose output directory already exists and is non-empty",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownload root: {args.output_dir}\n")

    successes, failures = [], []

    # --- Kaggle datasets ---
    print("=" * 60)
    print("KAGGLE DATASETS")
    print("=" * 60)
    for ds in KAGGLE_DATASETS:
        target = args.output_dir / ds["name"]
        if args.skip_existing and target.exists() and any(target.iterdir()):
            print(f"  ⊘ {ds['name']} already exists, skipping")
            successes.append(ds["name"])
            continue
        print(f"\n  [{ds['name']}] {ds['desc']}")
        ok = download_kaggle(ds["slug"], args.output_dir, ds["name"])
        (successes if ok else failures).append(ds["name"])

    # --- GitHub repos ---
    print("\n" + "=" * 60)
    print("GITHUB REPOSITORIES")
    print("=" * 60)
    for repo in GITHUB_REPOS:
        print(f"\n  [{repo['name']}] {repo['desc']}")
        ok = clone_github(repo["url"], args.output_dir, repo["name"])
        (successes if ok else failures).append(repo["name"])

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Succeeded: {len(successes)}/{len(successes) + len(failures)}")
    if failures:
        print(f"  Failed:    {', '.join(failures)}")
        print("  → Download these manually using the URLs printed above.")
    print(f"\n  Output:    {args.output_dir}")

    # --- Notes on non-downloadable sources ---
    print("\n" + "-" * 60)
    print("NON-DOWNLOADABLE SOURCES (API access or manual):")
    print("-" * 60)
    print("  • Sentinel-2 NDVI   — accessed live via Google Earth Engine API")
    print("  • WeatherNext 3     — accessed live via Google Maps Platform API")
    print("  • data.gov.in       — browse https://data.gov.in for regional crop data")
    print("                        (download manually, place in data/raw/crop_history/)")


if __name__ == "__main__":
    main()
