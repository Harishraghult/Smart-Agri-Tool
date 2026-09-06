"""
generate_crop_recommendation_csv.py

Generates the benchmark 2,200-row Crop Recommendation tabular dataset.
Covers 22 crop classes across 7 features: N, P, K, temperature, humidity, ph, rainfall.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "crop_recommendation"

np.random.seed(42)

CROPS_PROFILES = {
    "rice": {"N": (60, 100), "P": (35, 60), "K": (35, 45), "temp": (20, 27), "hum": (80, 85), "ph": (6.0, 7.8), "rain": (180, 300)},
    "maize": {"N": (60, 100), "P": (35, 60), "K": (15, 25), "temp": (18, 27), "hum": (55, 75), "ph": (5.5, 7.0), "rain": (60, 110)},
    "chickpea": {"N": (20, 50), "P": (55, 80), "K": (75, 85), "temp": (17, 21), "hum": (14, 20), "ph": (5.9, 8.8), "rain": (65, 95)},
    "kidneybeans": {"N": (15, 40), "P": (55, 80), "K": (15, 25), "temp": (15, 25), "hum": (20, 25), "ph": (5.5, 5.9), "rain": (60, 150)},
    "pigeonpeas": {"N": (15, 40), "P": (55, 80), "K": (18, 25), "temp": (27, 37), "hum": (30, 65), "ph": (4.5, 7.5), "rain": (90, 195)},
    "mothbeans": {"N": (0, 40), "P": (35, 60), "K": (15, 25), "temp": (24, 32), "hum": (40, 65), "ph": (3.5, 10.0), "rain": (30, 75)},
    "mungbean": {"N": (0, 40), "P": (35, 60), "K": (15, 25), "temp": (27, 30), "hum": (80, 90), "ph": (6.2, 7.2), "rain": (35, 60)},
    "blackgram": {"N": (40, 60), "P": (55, 80), "K": (15, 25), "temp": (25, 35), "hum": (60, 70), "ph": (6.5, 7.5), "rain": (60, 75)},
    "lentil": {"N": (15, 40), "P": (55, 80), "K": (15, 25), "temp": (18, 30), "hum": (60, 70), "ph": (5.9, 7.8), "rain": (35, 55)},
    "pomegranate": {"N": (0, 40), "P": (10, 30), "K": (35, 45), "temp": (18, 25), "hum": (85, 95), "ph": (5.5, 7.2), "rain": (100, 115)},
    "banana": {"N": (80, 120), "P": (70, 95), "K": (45, 55), "temp": (25, 30), "hum": (75, 85), "ph": (5.5, 6.8), "rain": (90, 120)},
    "mango": {"N": (0, 40), "P": (15, 40), "K": (25, 35), "temp": (27, 36), "hum": (45, 55), "ph": (4.5, 7.0), "rain": (85, 100)},
    "grapes": {"N": (10, 40), "P": (120, 145), "K": (195, 205), "temp": (8, 42), "hum": (80, 85), "ph": (5.5, 6.5), "rain": (60, 75)},
    "watermelon": {"N": (80, 120), "P": (5, 30), "K": (45, 55), "temp": (24, 27), "hum": (80, 90), "ph": (6.0, 6.8), "rain": (40, 60)},
    "muskmelon": {"N": (80, 120), "P": (5, 30), "K": (45, 55), "temp": (27, 30), "hum": (90, 95), "ph": (6.0, 6.8), "rain": (20, 30)},
    "apple": {"N": (0, 40), "P": (120, 145), "K": (195, 205), "temp": (21, 24), "hum": (90, 95), "ph": (5.5, 6.5), "rain": (100, 125)},
    "orange": {"N": (0, 40), "P": (5, 30), "K": (5, 15), "temp": (10, 35), "hum": (90, 95), "ph": (6.0, 7.8), "rain": (100, 120)},
    "papaya": {"N": (30, 70), "P": (45, 70), "K": (45, 55), "temp": (23, 44), "hum": (90, 95), "ph": (6.5, 7.0), "rain": (40, 250)},
    "coconut": {"N": (17, 40), "P": (5, 30), "K": (25, 35), "temp": (25, 28), "hum": (90, 99), "ph": (5.5, 6.5), "rain": (130, 225)},
    "cotton": {"N": (110, 140), "P": (35, 60), "K": (15, 25), "temp": (22, 26), "hum": (75, 85), "ph": (6.0, 8.0), "rain": (60, 90)},
    "jute": {"N": (60, 100), "P": (35, 60), "K": (35, 45), "temp": (23, 26), "hum": (70, 80), "ph": (6.0, 7.5), "rain": (150, 200)},
    "coffee": {"N": (80, 120), "P": (15, 40), "K": (25, 35), "temp": (23, 28), "hum": (50, 70), "ph": (6.0, 7.2), "rain": (115, 199)}
}

def generate_data(samples_per_crop=100):
    rows = []
    for crop, prof in CROPS_PROFILES.items():
        for _ in range(samples_per_crop):
            n = round(float(np.random.uniform(*prof["N"])), 2)
            p = round(float(np.random.uniform(*prof["P"])), 2)
            k = round(float(np.random.uniform(*prof["K"])), 2)
            temp = round(float(np.random.uniform(*prof["temp"])), 2)
            hum = round(float(np.random.uniform(*prof["hum"])), 2)
            ph = round(float(np.random.uniform(*prof["ph"])), 2)
            rain = round(float(np.random.uniform(*prof["rain"])), 2)
            rows.append({
                "N": n, "P": p, "K": k,
                "temperature": temp, "humidity": hum, "ph": ph,
                "rainfall": rain, "label": crop
            })
    return pd.DataFrame(rows)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_data(100)
    csv_path = OUTPUT_DIR / "Crop_recommendation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} rows across {df['label'].nunique()} crop classes.")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
