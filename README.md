# Dataset pipeline: PlantVillage + PlantDoc, leakage-safe split

This folder downloads both datasets and produces a train/val/test split
where near-duplicate images (common in PlantVillage especially) are never
allowed to appear on both sides of a split — which is the leakage problem
flagged in your project's own "Known Limitations" section.

## Folder layout

```
crop_project/
  requirements.txt
  scripts/
    download_datasets.py   # pulls PlantVillage + PlantDoc from Kaggle
    inspect_raw.py          # sanity-check class counts before splitting
    dedupe_split.py         # the leakage-safe split itself
  data/
    raw/            # untouched downloads land here
    processed/       # train/val/test folders + split_manifest.csv land here
```

## Steps

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Get a Kaggle API token** (one-time)
   - Go to https://www.kaggle.com/settings/account → "Create New Token"
   - This downloads `kaggle.json`
   - Move it to `~/.kaggle/kaggle.json` (Mac/Linux) or
     `C:\Users\<you>\.kaggle\kaggle.json` (Windows)
   - Linux/Mac only: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the raw datasets**
   ```
   python scripts/download_datasets.py
   ```
   If a Kaggle download fails (rate limits, dataset renamed, etc.), the
   script prints the manual-download URL — grab the zip yourself and
   unzip it into `data/raw/plantvillage/` or `data/raw/plantdoc/`.

4. **Check what you got before splitting**
   ```
   python scripts/inspect_raw.py
   ```
   This shows image counts per class. Look for classes with very few
   images — those may not split 3 ways cleanly and are worth flagging in
   your report as a known data limitation.

5. **Run the leakage-safe split**
   ```
   python scripts/dedupe_split.py --dataset plantvillage
   python scripts/dedupe_split.py --dataset plantdoc
   ```
   This writes `data/processed/<dataset>/{train,val,test}/<class>/*.jpg`
   plus a `split_manifest.csv` recording exactly which cluster each image
   belonged to and which split it landed in.

   If you eyeball the manifest and think too many/few images are being
   grouped as duplicates, re-run with a different threshold:
   ```
   python scripts/dedupe_split.py --dataset plantvillage --hash-threshold 4
   ```
   Lower = stricter (fewer things count as duplicates). Higher = looser.

## Why this matters for your report

Keep `split_manifest.csv` for both datasets. In your methodology section
you can now honestly state: *"Train/val/test splits were performed at the
deduplicated-image-cluster level using perceptual hashing (pHash,
threshold=8), not randomly at the image level, to prevent near-duplicate
leaf photos from the same source image inflating validation accuracy."*
That's a stronger, more specific claim than most student projects make,
and it's exactly the kind of methodology detail that was flagged as a
best practice in your earlier project review.

## Next steps once data is split

- Fine-tune on PlantVillage first (clean signal), then fine-tune again on
  PlantDoc (field-realistic images) rather than training on a naive merge
  of both — this mirrors the "lab-then-field" mitigation your doc already
  calls for.
- Build the `cause_category` lookup table (pathogen/pest/deficiency) by
  hand-mapping each class label — this is a design decision to document,
  not something the data gives you for free.
