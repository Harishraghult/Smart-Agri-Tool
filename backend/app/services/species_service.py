"""
Plant species identification with BioCLIP-2 (imageomics/bioclip-2, MIT licence), a CLIP model
trained on 200M organism photos (TreeOfLife-200M). Used zero-shot when the user leaves the crop
on "Auto-detect":

  * a confidently identified supported crop restricts the disease diagnosis to that crop
    (PlantDoc farm test set: 75.2% -> 79.4%)
  * a confidently identified unsupported plant (pear, betel, ...) is reported instead of being
    forced into a wrong disease class (0.5% false "unsupported" rate on supported-crop photos)

Optional: if open_clip is not installed or the weights cannot be loaded, identify() returns None
and diagnosis falls back to the disease models' own crop guess.
"""
import torch
from PIL import Image

from app.config import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "hf-hub:imageomics/bioclip-2"
# Below this, the species guess is not used (measured: 0.9 gave the best diagnosis accuracy)
MIN_CONFIDENCE = 0.9

# Display name -> scientific name. The first 14 are the crops the disease models support
# (keys match diagnosis_service.SUPPORTED_CROPS); the rest are common crops they do not.
SUPPORTED_SPECIES = {
    "Apple": "Malus domestica", "Blueberry": "Vaccinium corymbosum", "Cherry": "Prunus avium",
    "Corn (Maize)": "Zea mays", "Grape": "Vitis vinifera", "Orange": "Citrus sinensis", "Peach": "Prunus persica",
    "Bell Pepper": "Capsicum annuum", "Potato": "Solanum tuberosum", "Raspberry": "Rubus idaeus",
    "Soybean": "Glycine max", "Squash": "Cucurbita pepo", "Strawberry": "Fragaria ananassa",
    "Tomato": "Solanum lycopersicum",
}
UNSUPPORTED_SPECIES = {
    "Pear": "Pyrus communis", "Betel": "Piper betle", "Mango": "Mangifera indica", "Banana": "Musa acuminata",
    "Rice": "Oryza sativa", "Wheat": "Triticum aestivum", "Cassava": "Manihot esculenta",
    "Cotton": "Gossypium hirsutum", "Coffee": "Coffea arabica", "Cucumber": "Cucumis sativus",
    "Eggplant": "Solanum melongena", "Okra": "Abelmoschus esculentus", "Guava": "Psidium guajava",
    "Papaya": "Carica papaya", "Coconut": "Cocos nucifera", "Tea": "Camellia sinensis",
    "Sugarcane": "Saccharum officinarum", "Groundnut": "Arachis hypogaea", "Onion": "Allium cepa",
    "Cabbage": "Brassica oleracea", "Common bean": "Phaseolus vulgaris", "Lemon": "Citrus limon",
    "Plum": "Prunus domestica", "Watermelon": "Citrullus lanatus", "Rose": "Rosa",
}
TEMPLATES = ["a photo of {sci}.", "a photo of {common} ({sci}) leaves.", "a photo of a {common} plant."]


class SpeciesService:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.names = list(SUPPORTED_SPECIES) + list(UNSUPPORTED_SPECIES)
        if settings.SPECIES_ID_ENABLED:
            self._load()

    def _load(self):
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
            tokenizer = open_clip.get_tokenizer(MODEL_ID)
            model = model.to(DEVICE).eval()
            species = {**SUPPORTED_SPECIES, **UNSUPPORTED_SPECIES}
            with torch.no_grad():
                feats = []
                for common, sci in species.items():
                    tokens = tokenizer([t.format(common=common.lower(), sci=sci) for t in TEMPLATES]).to(DEVICE)
                    f = model.encode_text(tokens)
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.mean(0))
                feats = torch.stack(feats)
            self.text_features = feats / feats.norm(dim=-1, keepdim=True)
            self.model, self.preprocess = model, preprocess
            print("Loaded BioCLIP-2 species identifier")
        except Exception as e:
            print(f"Species identifier unavailable ({e}); crop auto-detect uses the disease models only")

    def identify(self, image: Image.Image):
        """Return {"species", "scientific_name", "confidence", "supported", "confident"} or None."""
        if self.model is None:
            return None
        with torch.no_grad():
            f = self.model.encode_image(self.preprocess(image).unsqueeze(0).to(DEVICE))
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ self.text_features.T).softmax(dim=-1)[0]
        conf, idx = probs.max(dim=0)
        name = self.names[idx.item()]
        return {
            "species": name,
            "scientific_name": SUPPORTED_SPECIES.get(name) or UNSUPPORTED_SPECIES.get(name),
            "confidence": round(float(conf), 3),
            "supported": name in SUPPORTED_SPECIES,
            "confident": float(conf) >= MIN_CONFIDENCE,
        }


species_service = SpeciesService()
