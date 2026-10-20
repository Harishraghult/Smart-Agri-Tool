import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # matches the UI's "Max 15MB"
MIN_SIDE_PX = 32
MAX_ASPECT_RATIO = 8.0
# Grey-level standard deviation below this means a blank / single-colour frame
# (JPEG noise on a solid fill is ~0-1; a photo shot in very low light is still ~4+)
MIN_PIXEL_STD = 2.0
# Huge uploads are downscaled first: models see 224px anyway, and the OpenCV heuristics get faster
MAX_WORKING_SIDE_PX = 2048
# Low-light / haze correction triggers. Measured on ~2,000 real test photos (PlantVillage, PlantDoc,
# bananas, pests, DeepWeeds): normal photos have 99th-percentile brightness >= 117 and a p1-p99 range
# >= 76, so these thresholds never fire on them. Autocontrast lifted dark-photo disease accuracy
# from 10% to 90%, but costs ~9 points on normal photos, so it is applied only when detected.
# Colour casts and overexposure are NOT corrected: they could not be told apart from normal photos.
DARK_P99_THRESHOLD = 60
LOW_CONTRAST_RANGE_THRESHOLD = 60


class ImageValidationError(ValueError):
    """Upload is not a usable photo; message is safe to show to the user."""


def _correct_exposure(image: Image.Image):
    """Autocontrast only when the photo is detectably dark or hazy. Returns (image, step or None)."""
    grey = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    p1, p99 = np.percentile(grey, [1, 99])
    if p99 < DARK_P99_THRESHOLD:
        return ImageOps.autocontrast(image, cutoff=1), "low_light_correction"
    if p99 - p1 < LOW_CONTRAST_RANGE_THRESHOLD:
        return ImageOps.autocontrast(image, cutoff=1), "low_contrast_correction"
    return image, None


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    """Decode an upload into an upright RGB image, rejecting files no model can judge.

    Steps applied (listed in image.info["preprocessing"]): EXIF orientation, transparency ->
    white, 16-bit -> 8-bit, any mode -> RGB, downscale if huge, and low-light / low-contrast
    correction only when detected.
    """
    steps = []
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise ImageValidationError("File is not a readable image (corrupt, truncated, or unsupported format).") from e

    # Phone photos store rotation in EXIF; apply it so the model sees the leaf upright
    orientation = image.getexif().get(0x0112, 1)
    image = ImageOps.exif_transpose(image)
    if orientation not in (1, None):
        steps.append("exif_orientation")

    if image.mode in ("RGBA", "LA", "P"):
        # Composite transparency on white instead of letting it turn black
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
        steps.append("transparency_to_white")
    elif image.mode.startswith("I;16") or image.mode == "I":
        # 16-bit greyscale: scale to 8-bit before RGB conversion
        arr = np.asarray(image, dtype=np.float32)
        span = max(float(arr.max() - arr.min()), 1.0)
        image = Image.fromarray(((arr - arr.min()) / span * 255).astype(np.uint8))
        steps.append("16bit_to_8bit")
    if image.mode != "RGB":
        steps.append(f"{image.mode.lower()}_to_rgb")
        image = image.convert("RGB")

    w, h = image.size
    if min(w, h) < MIN_SIDE_PX:
        raise ImageValidationError(f"Image is too small ({w}x{h}); use a photo at least {MIN_SIDE_PX}px on each side.")
    if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
        raise ImageValidationError(f"Image is an extreme strip ({w}x{h}); crop it closer to a normal photo shape.")

    grey = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    if grey.std() < MIN_PIXEL_STD:
        raise ImageValidationError("Image is blank or a single colour; upload a real photo.")

    if max(w, h) > MAX_WORKING_SIDE_PX:
        image.thumbnail((MAX_WORKING_SIDE_PX, MAX_WORKING_SIDE_PX), Image.LANCZOS)
        steps.append(f"downscaled_from_{w}x{h}")

    image, exposure_step = _correct_exposure(image)
    if exposure_step:
        steps.append(exposure_step)

    image.info["preprocessing"] = steps
    return image
