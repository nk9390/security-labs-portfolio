import io
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    return np.array(Image.open(buf))


def gaussian_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.round(img + rng.normal(0, sigma, img.shape)), 0, 255).astype(np.uint8)


def resize_roundtrip(img, scale=0.5):
    h, w = img.shape
    small = Image.fromarray(img).resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return np.array(small.resize((w, h), Image.BICUBIC))


def median3(img):
    return median_filter(img, size=3)


def crop_blank(img, frac=0.25):
    """Blank out a centred square covering `frac` of the image area."""
    out = img.copy()
    h, w = img.shape
    s = int(np.sqrt(frac) * h)
    t, l = (h - s) // 2, (w - s) // 2
    out[t:t + s, l:l + s] = 0
    return out


ATTACKS = {
    "none": lambda x: x,
    "JPEG q=90": lambda x: jpeg(x, 90),
    "JPEG q=70": lambda x: jpeg(x, 70),
    "JPEG q=50": lambda x: jpeg(x, 50),
    "Gaussian noise s=2": lambda x: gaussian_noise(x, 2),
    "Gaussian noise s=5": lambda x: gaussian_noise(x, 5),
    "Resize 0.5x": resize_roundtrip,
    "Median 3x3": median3,
    "Blank 25% area": crop_blank,
}
