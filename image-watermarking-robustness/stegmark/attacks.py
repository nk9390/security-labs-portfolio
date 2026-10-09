"""Image attacks for robustness testing (work on grayscale, RGB and RGBA uint8)."""
import io

import numpy as np
from PIL import Image, ImageFilter


def _pil(img):
    return Image.fromarray(img)


def jpeg(img, q):
    im = _pil(img)
    alpha = None
    if im.mode == "RGBA":
        alpha = img[..., 3:]
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)          # Pillow default: 4:2:0 chroma subsampling
    out = np.array(Image.open(buf))
    return np.concatenate([out, alpha], axis=2) if alpha is not None else out


def noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    out = img.astype(np.float64)
    sl = (Ellipsis, slice(0, 3)) if img.ndim == 3 else Ellipsis
    out[sl] += rng.normal(0, sigma, out[sl].shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def resize_roundtrip(img, scale=0.5):
    h, w = img.shape[:2]
    small = _pil(img).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return np.array(small.resize((w, h), Image.BICUBIC))


def median3(img):
    return np.array(_pil(img).filter(ImageFilter.MedianFilter(3)))


def blur(img, radius=1.0):
    return np.array(_pil(img).filter(ImageFilter.GaussianBlur(radius)))


def brightness(img, delta=20):
    out = img.astype(np.int16)
    if img.ndim == 3:
        out[..., :3] += delta
    else:
        out += delta
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast(img, factor=0.9):
    out = img.astype(np.float64)
    sl = (Ellipsis, slice(0, 3)) if img.ndim == 3 else Ellipsis
    out[sl] = (out[sl] - 128) * factor + 128
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def blank(img, frac=0.25):
    out = img.copy()
    h, w = img.shape[:2]
    s = int(np.sqrt(frac) * min(h, w))
    t, l = (h - s) // 2, (w - s) // 2
    out[t:t + s, l:l + s, ...] = 0
    return out


def crop_border(img, frac=0.05):
    h, w = img.shape[:2]
    dy, dx = int(h * frac), int(w * frac)
    return img[dy:h - dy, dx:w - dx].copy()


def social_media(img):
    """A typical share pipeline: downscale to 75%, upscale back, JPEG q=75."""
    return jpeg(resize_roundtrip(img, 0.75), 75)


ATTACKS = {
    "none": lambda x: x,
    "JPEG q=90": lambda x: jpeg(x, 90),
    "JPEG q=75": lambda x: jpeg(x, 75),
    "JPEG q=50": lambda x: jpeg(x, 50),
    "Noise sigma=3": lambda x: noise(x, 3),
    "Noise sigma=8": lambda x: noise(x, 8),
    "Resize 0.5x round trip": resize_roundtrip,
    "Median 3x3": median3,
    "Gaussian blur r=1": blur,
    "Brightness +20": brightness,
    "Contrast x0.9": contrast,
    "Blank 25% area": blank,
    "Resize 0.75x + JPEG 75": social_media,
    "Crop 5% border": crop_border,
}
