"""High-level API: hide text in an image file and read it back."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .core import (DOMAINS, EmbedInfo, Extracted, StegError, capacity, embed_payload,
                   extract_payload)

SALT, NONCE = 16, 12
CRYPTO_OVERHEAD = SALT + NONCE + 16          # salt + nonce + GCM tag


# ----------------------------------------------------------------- crypto
def _key(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    return Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(password.encode())


def encrypt(data: bytes, password: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce = os.urandom(SALT), os.urandom(NONCE)
    return salt + nonce + AESGCM(_key(password, salt)).encrypt(nonce, data, b"stegmark")


def decrypt(blob: bytes, password: str) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce, ct = blob[:SALT], blob[SALT:SALT + NONCE], blob[SALT + NONCE:]
    try:
        return AESGCM(_key(password, salt)).decrypt(nonce, ct, b"stegmark")
    except InvalidTag:
        raise StegError("wrong password (decryption failed)")


# ----------------------------------------------------------------- image I/O
def load_image(path: str) -> np.ndarray:
    """Load as uint8 array: (H,W) grayscale, (H,W,3) RGB or (H,W,4) RGBA."""
    im = Image.open(path)
    im.load()
    if im.mode in ("L", "RGB", "RGBA"):
        pass
    elif im.mode in ("LA",):
        im = im.convert("RGBA")
    elif im.mode in ("I;16", "I", "F"):
        im = im.convert("L")
    elif "A" in im.getbands() or im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB")
    return np.array(im)


def save_image(arr: np.ndarray, path: str, quality: int = 95) -> None:
    ext = os.path.splitext(path)[1].lower()
    im = Image.fromarray(arr)
    if ext in (".jpg", ".jpeg"):
        if im.mode == "RGBA":
            im = im.convert("RGB")
        im.save(path, quality=quality, subsampling=0)
    else:
        im.save(path)


# ----------------------------------------------------------------- metrics
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


# ----------------------------------------------------------------- hide / reveal
def hide(img: np.ndarray, message: str | bytes, method: str = "DWT", password: str = "",
         strength: str = "medium") -> tuple[np.ndarray, EmbedInfo]:
    method = method.upper()
    if method not in DOMAINS:
        raise StegError(f"unknown method {method!r}; choose LSB, DCT or DWT")
    data = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if password:
        data = encrypt(data, password)
    return embed_payload(img, data, method, password=password, strength=strength,
                         encrypted=bool(password))


@dataclass
class Revealed:
    found: bool
    method: str | None = None
    strength: str | None = None
    data: bytes | None = None
    encrypted: bool = False
    damaged: bool = False
    body_ber: float | None = None

    @property
    def text(self) -> str | None:
        if self.data is None:
            return None
        try:
            return self.data.decode("utf-8")
        except UnicodeDecodeError:
            return None


def reveal(img: np.ndarray, password: str = "", method: str | None = None) -> Revealed:
    methods = (method.upper(),) if method else ("LSB", "DCT", "DWT")
    ex: Extracted | None = extract_payload(img, password=password, methods=methods)
    if ex is None:
        return Revealed(found=False)
    if ex.payload is None:
        return Revealed(True, ex.method, ex.strength, None, ex.encrypted, damaged=True)
    data = ex.payload
    if ex.encrypted:
        if not password:
            raise StegError("this watermark is encrypted; supply the password")
        data = decrypt(data, password)
    return Revealed(True, ex.method, ex.strength, data, ex.encrypted, False, ex.body_ber)


def max_message_bytes(img: np.ndarray, method: str, password: str = "") -> int:
    c = capacity(img.shape, method.upper())
    return max(0, c - (CRYPTO_OVERHEAD if password else 0))
