"""Keyed image watermarking in the spatial (LSB), DCT and DWT domains.

All three methods embed the same payload with the same redundancy so they can
be compared fairly:
  message (32 bytes) -> [optional Reed-Solomon] -> bits -> repeated to fill
  `n_slots` embedding positions chosen by a secret key.
Extraction combines the repeats by majority vote (robust to a blanked region), then optionally RS-decodes.
"""
import numpy as np
import pywt
from reedsolo import RSCodec, ReedSolomonError
from scipy.fft import dctn, idctn

N_SLOTS = 4096          # embedding positions per image (matches 512x512 / 8x8 blocks)
RS_SYMBOLS = 32         # parity bytes -> corrects up to 16 corrupted bytes
MSG_LEN = 32            # message bytes
COEF = (1, 2)           # low-mid frequency DCT coefficient per block (survives JPEG better than higher ones)


# ---------- payload ----------
def pad_message(msg: bytes) -> bytes:
    if len(msg) > MSG_LEN:
        raise ValueError(f"message must be <= {MSG_LEN} bytes")
    return msg.ljust(MSG_LEN, b"\x00")


def encode_payload(msg: bytes, use_rs: bool) -> np.ndarray:
    data = pad_message(msg)
    if use_rs:
        data = bytes(RSCodec(RS_SYMBOLS).encode(data))
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def decode_payload(bits: np.ndarray, use_rs: bool):
    """Return the recovered message bytes, or None if RS decoding fails."""
    data = np.packbits(bits.astype(np.uint8)).tobytes()
    if not use_rs:
        return data
    try:
        out = RSCodec(RS_SYMBOLS).decode(data)[0]
        return bytes(out)
    except ReedSolomonError:
        return None


def _tile(bits: np.ndarray, n: int) -> np.ndarray:
    reps = int(np.ceil(n / len(bits)))
    return np.tile(bits, reps)[:n]


def _combine(values: np.ndarray, n_bits: int) -> np.ndarray:
    """Average the repeated copies of each payload bit. values has N_SLOTS entries."""
    reps = int(np.ceil(len(values) / n_bits))
    padded = np.full(reps * n_bits, np.nan)
    padded[: len(values)] = values
    return np.nanmean(padded.reshape(reps, n_bits), axis=0)


def _positions(key: int, population: int, n: int) -> np.ndarray:
    return np.random.default_rng(key).permutation(population)[:n]


# ---------- quantisation index modulation (QIM) ----------
def _qim_embed(c: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
    off = bits * step / 2.0
    return np.round((c - off) / step) * step + off


def _qim_score(c: np.ndarray, step: float) -> np.ndarray:
    """Positive => bit 0 is closer, negative => bit 1 is closer (in [-step/4, step/4])."""
    d0 = np.abs(c - np.round(c / step) * step)
    d1 = np.abs(c - (np.round((c - step / 2) / step) * step + step / 2))
    return d1 - d0


# ---------- LSB ----------
def lsb_embed(img: np.ndarray, bits: np.ndarray, key: int) -> np.ndarray:
    flat = img.copy().ravel()
    pos = _positions(key, flat.size, N_SLOTS)
    flat[pos] = (flat[pos] & 0xFE) | _tile(bits, N_SLOTS).astype(np.uint8)
    return flat.reshape(img.shape)


def lsb_extract(img: np.ndarray, n_bits: int, key: int) -> np.ndarray:
    pos = _positions(key, img.size, N_SLOTS)
    vals = (img.ravel()[pos] & 1).astype(float)
    return (_combine(vals, n_bits) > 0.5).astype(np.uint8)


# ---------- DCT (8x8 blocks, one mid-frequency coefficient per block) ----------
def _blocks(img):
    h, w = img.shape
    return img.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)  # (bh, bw, 8, 8)


def _unblocks(b):
    bh, bw = b.shape[:2]
    return b.transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8)


def dct_embed(img: np.ndarray, bits: np.ndarray, key: int, step: float = 32.0) -> np.ndarray:
    b = dctn(_blocks(img.astype(float)), axes=(2, 3), norm="ortho")
    flat = b[:, :, COEF[0], COEF[1]].ravel()
    pos = _positions(key, flat.size, N_SLOTS)
    flat[pos] = _qim_embed(flat[pos], _tile(bits, N_SLOTS), step)
    b[:, :, COEF[0], COEF[1]] = flat.reshape(b.shape[:2])
    out = _unblocks(idctn(b, axes=(2, 3), norm="ortho"))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def dct_extract(img: np.ndarray, n_bits: int, key: int, step: float = 32.0) -> np.ndarray:
    b = dctn(_blocks(img.astype(float)), axes=(2, 3), norm="ortho")
    flat = b[:, :, COEF[0], COEF[1]].ravel()
    pos = _positions(key, flat.size, N_SLOTS)
    return (_combine(np.sign(_qim_score(flat[pos], step)), n_bits) < 0).astype(np.uint8)


# ---------- DWT (2-level Haar, HL subband) ----------
def dwt_embed(img: np.ndarray, bits: np.ndarray, key: int, step: float = 40.0) -> np.ndarray:
    coeffs = pywt.wavedec2(img.astype(float), "haar", level=2)
    hl = coeffs[1][0].copy()                      # level-2 horizontal detail
    flat = hl.ravel()
    pos = _positions(key, flat.size, N_SLOTS)
    flat[pos] = _qim_embed(flat[pos], _tile(bits, N_SLOTS), step)
    coeffs[1] = (flat.reshape(hl.shape),) + tuple(coeffs[1][1:])
    out = pywt.waverec2(coeffs, "haar")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def dwt_extract(img: np.ndarray, n_bits: int, key: int, step: float = 40.0) -> np.ndarray:
    coeffs = pywt.wavedec2(img.astype(float), "haar", level=2)
    flat = coeffs[1][0].ravel()
    pos = _positions(key, flat.size, N_SLOTS)
    return (_combine(np.sign(_qim_score(flat[pos], step)), n_bits) < 0).astype(np.uint8)


METHODS = {
    "LSB": (lsb_embed, lsb_extract),
    "DCT": (dct_embed, dct_extract),
    "DWT": (dwt_embed, dwt_extract),
}
