"""Embedding engine: hide / recover a byte payload in an image with LSB, DCT or DWT.

Works on real images: any size, grayscale / RGB / RGBA, and decodes from the
image alone (no message length, filename or method needs to be remembered).

Layout of what is hidden
------------------------
    header (8 bytes)  = b"SM" | version+flags | method id | RS parity count | payload length (uint32)
    header -> Reed-Solomon(8 parity bytes) -> 128 bits, repeated `rh` times
    payload           = UTF-8 text, or AES-GCM ciphertext when a password is given
    payload -> Reed-Solomon(nsym parity bytes per 255-byte block) -> bits, repeated `rb` times

Slots are visited in a pseudo-random order derived from the password (or a
public default key), so the hidden bits are spread across the whole image and a
damaged region only removes some of the copies of each bit.

Domains
-------
* LSB : least significant bit of R, G, B values (alpha untouched). Highest
        capacity, invisible, but any re-encoding destroys it.
* DCT : dithered quantisation index modulation (QIM) on DCT coefficient (1,2) of every
        8x8 luma block. Survives JPEG, noise, resizing.
* DWT : QIM on the level-2 Haar horizontal-detail coefficient of keyed 4x4
        luma blocks (one in four blocks). Same robustness family as DCT.

DCT/DWT change only luma (the same offset is added to R, G and B), so colours
are not tinted. Embedding is iterated so that rounding to 8-bit and clipping at
0/255 (e.g. pure black or white areas) do not corrupt the embedded bits.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import pywt
from reedsolo import RSCodec, ReedSolomonError
from scipy.fft import idctn

MAGIC = b"SM"
VERSION = 1
HEADER_BYTES = 8
HEADER_RS = 8
HEADER_BITS = (HEADER_BYTES + HEADER_RS) * 8          # 128
METHOD_IDS = {"LSB": 0, "DCT": 1, "DWT": 2}
METHOD_NAMES = {v: k for k, v in METHOD_IDS.items()}

# QIM step sizes per strength preset. "medium" equals the values validated in
# the robustness study (src/ + results/): DCT 32, DWT 40.
STEPS = {
    "DCT": {"low": 16.0, "medium": 32.0, "high": 48.0},
    "DWT": {"low": 20.0, "medium": 40.0, "high": 60.0},
}
DEFAULT_NSYM = {"LSB": 8, "DCT": 32, "DWT": 32}
NSYM_CHOICES = (32, 24, 16, 12, 8)     # robust methods: most parity that still fits with >= 3 copies
MIN_ROBUST_REPEATS = 3


class StegError(Exception):
    """Raised for user-facing problems (message too long, image too small...)."""


# ----------------------------------------------------------------- helpers
def _seed(password: str, method: str) -> int:
    h = hashlib.sha256(f"stegmark|{method}|{password}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _rs_len(n: int, nsym: int) -> int:
    """Length of reedsolo output for an n-byte message (it chunks at 255)."""
    k = 255 - nsym
    return n + nsym * (-(-n // k)) if n else nsym


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def luma(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma as float. img is (H,W) or (H,W,C>=3) uint8."""
    if img.ndim == 2:
        return img.astype(np.float64)
    rgb = img[..., :3].astype(np.float64)
    return rgb @ np.array([0.299, 0.587, 0.114])


def _add_luma(img: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Add a luma offset field by shifting R, G and B equally (chroma unchanged)."""
    out = img.astype(np.float64)
    if img.ndim == 2:
        out += d
    else:
        out[..., :3] += d[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- QIM
def qim_embed(c: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
    off = bits * (step / 2.0)
    return np.round((c - off) / step) * step + off


def qim_score(c: np.ndarray, step: float) -> np.ndarray:
    """> 0 means bit 0 is closer, < 0 means bit 1; magnitude in [0, step/4]."""
    d0 = np.abs(c - np.round(c / step) * step)
    d1 = np.abs(c - (np.round((c - step / 2) / step) * step + step / 2))
    return d1 - d0


# ----------------------------------------------------------------- domains
class LSBDomain:
    name = "LSB"
    robust = False

    def n_slots(self, shape) -> int:
        h, w = shape[:2]
        return h * w * (1 if len(shape) == 2 else 3)

    def _flat_index(self, shape, pos):
        if len(shape) == 2:
            return pos
        c_total = shape[2]
        pix, ch = pos // 3, pos % 3
        return pix * c_total + ch

    def read(self, img, pos, step=None, dither=None):
        v = img.reshape(-1)[self._flat_index(img.shape, pos)] & 1
        return 1.0 - 2.0 * v                      # +1 => bit 0, -1 => bit 1

    def write(self, img, pos, bits, step=None, dither=None):
        out = img.copy()
        flat = out.reshape(-1)
        idx = self._flat_index(img.shape, pos)
        flat[idx] = (flat[idx] & 0xFE) | bits.astype(np.uint8)
        return out


class BlockQIMDomain:
    """QIM on one linear coefficient per BxB luma block.

    The coefficient is <block, pattern>; for the DCT the pattern is the (1,2)
    basis function, for the DWT it is the level-2 Haar horizontal-detail
    footprint (Haar level-2 coefficients depend only on one 4x4 block, so this
    is exactly pywt.wavedec2(..., 'haar', level=2)[1][0]; see the tests).
    """
    robust = True

    def __init__(self, name, block, pattern, density):
        self.name, self.B, self.pattern, self.density = name, block, pattern, density

    def _grid(self, shape):
        return shape[0] // self.B, shape[1] // self.B

    def n_slots(self, shape) -> int:
        gh, gw = self._grid(shape)
        return (gh * gw) // self.density

    def n_blocks(self, shape) -> int:
        gh, gw = self._grid(shape)
        return gh * gw

    def coeffs(self, Y: np.ndarray) -> np.ndarray:
        gh, gw = self._grid(Y.shape)
        B = self.B
        blocks = Y[: gh * B, : gw * B].reshape(gh, B, gw, B)
        return np.einsum("ibjc,bc->ij", blocks, self.pattern).ravel()

    def delta_field(self, shape, block_ids, delta):
        gh, gw = self._grid(shape)
        B = self.B
        g = np.zeros(gh * gw)
        g[block_ids] = delta
        field = np.zeros(shape[:2])
        field[: gh * B, : gw * B] = (g.reshape(gh, 1, gw, 1) * self.pattern[None, :, None, :]).reshape(gh * B, gw * B)
        return field

    def read(self, img, pos, step, dither=None):
        ids = pos                                   # slot positions are block ids
        dz = 0.0 if dither is None else dither * step
        return qim_score(self.coeffs(luma(img))[ids] - dz, step)

    def write(self, img, pos, bits, step, dither=None, iters=10):
        ids = pos
        dz = 0.0 if dither is None else dither * step
        cur = img
        for _ in range(iters):
            c = self.coeffs(luma(cur))[ids]
            target = qim_embed(c - dz, bits, step) + dz
            delta = target - c
            if np.max(np.abs(delta)) < 0.02 * step:
                break
            cur = _add_luma(cur, self.delta_field(img.shape, ids, delta))
        return cur


def _dct_pattern():
    e = np.zeros((8, 8)); e[1, 2] = 1.0
    return idctn(e, norm="ortho")


def _haar_pattern():
    """Recover the 4x4 footprint of a level-2 Haar cH coefficient from pywt."""
    z = [np.zeros((1, 1)), (np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
         (np.zeros((2, 2)),) * 3]
    return pywt.waverec2(z, "haar")           # orthonormal => synthesis == analysis pattern


DOMAINS = {
    "LSB": LSBDomain(),
    "DCT": BlockQIMDomain("DCT", 8, _dct_pattern(), density=1),
    "DWT": BlockQIMDomain("DWT", 4, _haar_pattern(), density=4),
}


# ----------------------------------------------------------------- layout
def _positions(domain, shape, password, count):
    n = domain.n_slots(shape)
    rng = np.random.default_rng(_seed(password, domain.name))
    if domain.name == "LSB":                 # permute pixels, then use their R,G,B in turn
        npix = n if len(shape) == 2 else n // 3
        per = 1 if len(shape) == 2 else 3
        need = -(-count // per)
        perm = rng.permutation(npix).astype(np.int64)[:need]
        if per == 1:
            return perm[:count]
        return (perm[:, None] * 3 + np.arange(3)[None, :]).ravel()[:count]
    # block domains: a keyed subset of all blocks (DWT uses 1 block in 4)
    return rng.permutation(domain.n_blocks(shape))[:count]


def _dither(domain, shape, password, pos):
    """Keyed dither in [0, 1) per block (dithered QIM): the lattice is secret and an
    erased (e.g. blanked) block reads as noise instead of a confident bit 0."""
    if not domain.robust:
        return None
    rng = np.random.default_rng(_seed(password, domain.name + "|dither"))
    return rng.random(domain.n_blocks(shape))[pos]


def header_repeats(domain, shape) -> int:
    n = domain.n_slots(shape)
    if not domain.robust:
        return 1
    return int(np.clip(n // (4 * HEADER_BITS), 1, 15))    # header uses <= 1/4 of the slots


@dataclass
class EmbedInfo:
    method: str
    slots: int
    header_repeats: int
    body_bits: int
    body_repeats: int
    payload_bytes: int
    encrypted: bool
    unreliable_slots: int


def capacity(shape, method: str, nsym: int | None = None) -> int:
    """Largest payload (bytes, after encryption) that fits with the method's
    minimum repetition (1 for LSB, 3 for DCT/DWT)."""
    d = DOMAINS[method]
    if nsym is None:
        nsym = min(NSYM_CHOICES) if d.robust else DEFAULT_NSYM[method]
    n = d.n_slots(shape)
    free = n - HEADER_BITS * header_repeats(d, shape)
    min_r = MIN_ROBUST_REPEATS if d.robust else 1
    budget = free // (8 * min_r)
    lo, hi = 0, max(budget, 0)
    while lo < hi:                       # largest L with _rs_len(L) <= budget
        mid = (lo + hi + 1) // 2
        if _rs_len(mid, nsym) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def embed_payload(img: np.ndarray, payload: bytes, method: str, password: str = "",
                  strength: str = "medium", encrypted: bool = False,
                  nsym: int | None = None) -> tuple[np.ndarray, EmbedInfo]:
    d = DOMAINS[method]
    n = d.n_slots(img.shape)
    if nsym is None:
        nsym = DEFAULT_NSYM[method]
        if d.robust and payload:
            free0 = n - HEADER_BITS * header_repeats(d, img.shape)
            fits = [k for k in NSYM_CHOICES if _rs_len(len(payload), k) * 8 * MIN_ROBUST_REPEATS <= free0]
            nsym = fits[0] if fits else min(NSYM_CHOICES)
    step = STEPS[method][strength] if d.robust else None
    if n < HEADER_BITS:
        raise StegError(f"image too small for {method} ({n} slots)")

    flags = (VERSION << 4) | (1 if encrypted else 0)
    header = MAGIC + bytes([flags, METHOD_IDS[method], nsym]) + struct.pack(">I", len(payload))[1:]
    hbits = _bits(bytes(RSCodec(HEADER_RS).encode(header)))
    body = bytes(RSCodec(nsym).encode(payload)) if payload else b""
    bbits = _bits(body)

    rh = header_repeats(d, img.shape)
    H = HEADER_BITS * rh
    free = n - H
    if d.robust:
        rb = free // max(len(bbits), 1)
    else:
        rb = 1
    if len(bbits) and (rb < 1 or len(bbits) > free):
        raise StegError(f"message too long for {method} on a {img.shape[1]}x{img.shape[0]} image: "
                        f"max about {capacity(img.shape, method)} bytes")
    if d.robust and len(bbits) and rb < MIN_ROBUST_REPEATS:
        raise StegError(f"message too long for robust {method} embedding on this image "
                        f"(max {capacity(img.shape, method)} bytes); shorten it, use a larger image, or use LSB")

    total = H + rb * len(bbits)
    pos = _positions(d, img.shape, password, total)
    bits = np.concatenate([np.tile(hbits, rh), np.tile(bbits, rb)]) if len(bbits) else np.tile(hbits, rh)
    dz = _dither(d, img.shape, password, pos)
    out = d.write(img, pos, bits, step, dither=dz)

    # verify every slot on the actual 8-bit output
    sc = d.read(out, pos, step, dither=dz)
    bad = int(np.sum((sc <= 0) != (bits == 1)))
    info = EmbedInfo(method, n, rh, len(bbits), rb, len(payload), encrypted, bad)
    return out, info


@dataclass
class Extracted:
    method: str
    payload: bytes | None    # None => watermark detected but too damaged to read
    encrypted: bool
    strength: str | None
    header_ber: float
    body_ber: float          # raw bit error rate of the majority-voted body, before RS


def _vote(scores, nbits, reps):
    return (scores[: nbits * reps].reshape(reps, nbits).mean(axis=0) < 0).astype(np.uint8)


def extract_payload(img: np.ndarray, password: str = "", methods=("LSB", "DCT", "DWT")) -> Extracted | None:
    """Try every method / strength; return the first valid payload or None."""
    for method in methods:
        d = DOMAINS[method]
        if d.n_slots(img.shape) < HEADER_BITS:
            continue
        strengths = ["medium", "low", "high"] if d.robust else [None]
        rh = header_repeats(d, img.shape)
        H = HEADER_BITS * rh
        n = d.n_slots(img.shape)
        for strength in strengths:
            step = STEPS[method][strength] if strength else None
            hpos = _positions(d, img.shape, password, H)
            hsc = d.read(img, hpos, step, dither=_dither(d, img.shape, password, hpos))
            hbits = _vote(hsc, HEADER_BITS, rh)
            try:
                header = bytes(RSCodec(HEADER_RS).decode(_bytes(hbits))[0])
            except ReedSolomonError:
                continue
            if header[:2] != MAGIC or header[3] != METHOD_IDS[method] or (header[2] >> 4) != VERSION:
                continue
            encrypted = bool(header[2] & 1)
            nsym = header[4]
            length = int.from_bytes(b"\x00" + header[5:8], "big")
            hdr_enc = _bits(bytes(RSCodec(HEADER_RS).encode(header)))
            header_ber = float(np.mean(hbits != hdr_enc))
            if length == 0:
                return Extracted(method, b"", encrypted, strength, header_ber, 0.0)
            nb = _rs_len(length, nsym) * 8
            rb = (n - H) // nb if d.robust else 1
            if rb < 1:
                continue
            pos = _positions(d, img.shape, password, H + rb * nb)
            bsc = d.read(img, pos[H:], step, dither=_dither(d, img.shape, password, pos[H:]))
            bbits = _vote(bsc, nb, rb)
            try:
                payload = bytes(RSCodec(nsym).decode(_bytes(bbits))[0])
            except ReedSolomonError:
                # A valid header proves a watermark is present, but the body is too damaged.
                return Extracted(method, None, encrypted, strength, header_ber, float("nan"))
            re = _bits(bytes(RSCodec(nsym).encode(payload)))
            return Extracted(method, payload, encrypted, strength, header_ber, float(np.mean(re != bbits)))
    return None
