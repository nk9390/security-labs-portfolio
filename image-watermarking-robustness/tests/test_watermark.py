import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from skimage import data
from watermark import METHODS, encode_payload, decode_payload

MSG = b"Karthikey Nori | 2026"
KEY = 1234


@pytest.fixture(scope="module")
def img():
    return (data.camera()).astype(np.uint8)


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("use_rs", [False, True])
def test_roundtrip_no_attack(img, name, use_rs):
    emb, ext = METHODS[name]
    bits = encode_payload(MSG, use_rs)
    out = decode_payload(ext(emb(img, bits, KEY), len(bits), KEY), use_rs)
    assert out is not None and out.rstrip(b"\x00") == MSG


@pytest.mark.parametrize("name", METHODS)
def test_wrong_key_fails(img, name):
    emb, ext = METHODS[name]
    bits = encode_payload(MSG, False)
    out = decode_payload(ext(emb(img, bits, KEY), len(bits), KEY + 1), False)
    assert out.rstrip(b"\x00") != MSG


def test_embedding_is_imperceptible(img):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    bits = encode_payload(MSG, True)
    for name, (emb, _) in METHODS.items():
        assert psnr(img, emb(img, bits, KEY)) > 40, name
