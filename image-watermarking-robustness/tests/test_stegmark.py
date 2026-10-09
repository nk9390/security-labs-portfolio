"""Tests for the stegmark tool (real-image path: any size, colour, auto-detect)."""
import numpy as np
import pytest
import pywt
from skimage import data

from stegmark import StegError, hide, reveal
from stegmark import attacks as A
from stegmark.api import load_image, psnr, save_image
from stegmark.core import DOMAINS

MSG = "Karthikey Nori | owner 2026"
METHODS = ["LSB", "DCT", "DWT"]


def _images():
    rng = np.random.default_rng(0)
    return {
        "gray512": data.camera(),
        "rgb_odd": data.chelsea(),                      # 300x451, not a multiple of 8
        "rgba": data.logo(),                            # alpha channel
        "black": np.zeros((256, 320, 3), np.uint8),     # fully saturated low
        "white": np.full((400, 480), 255, np.uint8),    # fully saturated high
        "noise": rng.integers(0, 256, (300, 400, 3), dtype=np.uint8),
    }


IMAGES = _images()


@pytest.mark.parametrize("name", IMAGES)
@pytest.mark.parametrize("method", METHODS)
def test_roundtrip_autodetect(name, method):
    img = IMAGES[name]
    stego, info = hide(img, MSG, method)
    assert stego.shape == img.shape and stego.dtype == np.uint8
    r = reveal(stego)                                   # no method given: auto-detect
    assert r.found and r.method == method and r.text == MSG


@pytest.mark.parametrize("method", METHODS)
def test_password_and_encryption(method):
    img = data.coffee()
    stego, info = hide(img, MSG, method, password="s3cret")
    assert info.encrypted
    assert reveal(stego, "s3cret").text == MSG
    assert not reveal(stego, "wrong").found            # positions are keyed by the password
    assert not reveal(stego).found


@pytest.mark.parametrize("name", ["gray512", "rgb_odd", "rgba", "black", "noise"])
def test_no_false_positive_on_clean_images(name):
    assert not reveal(IMAGES[name]).found


def test_rgba_alpha_untouched_and_no_colour_shift():
    img = data.logo()
    for m in ("DCT", "DWT"):
        st, _ = hide(img, MSG, m)
        assert np.array_equal(st[..., 3], img[..., 3])
        # luma-only embedding: R, G and B move by (almost) the same amount -> no tint
        d = st[..., :3].astype(int) - img[..., :3].astype(int)
        unclipped = (img[..., :3] > 30).all(-1) & (img[..., :3] < 225).all(-1)
        spread = (d.max(-1) - d.min(-1))[unclipped]
        assert spread.max() <= 1


@pytest.mark.parametrize("method", ["DCT", "DWT"])
@pytest.mark.parametrize("attack", ["JPEG q=75", "JPEG q=50", "Noise sigma=3", "Resize 0.5x round trip",
                                    "Brightness +20"])
def test_robust_methods_survive(method, attack):
    img = data.coffee()
    st, _ = hide(img, MSG, method)
    assert reveal(A.ATTACKS[attack](st), method=method).text == MSG


def test_lsb_is_fragile_to_jpeg():
    st, _ = hide(data.coffee(), MSG, "LSB")
    assert reveal(A.jpeg(st, 90), method="LSB").text != MSG


def test_message_too_long():
    with pytest.raises(StegError):
        hide(data.chelsea(), "x" * 5000, "DWT")


def test_long_message_lsb_and_unicode():
    msg = ("तस्वीर में छुपा संदेश ✓ " * 200)
    st, _ = hide(data.astronaut(), msg, "LSB")
    assert reveal(st).text == msg


def test_empty_message():
    st, _ = hide(data.coffee(), "", "DCT")
    r = reveal(st)
    assert r.found and r.text == ""


def test_haar_pattern_matches_pywt():
    rng = np.random.default_rng(1)
    Y = rng.uniform(0, 255, (64, 96))
    ref = pywt.wavedec2(Y, "haar", level=2)[1][0].ravel()
    assert np.allclose(DOMAINS["DWT"].coeffs(Y), ref)


def test_dct_pattern_matches_scipy():
    from scipy.fft import dctn
    rng = np.random.default_rng(2)
    Y = rng.uniform(0, 255, (32, 40))
    blocks = Y.reshape(4, 8, 5, 8).transpose(0, 2, 1, 3)
    ref = dctn(blocks, axes=(2, 3), norm="ortho")[:, :, 1, 2].ravel()
    assert np.allclose(DOMAINS["DCT"].coeffs(Y), ref)


def test_imperceptible():
    img = data.coffee()
    for m, floor in (("LSB", 60), ("DCT", 42), ("DWT", 40)):
        st, _ = hide(img, MSG, m)
        assert psnr(img, st) > floor, m


def test_file_roundtrip_png_and_jpeg(tmp_path):
    img = data.rocket()
    st, _ = hide(img, MSG, "DWT")
    for ext in ("png", "jpg"):
        p = str(tmp_path / f"s.{ext}")
        save_image(st, p)
        assert reveal(load_image(p)).text == MSG


def test_cli(tmp_path, capsys):
    from stegmark.cli import main
    src, out = str(tmp_path / "c.png"), str(tmp_path / "o.png")
    save_image(data.coffee(), src)
    assert main(["hide", src, out, "-m", MSG, "--method", "dct", "-p", "pw"]) == 0
    assert main(["reveal", out, "-p", "pw"]) == 0
    assert MSG in capsys.readouterr().out
    assert main(["reveal", src]) == 1
