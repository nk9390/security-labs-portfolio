# Robust Image Watermarking: LSB vs DCT vs DWT with Reed-Solomon

A small, reproducible study of how well three embedding domains keep a hidden ownership message alive when the image is attacked. All three methods hide the same 32-byte message with the same redundancy and the same secret key, so the comparison is like for like.

**What is implemented (Python, `src/watermark.py`)**
- **LSB** (spatial): least significant bit of keyed pseudo-random pixels.
- **DCT**: quantisation index modulation (QIM) on one low-mid-frequency coefficient (1,2) of keyed 8x8 blocks.
- **DWT**: QIM on the level-2 Haar HL sub-band at keyed positions.
- **Reed-Solomon** (32 parity bytes) as an optional error-correcting layer; each coded bit is repeated 8x and recovered by majority vote.
- **Attacks** (`src/attacks.py`): JPEG (q = 90/70/50), Gaussian noise (sigma 2 and 5), 0.5x resize round trip, 3x3 median filter, blanking 25% of the area.
- **Metrics:** PSNR, SSIM, bit error rate (BER) before error correction, and exact message recovery.

## Results (4 held-out 512x512 grayscale images)
The design (which DCT coefficient, step size) was tuned on two other images, and every number below comes from four images the tuning never saw. Full tables are in [`results/results.md`](results/results.md) and the raw data in `results/results.csv`.

| Method | PSNR (dB) | SSIM | JPEG q=50 | Noise s=5 | Resize 0.5x | Median 3x3 | Blank 25% |
|---|---|---|---|---|---|---|---|
| LSB | 69.1 | 0.9999 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| DCT (step 32) | 46.8 | 0.9866 | 4/4 | 4/4 | 4/4 | 1/4 | 0/4 |
| DWT (step 40) | 44.9 | 0.9801 | 4/4 | 4/4 | 4/4 | 3/4 | 0/4 |

*Cells show images where the exact message was recovered, with Reed-Solomon on.*

**Findings**
- LSB is nearly invisible but is destroyed by any re-compression or noise (BER about 0.4 to 0.5, which is chance).
- DCT and DWT survive JPEG down to q=50, noise and resizing with 0 bit errors before error correction. DWT is more robust to the median filter, at slightly lower PSNR.
- Imperceptibility and robustness trade off through the QIM step size (see `results/tradeoff.png`).
- Reed-Solomon mattered for resizing and median filtering, where it turned a partly corrupted payload into an exact recovery.
- My first DCT design used a higher-frequency coefficient (3,4) and failed under JPEG q<=70, because JPEG's quantiser removes it. Moving to (1,2) fixed that. This is why the tuning/held-out split exists.

**Limitations**
- Only 4 held-out images, so recovery fractions move in steps of 0.25. Treat the numbers as indicative, not statistically strong.
- No geometric attacks (rotation, cropping and rescaling with resynchronisation) and no steganalysis tests.
- Blanking 25% of the image defeats all three methods; handling it needs more redundancy or a synchronisation scheme.
- Grayscale images only, fixed 32-byte payload.

## Run it
```
pip install -r requirements.txt
python -m pytest -q tests          # 10 tests: round trips, wrong-key failure, PSNR floor
python src/run_experiments.py      # regenerates everything in results/
```

## Layout
```
src/watermark.py        embedding / extraction (LSB, DCT, DWT) + Reed-Solomon payload
src/attacks.py          attack functions
src/run_experiments.py  evaluation, tables, figures
tests/                  pytest suite
results/                results.md, results.csv, recovery_by_attack.png, tradeoff.png, example.png
```

## Background
This follows up the team course project [LSB, DCT and DWT image data hiding](../steganography-watermarking/team-report/) (VIT, 2025), which compared the three methods on PSNR and capacity only. This repository is my own re-implementation, extending it with keyed embedding, error correction and attack testing.

*Development note: written with assistance from Claude (Anthropic); I ran, tested and can explain all of the code.*
