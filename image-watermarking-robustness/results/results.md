# Results

4 held-out grayscale 512x512 images (coffee, chelsea, rocket, moon); the design was tuned on two other images (astronaut, camera). Payload = 32 bytes, 4096 keyed embedding slots, RS parity = 32 bytes. Numbers are means over the held-out images. With only 4 images, recovery fractions move in steps of 0.25.

## Imperceptibility (no attack)

| Method | PSNR (dB) | SSIM |
|---|---|---|
| LSB | 69.14 | 0.9999 |
| DCT | 46.83 | 0.9866 |
| DWT | 44.93 | 0.9801 |

## Message recovered exactly, ECC = none (fraction of images)

| Attack | LSB | DCT | DWT |
|---|---|---|---|
| none | 1.00 | 1.00 | 1.00 |
| JPEG q=90 | 0.00 | 1.00 | 1.00 |
| JPEG q=70 | 0.00 | 1.00 | 1.00 |
| JPEG q=50 | 0.00 | 1.00 | 1.00 |
| Gaussian noise s=2 | 0.00 | 1.00 | 1.00 |
| Gaussian noise s=5 | 0.00 | 1.00 | 1.00 |
| Resize 0.5x | 0.00 | 0.75 | 1.00 |
| Median 3x3 | 0.00 | 0.25 | 0.75 |
| Blank 25% area | 0.25 | 0.25 | 0.00 |

## Message recovered exactly, ECC = RS (fraction of images)

| Attack | LSB | DCT | DWT |
|---|---|---|---|
| none | 1.00 | 1.00 | 1.00 |
| JPEG q=90 | 0.00 | 1.00 | 1.00 |
| JPEG q=70 | 0.00 | 1.00 | 1.00 |
| JPEG q=50 | 0.00 | 1.00 | 1.00 |
| Gaussian noise s=2 | 0.00 | 1.00 | 1.00 |
| Gaussian noise s=5 | 0.00 | 1.00 | 1.00 |
| Resize 0.5x | 0.00 | 1.00 | 1.00 |
| Median 3x3 | 0.00 | 0.25 | 0.75 |
| Blank 25% area | 0.00 | 0.00 | 0.00 |

## Bit error rate before ECC (lower is better)

| Attack | LSB | DCT | DWT |
|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 |
| JPEG q=90 | 0.477 | 0.000 | 0.000 |
| JPEG q=70 | 0.481 | 0.000 | 0.000 |
| JPEG q=50 | 0.413 | 0.000 | 0.000 |
| Gaussian noise s=2 | 0.482 | 0.000 | 0.000 |
| Gaussian noise s=5 | 0.463 | 0.000 | 0.000 |
| Resize 0.5x | 0.472 | 0.001 | 0.000 |
| Median 3x3 | 0.266 | 0.050 | 0.008 |
| Blank 25% area | 0.008 | 0.006 | 0.023 |

## Step-size sweep (JPEG q=70, RS on)

| Method | step | PSNR (dB) | recovered |
|---|---|---|---|
| DCT | 8 | 57.99 | 0.00 |
| DCT | 16 | 52.24 | 1.00 |
| DCT | 24 | 48.96 | 1.00 |
| DCT | 32 | 46.54 | 1.00 |
| DCT | 48 | 43.13 | 1.00 |
| DWT | 10 | 56.65 | 0.00 |
| DWT | 20 | 51.15 | 0.75 |
| DWT | 40 | 44.59 | 1.00 |
| DWT | 60 | 41.17 | 1.00 |
| DWT | 80 | 38.46 | 1.00 |
