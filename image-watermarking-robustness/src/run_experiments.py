"""Run the full evaluation and write results/ (CSV, markdown tables, figures)."""
import sys, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image
from skimage import data, color
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import watermark as wm
from attacks import ATTACKS

MSG = b"Karthikey Nori | owner 2026"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)


def load_images():
    names = ["astronaut", "camera", "coffee", "chelsea", "rocket", "moon"]
    out = {}
    for n in names:
        a = getattr(data, n)()
        g = (color.rgb2gray(a) * 255).round().astype(np.uint8) if a.ndim == 3 else a
        out[n] = np.array(Image.fromarray(g).resize((512, 512), Image.BICUBIC))
    return out


def trial(img, method, use_rs, key, attack, step=None):
    emb, ext = wm.METHODS[method]
    kw = {} if step is None or method == "LSB" else {"step": step}
    bits = wm.encode_payload(MSG, use_rs)
    stego = emb(img, bits, key, **kw)
    got = ext(attack(stego), len(bits), key, **kw)
    ber = float(np.mean(got != bits))
    dec = wm.decode_payload(got, use_rs)
    ok = dec is not None and dec.rstrip(b"\x00") == MSG
    return stego, ber, ok


def main():
    # astronaut + camera were used to tune the DCT coefficient/step; all reported numbers use the other four.
    imgs = {k: v for k, v in load_images().items() if k not in ("astronaut", "camera")}
    rows = []
    for method in wm.METHODS:
        for use_rs in (False, True):
            for aname, atk in ATTACKS.items():
                bers, oks, ps, ss = [], [], [], []
                for i, (n, im) in enumerate(imgs.items()):
                    stego, ber, ok = trial(im, method, use_rs, 1000 + i, atk)
                    bers.append(ber); oks.append(ok)
                    ps.append(psnr(im, stego)); ss.append(ssim(im, stego, data_range=255))
                rows.append(dict(method=method, rs="RS" if use_rs else "none", attack=aname,
                                 psnr=np.mean(ps), ssim=np.mean(ss), ber=np.mean(bers),
                                 recovered=np.mean(oks)))
    with open(RES / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        for r in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # --- markdown tables ---
    md = ["# Results\n", f"{len(imgs)} held-out grayscale 512x512 images ({', '.join(imgs)}); the design was tuned on two other images (astronaut, camera). Payload = {wm.MSG_LEN} bytes, "
          f"{wm.N_SLOTS} keyed embedding slots, RS parity = {wm.RS_SYMBOLS} bytes. "
          "Numbers are means over the held-out images. With only 4 images, recovery fractions move in steps of 0.25.\n"]
    q = {(r['method'], r['rs'], r['attack']): r for r in rows}
    md.append("## Imperceptibility (no attack)\n\n| Method | PSNR (dB) | SSIM |\n|---|---|---|")
    for m in wm.METHODS:
        r = q[(m, "none", "none")]; md.append(f"| {m} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
    for rs in ("none", "RS"):
        md.append(f"\n## Message recovered exactly, ECC = {rs} (fraction of images)\n")
        md.append("| Attack | " + " | ".join(wm.METHODS) + " |\n|---|" + "---|" * len(wm.METHODS))
        for a in ATTACKS:
            md.append(f"| {a} | " + " | ".join(f"{q[(m, rs, a)]['recovered']:.2f}" for m in wm.METHODS) + " |")
    md.append("\n## Bit error rate before ECC (lower is better)\n")
    md.append("| Attack | " + " | ".join(wm.METHODS) + " |\n|---|" + "---|" * len(wm.METHODS))
    for a in ATTACKS:
        md.append(f"| {a} | " + " | ".join(f"{q[(m, 'none', a)]['ber']:.3f}" for m in wm.METHODS) + " |")

    # --- step-size sweep: imperceptibility vs robustness (JPEG q=70, RS on) ---
    md.append("\n## Step-size sweep (JPEG q=70, RS on)\n\n| Method | step | PSNR (dB) | recovered |\n|---|---|---|---|")
    sweep = {}
    for m, steps in (("DCT", [8, 16, 24, 32, 48]), ("DWT", [10, 20, 40, 60, 80])):
        for s in steps:
            ps, oks = [], []
            for i, (n, im) in enumerate(imgs.items()):
                st, _, ok = trial(im, m, True, 2000 + i, ATTACKS["JPEG q=70"], step=s)
                ps.append(psnr(im, st)); oks.append(ok)
            sweep[(m, s)] = (np.mean(ps), np.mean(oks))
            md.append(f"| {m} | {s} | {np.mean(ps):.2f} | {np.mean(oks):.2f} |")
    (RES / "results.md").write_text("\n".join(md) + "\n")

    # --- figures ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for k, rs in enumerate(("none", "RS")):
        x = np.arange(len(ATTACKS)); wdt = 0.27
        for j, m in enumerate(wm.METHODS):
            ax[k].bar(x + (j - 1) * wdt, [q[(m, rs, a)]["recovered"] for a in ATTACKS], wdt, label=m)
        ax[k].set_xticks(x); ax[k].set_xticklabels(list(ATTACKS), rotation=60, ha="right", fontsize=8)
        ax[k].set_title(f"ECC: {rs}"); ax[k].set_ylim(0, 1.05)
    ax[0].set_ylabel("fraction of images with exact recovery"); ax[0].legend()
    plt.tight_layout(); plt.savefig(RES / "recovery_by_attack.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in ("DCT", "DWT"):
        pts = sorted((v[0], v[1], s) for (mm, s), v in sweep.items() if mm == m)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=m)
        for p in pts: ax.annotate(str(p[2]), (p[0], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PSNR (dB)"); ax.set_ylabel("recovered after JPEG q=70"); ax.legend()
    ax.set_title("Imperceptibility vs robustness (label = step)"); plt.tight_layout()
    plt.savefig(RES / "tradeoff.png", dpi=150); plt.close()

    # --- visual example ---
    im = imgs["coffee"]; bits = wm.encode_payload(MSG, True)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(im, cmap="gray"); ax[0].set_title("cover")
    for k, m in enumerate(wm.METHODS, 1):
        s = wm.METHODS[m][0](im, bits, 7)
        ax[k].imshow(np.abs(s.astype(int) - im) * 20, cmap="magma", vmin=0, vmax=255)
        ax[k].set_title(f"{m} difference x20\nPSNR {psnr(im, s):.1f} dB")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(RES / "example.png", dpi=150); plt.close()
    print("\n".join(md))


if __name__ == "__main__":
    main()
