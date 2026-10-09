"""Command line interface.

    stegmark hide     cover.png out.png -m "text" [--method DWT] [-p PASSWORD] [--strength medium]
    stegmark reveal   out.png [-p PASSWORD]
    stegmark compare  cover.png out_lsb.png out_dct.png ... [--figure fig.png]
    stegmark test     cover.png -m "text" [-p PASSWORD]      # robustness against attacks
    stegmark capacity cover.png
    stegmark                                                # interactive menu
"""
from __future__ import annotations

import argparse
import csv
import getpass
import os
import sys

import numpy as np

from .api import (hide, load_image, max_message_bytes, mse, psnr, reveal, save_image, ssim)
from .core import StegError

METHODS = ("LSB", "DCT", "DWT")


def _warn_lossy(path, method):
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
        if method == "LSB":
            print("WARNING: LSB does not survive JPEG. Saving as JPEG will destroy the message; use .png.")
        else:
            print("note: saving as JPEG (quality 95). DCT/DWT survive this, but .png keeps the full margin.")


def cmd_hide(a):
    img = load_image(a.input)
    msg = a.message if a.message is not None else open(a.message_file, encoding="utf-8").read()
    stego, info = hide(img, msg, a.method, a.password or "", a.strength)
    _warn_lossy(a.output, info.method)
    save_image(stego, a.output)
    print(f"hidden {len(msg.encode())} bytes with {info.method} in {a.output}")
    print(f"  PSNR {psnr(img, stego):.2f} dB   SSIM {ssim(img, stego):.4f}   "
          f"copies of each bit: {info.body_repeats}   encrypted: {info.encrypted}")
    if info.unreliable_slots:
        print(f"  note: {info.unreliable_slots} of {info.header_repeats * 128 + info.body_repeats * info.body_bits} "
              f"embedded bits did not settle (corrected by repetition + Reed-Solomon)")
    back = reveal(load_image(a.output), a.password or "", info.method)     # verify the file on disk
    if not back.found or back.damaged or back.text != msg:
        print("ERROR: verification failed, the saved file does not read back correctly", file=sys.stderr)
        return 1
    print(f"  verified: {a.output} reads back correctly")
    return 0


def cmd_reveal(a):
    img = load_image(a.input)
    try:
        r = reveal(img, a.password or "", a.method)
    except StegError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not r.found:
        print("no watermark found (not embedded, wrong password, or the image was cropped/rotated/rescaled)")
        return 1
    if r.damaged:
        print(f"a {r.method} watermark is present, but too damaged to read the message")
        return 1
    print(f"method: {r.method}" + (f" ({r.strength})" if r.strength else "")
          + f"   encrypted: {r.encrypted}" + (f"   raw bit errors corrected: {r.body_ber:.1%}" if r.body_ber else ""))
    if a.output:
        open(a.output, "wb").write(r.data)
        print(f"message written to {a.output}")
    else:
        print("message:", r.text if r.text is not None else r.data)
    return 0


def compare(original_path, others, figure=None, csv_path=None):
    orig = load_image(original_path)
    rows = []
    for p in others:
        im = load_image(p)
        if im.shape != orig.shape:
            print(f"skip {p}: size {im.shape} differs from original {orig.shape}")
            continue
        rows.append((os.path.basename(p), mse(orig, im), psnr(orig, im), ssim(orig, im)))
    print(f"{'image':32s} {'MSE':>9s} {'PSNR dB':>8s} {'SSIM':>7s}")
    for n, m, p, s in rows:
        print(f"{n:32s} {m:9.4f} {p:8.2f} {s:7.4f}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image", "mse", "psnr_db", "ssim"]); w.writerows(rows)
    if figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ims = [orig] + [load_image(p) for p in others]
        titles = ["ORIGINAL"] + [f"{os.path.basename(p)}\nPSNR {r[2]:.1f} dB" for p, r in zip(others, rows)]
        fig, ax = plt.subplots(2, len(ims), figsize=(4 * len(ims), 8))
        for k, (im, t) in enumerate(zip(ims, titles)):
            ax[0, k].imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
            ax[0, k].set_title(t, fontsize=11)
            diff = np.abs(im.astype(int) - orig.astype(int))
            diff = diff.max(axis=2) if diff.ndim == 3 else diff
            ax[1, k].imshow(np.minimum(diff * 10, 255), cmap="magma", vmin=0, vmax=255)
            ax[1, k].set_title("difference x10" if k else "(reference)", fontsize=10)
        for a_ in ax.ravel():
            a_.axis("off")
        plt.tight_layout(); plt.savefig(figure, dpi=120); plt.close()
        print(f"figure saved to {figure}")
    return rows


def cmd_compare(a):
    compare(a.original, a.images, a.figure, a.csv)
    return 0


def robustness(img, message, password="", methods=METHODS, quiet=False):
    from .attacks import ATTACKS
    results = {}
    for m in methods:
        try:
            stego, _ = hide(img, message, m, password)
        except StegError as e:
            print(f"{m}: {e}")
            continue
        for name, atk in ATTACKS.items():
            try:
                r = reveal(atk(stego), password, m)
                ok = r.found and not r.damaged and r.data == message.encode()
                status = "OK" if ok else ("damaged" if r.found else "lost")
            except StegError:
                status = "lost"
            results[(m, name)] = status
    if not quiet:
        names = list(ATTACKS)
        print(f"{'attack':26s}" + "".join(f"{m:>9s}" for m in methods))
        for n in names:
            print(f"{n:26s}" + "".join(f"{results.get((m, n), '-'):>9s}" for m in methods))
    return results


def cmd_test(a):
    robustness(load_image(a.input), a.message, a.password or "")
    return 0


def cmd_capacity(a):
    img = load_image(a.input)
    print(f"{a.input}: {img.shape[1]}x{img.shape[0]}, {img.shape[2] if img.ndim == 3 else 1} channel(s)")
    for m in METHODS:
        print(f"  {m}: up to {max_message_bytes(img, m)} bytes"
              f" ({max_message_bytes(img, m, 'x')} with a password)")
    return 0


# ----------------------------------------------------------------- interactive menu
def interactive():
    print("stegmark: hide text in images with LSB, DCT and DWT")
    last = {}
    while True:
        print("\n 1) encode   2) decode   3) compare   4) robustness test   5) capacity   other) quit")
        c = input("choice: ").strip()
        try:
            if c == "1":
                src = input("cover image path: ").strip().strip('"')
                msg = input("message to hide: ")
                pw = getpass.getpass("password (Enter for none): ")
                out_dir = input("output folder [encoded]: ").strip() or "encoded"
                os.makedirs(out_dir, exist_ok=True)
                img = load_image(src)
                base = os.path.splitext(os.path.basename(src))[0]
                last = {"original": src, "encoded": [], "password": pw, "message": msg}
                for m in METHODS:
                    try:
                        stego, info = hide(img, msg, m, pw)
                    except StegError as e:
                        print(f"  {m}: skipped ({e})"); continue
                    p = os.path.join(out_dir, f"{m.lower()}_{base}.png")
                    save_image(stego, p); last["encoded"].append(p)
                    print(f"  {m}: {p}   PSNR {psnr(img, stego):.2f} dB")
            elif c == "2":
                p = input(f"stego image path{' [all from last encode]' if last.get('encoded') else ''}: ").strip().strip('"')
                paths = [p] if p else last.get("encoded", [])
                pw = last.get("password", "") if not p else getpass.getpass("password (Enter for none): ")
                for q in paths:
                    r = reveal(load_image(q), pw)
                    txt = (r.text if r.found and not r.damaged else
                           ("<damaged>" if r.found else "<no watermark found>"))
                    print(f"  {os.path.basename(q)}: [{r.method}] {txt}")
            elif c == "3":
                orig = input(f"original image [{last.get('original', '')}]: ").strip().strip('"') or last.get("original")
                others = input("stego images (comma separated, Enter = last encode): ").strip()
                others = [s.strip().strip('"') for s in others.split(",")] if others else last.get("encoded", [])
                os.makedirs("comparison", exist_ok=True)
                compare(orig, others, "comparison/comparison.png", "comparison/comparison.csv")
            elif c == "4":
                src = input(f"cover image [{last.get('original', '')}]: ").strip().strip('"') or last.get("original")
                msg = input(f"message [{last.get('message', 'test message')}]: ") or last.get("message", "test message")
                robustness(load_image(src), msg, last.get("password", ""))
            elif c == "5":
                src = input("image path: ").strip().strip('"')
                cmd_capacity(argparse.Namespace(input=src))
            else:
                print("bye"); return 0
        except (OSError, StegError) as e:
            print(f"error: {e}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return interactive()
    ap = argparse.ArgumentParser(prog="stegmark", description="Hide text in images (LSB / DCT / DWT).")
    sp = ap.add_subparsers(dest="cmd", required=True)

    h = sp.add_parser("hide", help="embed a message")
    h.add_argument("input"); h.add_argument("output")
    g = h.add_mutually_exclusive_group(required=True)
    g.add_argument("-m", "--message"); g.add_argument("--message-file")
    h.add_argument("--method", default="DWT", type=str.upper, choices=METHODS)
    h.add_argument("-p", "--password", help="keys the positions and AES-GCM encrypts the message")
    h.add_argument("--strength", default="medium", choices=["low", "medium", "high"],
                   help="DCT/DWT only: low = less visible, high = more robust")
    h.set_defaults(fn=cmd_hide)

    r = sp.add_parser("reveal", help="extract a message (method detected automatically)")
    r.add_argument("input"); r.add_argument("-p", "--password")
    r.add_argument("--method", type=str.upper, choices=METHODS)
    r.add_argument("-o", "--output", help="write the message bytes to a file")
    r.set_defaults(fn=cmd_reveal)

    c = sp.add_parser("compare", help="MSE / PSNR / SSIM against the original")
    c.add_argument("original"); c.add_argument("images", nargs="+")
    c.add_argument("--figure"); c.add_argument("--csv")
    c.set_defaults(fn=cmd_compare)

    t = sp.add_parser("test", help="embed with every method and run the attack suite")
    t.add_argument("input"); t.add_argument("-m", "--message", default="stegmark robustness test")
    t.add_argument("-p", "--password")
    t.set_defaults(fn=cmd_test)

    k = sp.add_parser("capacity", help="maximum message size per method")
    k.add_argument("input"); k.set_defaults(fn=cmd_capacity)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except StegError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
